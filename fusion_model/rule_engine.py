"""RuleGenerator — ephemeral corrections *and* persistent rule proposals.

**Concept:**  The RuleGenerator serves two roles:

1. **Ephemeral correction** — a small MLP takes the shared embedding ``h``
   and outputs a one-shot low-rank correction (``A_new @ (B_new @ h)``) plus
   a confidence scalar.  This correction is used for the current forward pass
   only.

2. **Rule proposal** — the generator maintains a circular history buffer of
   recent ``(h, decision, outcome)`` triples.  The proposal pipeline has
   three stages of cross-attention that operate *entirely on history* — the
   current input is deliberately excluded so that proposed rules reflect
   what actually worked rather than what the model is currently looking at:

   a. **Input→Decision cross-attention** — historical input embeddings
      query over historical decision embeddings (projected to
      ``embed_dim``) to discover which inputs led to which decisions.
   b. **Outcome cross-attention** — the input→decision attended result
      queries over historical outcome embeddings (a learned projection
      of the scalar loss signal) to identify which input→decision
      pairings produced good or bad outcomes.
   c. **Synthesis cross-attention** — a learned query token attends over
      the two attended representations ``[attended_input_dec,
      attended_outcome]`` to dynamically weight the two context sources,
      producing a single ``embed_dim`` vector that is projected to the
      proposed rule parameters ``(key, A, B)``.

   The proposal also outputs a **soft commit weight** computed as
   ``sigmoid((similarity - threshold) * temperature)`` where *similarity*
   is the cosine similarity between the proposed key and the mean
   historical embedding, *threshold* is a learnable scalar, and
   *temperature* controls sharpness.

This two-tier design lets the model invent hypotheses on the fly *and*
distill recurring patterns into the long-term rule bank.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RuleGenerator(nn.Module):
    """Generate ephemeral low-rank corrections and propose persistent rules.

    The rule proposer uses a **three-stage cross-attention** pipeline that
    operates entirely on the history buffer — no current-input dependency:

    1. Historical inputs cross-attend over historical decisions to learn
       which inputs led to which decisions.
    2. That result cross-attends over historical outcome signals to learn
       which input→decision pairings were effective.
    3. A learned synthesis query attends over the two attended
       representations to produce the final rule proposal.

    The proposal includes a **soft commit weight** derived from the cosine
    similarity between the proposed key and the mean historical embedding,
    a learnable threshold, and a learnable temperature, enabling partial
    blending into the target memory slot.

    :param embed_dim: Dimensionality of the shared input embedding.
    :param num_classes: Number of output classes.
    :param rank: Inner rank of the generated low-rank matrices.
    :param hidden_dim: Width of the internal MLP.
    :param history_size: Capacity of the circular ``(h, decision, outcome)``
        buffer.
    :param min_history: Minimum entries in the buffer before rule proposal
        is attempted.
    :param decision_embed_dim: Embedding dimension for stored decision
        indices before projection to ``embed_dim``.
    :param outcome_proj_dim: Hidden dimension for projecting scalar outcome
        signals to ``embed_dim``.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_classes: int = 9000,
        rank: int = 8,
        hidden_dim: int = 256,
        history_size: int = 512,
        min_history: int = 64,
        decision_embed_dim: int = 32,
        outcome_proj_dim: int = 64,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.rank = rank
        self.num_classes = num_classes
        self.history_size = history_size
        self.min_history = min_history
        self.decision_embed_dim = decision_embed_dim

        # ── Ephemeral correction MLP  ────────────────────────
        out_dim = (rank * embed_dim) + (embed_dim * rank) + 1
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, out_dim),
        )
        self.head = nn.Linear(embed_dim, num_classes)

        # ── History buffer (non-gradient circular buffer) ────────────────
        self.register_buffer("history_h", torch.zeros(history_size, embed_dim))
        self.register_buffer(
            "history_decisions", torch.full((history_size,), -1, dtype=torch.long),
        )
        self.register_buffer("history_outcomes", torch.zeros(history_size))
        self.register_buffer("history_ptr", torch.tensor(0, dtype=torch.long))
        self.register_buffer("history_count", torch.tensor(0, dtype=torch.long))

        # ── Rule proposer ────────────────────────────────────────────────
        self.decision_embed = nn.Embedding(num_classes, decision_embed_dim)

        # Stage 1: historical inputs query over historical decisions.
        self.input_dec_cross_attn = nn.MultiheadAttention(
            embed_dim, num_heads=4, batch_first=True,
        )
        self.dec_proj = nn.Linear(decision_embed_dim, embed_dim)

        # Stage 2: input→decision result queries over outcome embeddings.
        self.outcome_proj = nn.Sequential(
            nn.Linear(1, outcome_proj_dim),
            nn.GELU(),
            nn.Linear(outcome_proj_dim, embed_dim),
        )
        self.outcome_cross_attn = nn.MultiheadAttention(
            embed_dim, num_heads=4, batch_first=True,
        )

        # Stage 3: learned synthesis query attends over the two attended results.
        self.synthesis_query = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.synthesis_cross_attn = nn.MultiheadAttention(
            embed_dim, num_heads=4, batch_first=True,
        )

        # Light projection from the synthesis output to rule parameters.
        proposer_out = embed_dim + (embed_dim * rank) + (rank * embed_dim)
        self.rule_proj = nn.Linear(embed_dim, proposer_out)

        # Learnable commitment threshold and temperature for the soft
        # commit weight: sigmoid((cosine_sim - threshold) * temperature).
        self.commit_threshold_logit = nn.Parameter(torch.tensor(0.0))
        self.commit_temperature = nn.Parameter(torch.tensor(1.0))

    # ── History management ───────────────────────────────────────────────

    @torch.no_grad()
    def update_history(
        self,
        h: torch.Tensor,
        decisions: torch.Tensor,
        outcomes: torch.Tensor,
    ) -> None:
        """Append a batch of ``(h, decision, outcome)`` triples to the buffer.

        :param h: Shared embeddings ``(batch, embed_dim)`` — will be detached.
        :param decisions: Predicted class indices ``(batch,)``.
        :param outcomes: Per-sample outcome signal ``(batch,)`` — typically
            the per-sample loss (lower = better).
        """
        batch = h.shape[0]
        h_det = h.detach()
        dec_det = decisions.detach()
        out_det = outcomes.detach()

        ptr = self.history_ptr.item()
        end = ptr + batch

        if end <= self.history_size:
            self.history_h[ptr:end] = h_det
            self.history_decisions[ptr:end] = dec_det
            self.history_outcomes[ptr:end] = out_det
        else:
            first = self.history_size - ptr
            self.history_h[ptr:] = h_det[:first]
            self.history_decisions[ptr:] = dec_det[:first]
            self.history_outcomes[ptr:] = out_det[:first]
            self.history_h[: end - self.history_size] = h_det[first:]
            self.history_decisions[: end - self.history_size] = dec_det[first:]
            self.history_outcomes[: end - self.history_size] = out_det[first:]

        self.history_ptr.fill_(end % self.history_size)
        self.history_count.fill_(
            min(self.history_count.item() + batch, self.history_size),
        )

    # ── Rule proposal ────────────────────────────────────────────────────

    def propose_rule(self, batch_size: int) -> dict[str, torch.Tensor] | None:
        """Propose a persistent rule from historical context only.

        Three-stage cross-attention pipeline operating on history:

        1. Historical input embeddings query over historical decision
           embeddings — learning which inputs led to which decisions.
        2. The input→decision attended result queries over historical
           outcome embeddings — learning which pairings were effective.
        3. A learned synthesis query attends over the two attended
           representations to produce the final rule parameters.

        The current input ``h`` is deliberately excluded: rules should
        reflect what *worked* historically, not what the model is
        currently looking at.

        Returns ``None`` when the history buffer has fewer than
        ``min_history`` entries.

        :param batch_size: Number of proposals to generate (one per batch
            element, all derived from the same history).
        :return: Dict with ``key``, ``A``, ``B``, ``commit_weight`` (per
            batch element), or ``None``.
        """
        count = self.history_count.item()
        if count < self.min_history:
            return None

        valid_h = self.history_h[:count].clone()        # (count, embed_dim)
        valid_dec = self.history_decisions[:count].clone()  # (count,)
        valid_out = self.history_outcomes[:count].clone()   # (count,)

        # Expand history to batch dimension — every batch element sees the
        # same history but the attention weights are independent.
        hist_h = valid_h.unsqueeze(0).expand(batch_size, -1, -1)  # (B, count, E)

        # Stage 1: historical inputs cross-attend over historical decisions.
        dec_emb = self.dec_proj(self.decision_embed(valid_dec))  # (count, E)
        kv_dec = dec_emb.unsqueeze(0).expand(batch_size, -1, -1)  # (B, count, E)
        attended_input_dec, _ = self.input_dec_cross_attn(hist_h, kv_dec, kv_dec)
        # (B, count, E) — each historical input position now carries
        # information about what decisions were associated with similar inputs.

        # Stage 2: input→decision result cross-attends over outcome embeddings.
        outcome_emb = self.outcome_proj(valid_out.unsqueeze(-1))  # (count, E)
        kv_out = outcome_emb.unsqueeze(0).expand(batch_size, -1, -1)  # (B, count, E)
        attended_outcome, _ = self.outcome_cross_attn(
            attended_input_dec, kv_out, kv_out,
        )
        # (B, count, E) — now carries input→decision→outcome associations.

        # Pool the two attended sequences to single vectors.
        attended_input_dec_pooled = attended_input_dec.mean(dim=1)  # (B, E)
        attended_outcome_pooled = attended_outcome.mean(dim=1)      # (B, E)

        # Stage 3: learned synthesis query attends over the two pooled results.
        context_tokens = torch.stack(
            [attended_input_dec_pooled, attended_outcome_pooled], dim=1,
        )  # (B, 2, E)
        syn_query = self.synthesis_query.expand(batch_size, -1, -1)  # (B, 1, E)
        synthesised, _ = self.synthesis_cross_attn(
            syn_query, context_tokens, context_tokens,
        )
        synthesised = synthesised.squeeze(1)  # (B, E)

        raw = self.rule_proj(synthesised)  # (B, proposer_out)

        # Unpack proposed rule parameters.
        e, r = self.embed_dim, self.rank
        key = raw[:, :e]                                 # (B, E)
        a_flat = raw[:, e : e + e * r]
        b_flat = raw[:, e + e * r : e + 2 * e * r]

        A = a_flat.view(-1, e, r)   # (B, E, rank)
        B = b_flat.view(-1, r, e)   # (B, rank, E)

        # Soft commit weight: cosine similarity between proposed key and
        # the mean historical embedding (since we have no current input).
        mean_hist = valid_h.mean(dim=0, keepdim=True).expand(batch_size, -1)
        similarity = nn.functional.cosine_similarity(key, mean_hist, dim=-1)
        threshold = torch.sigmoid(self.commit_threshold_logit)
        temperature = self.commit_temperature.clamp(min=0.01)
        commit_weight = torch.sigmoid(
            (similarity - threshold) * temperature,
        ).unsqueeze(-1)  # (B, 1)

        return {
            "key": key,
            "A": A,
            "B": B,
            "commit_weight": commit_weight,
        }

    # ── Forward pass ─────────────────────────────────────────────────────

    def forward(
        self, h: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor] | None]:
        """Produce ephemeral rule logits, confidence, correction, and a rule proposal.

        :param h: Shared embedding ``(batch, embed_dim)``.
        :return: Tuple ``(logits_rule, confidence, correction, proposal)``
            where *correction* is the ephemeral low-rank correction vector
            ``(batch, embed_dim)`` before the classification head (used by
            the :class:`~fusion_model.decision.DecisionRouter`), and
            *proposal* is the dict from :meth:`propose_rule` (or ``None``
            if insufficient history).
        """
        batch = h.shape[0]
        params = self.mlp(h)

        b_size = self.rank * self.embed_dim
        a_size = self.embed_dim * self.rank
        b_flat = params[:, :b_size]
        a_flat = params[:, b_size : b_size + a_size]
        confidence = torch.sigmoid(params[:, -1:])

        b_mat = b_flat.view(batch, self.rank, self.embed_dim)
        a_mat = a_flat.view(batch, self.embed_dim, self.rank)

        compressed = torch.bmm(b_mat, h.unsqueeze(-1)).squeeze(-1)
        correction = torch.bmm(a_mat, compressed.unsqueeze(-1)).squeeze(-1)

        logits_rule = self.head(correction)

        proposal = self.propose_rule(batch)

        return logits_rule, confidence, correction, proposal
