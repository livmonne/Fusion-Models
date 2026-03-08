"""RuleGenerator — ephemeral corrections *and* persistent rule proposals.

**Concept:**  The RuleGenerator serves two roles:

1. **Ephemeral correction** — a small MLP takes the shared embedding ``h``
   and outputs a one-shot low-rank correction (``A_new @ (B_new @ h)``) plus
   a confidence scalar.  This correction is used for the current forward pass
   only.

2. **Rule proposal** — the generator maintains a circular history buffer of
   recent ``(h, prediction)`` pairs.  The proposal pipeline has three stages
   of cross-attention:

   a. **History cross-attention** — ``h`` queries over past embeddings to
      extract relevant historical context.
   b. **Decision cross-attention** — ``h`` queries over past decision
      embeddings (projected to ``embed_dim``) to extract relevant decision
      context.
   c. **Synthesis cross-attention** — ``h`` queries over the three-token
      sequence ``[h, attended_h, attended_dec]`` to dynamically weight
      which context source matters most for the current proposal, producing
      a single ``embed_dim`` vector that is projected to the proposed rule
      parameters ``(key, A, B)``.

   The proposal also outputs a **soft commit weight** computed as
   ``sigmoid((similarity - threshold) * temperature)`` where *similarity*
   is the cosine similarity between the proposed key and ``h``,
   *threshold* is a learnable scalar, and *temperature* controls
   sharpness.  This allows partial blending of the proposed rule into
   the target memory slot rather than a hard overwrite.

This two-tier design lets the model invent hypotheses on the fly *and*
distill recurring patterns into the long-term rule bank.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RuleGenerator(nn.Module):
    """Generate ephemeral low-rank corrections and propose persistent rules.

    The rule proposer uses a **three-stage cross-attention** pipeline over
    the circular history buffer:

    1. History cross-attention reads out past embeddings.
    2. Decision cross-attention reads out past decision embeddings
       (projected to ``embed_dim``).
    3. Synthesis cross-attention attends over the three-token sequence
       ``[h, attended_h, attended_dec]`` using ``h`` as the query,
       dynamically weighting which context source is most relevant for
       the current proposal.

    The proposal includes a **soft commit weight** derived from the cosine
    similarity between the proposed key and ``h``, a learnable threshold,
    and a learnable temperature, enabling partial blending into the target
    memory slot.

    :param embed_dim: Dimensionality of the shared input embedding.
    :param num_classes: Number of output classes.
    :param rank: Inner rank of the generated low-rank matrices.
    :param hidden_dim: Width of the internal MLP.
    :param history_size: Capacity of the circular ``(h, decision)`` buffer.
    :param min_history: Minimum entries in the buffer before rule proposal
        is attempted.
    :param decision_embed_dim: Embedding dimension for stored decision
        indices before projection to ``embed_dim``.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_classes: int = 28,
        rank: int = 8,
        hidden_dim: int = 256,
        history_size: int = 512,
        min_history: int = 64,
        decision_embed_dim: int = 32,
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
        self.register_buffer("history_decisions", torch.full((history_size,), -1, dtype=torch.long))
        self.register_buffer("history_ptr", torch.tensor(0, dtype=torch.long))
        self.register_buffer("history_count", torch.tensor(0, dtype=torch.long))

        # ── Rule proposer ────────────────────────────────────────────────
        self.decision_embed = nn.Embedding(num_classes, decision_embed_dim)

        # Cross-attention over history embeddings: query = h, key/value = valid_h.
        self.history_cross_attn = nn.MultiheadAttention(
            embed_dim, num_heads=4, batch_first=True,
        )
        # Cross-attention over decision embeddings: query = h, key/value = dec_emb
        # projected to embed_dim so the attention heads have full width.
        self.dec_proj = nn.Linear(decision_embed_dim, embed_dim)
        self.dec_cross_attn = nn.MultiheadAttention(
            embed_dim, num_heads=4, batch_first=True,
        )

        # Synthesis cross-attention: query = h, keys/values = [h, attended_h, attended_dec].
        # Dynamically weights which context source matters most per input.
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
    def update_history(self, h: torch.Tensor, decisions: torch.Tensor) -> None:
        """Append a batch of ``(h, decision)`` pairs to the circular buffer.

        :param h: Shared embeddings ``(batch, embed_dim)`` — will be detached.
        :param decisions: Predicted class indices ``(batch,)``.
        """
        batch = h.shape[0]
        h_det = h.detach()
        dec_det = decisions.detach()

        ptr = self.history_ptr.item()
        end = ptr + batch

        if end <= self.history_size:
            self.history_h[ptr:end] = h_det
            self.history_decisions[ptr:end] = dec_det
        else:
            first = self.history_size - ptr
            self.history_h[ptr:] = h_det[:first]
            self.history_decisions[ptr:] = dec_det[:first]
            self.history_h[: end - self.history_size] = h_det[first:]
            self.history_decisions[: end - self.history_size] = dec_det[first:]

        self.history_ptr.fill_(end % self.history_size)
        self.history_count.fill_(min(self.history_count.item() + batch, self.history_size))

    # ── Rule proposal ────────────────────────────────────────────────────

    def propose_rule(self, h: torch.Tensor) -> dict[str, torch.Tensor] | None:
        """Propose a persistent rule from current input + history context.

        Three-stage cross-attention pipeline:

        1. ``h`` queries over past embeddings (history cross-attention).
        2. ``h`` queries over past decision embeddings (decision cross-attention).
        3. ``h`` queries over the three-token sequence
           ``[h, attended_h, attended_dec]`` (synthesis cross-attention),
           dynamically weighting which context source is most relevant.

        The synthesis output is projected to the proposed rule parameters
        ``(key, A, B)``.  A **soft commit weight** is computed from the
        cosine similarity between the proposed key and ``h``, passed through
        ``sigmoid((similarity - threshold) * temperature)``.

        Returns ``None`` when the history buffer has fewer than
        ``min_history`` entries.

        :param h: Shared embedding ``(batch, embed_dim)``.
        :return: Dict with ``key``, ``A``, ``B``, ``commit_weight`` (per
            batch element), or ``None``.
        """
        count = self.history_count.item()
        if count < self.min_history:
            return None

        batch = h.shape[0]
        valid_h = self.history_h[:count].clone()  # (count, embed_dim)
        valid_dec = self.history_decisions[:count].clone()  # (count,)

        query = h.unsqueeze(1)  # (batch, 1, embed_dim)

        # Stage 1: cross-attend over history embeddings.
        kv_h = valid_h.unsqueeze(0).expand(batch, -1, -1)  # (batch, count, embed_dim)
        attended_h, _ = self.history_cross_attn(query, kv_h, kv_h)
        attended_h = attended_h.squeeze(1)  # (batch, embed_dim)

        # Stage 2: cross-attend over decision embeddings (projected to embed_dim).
        dec_emb = self.dec_proj(self.decision_embed(valid_dec))  # (count, embed_dim)
        kv_dec = dec_emb.unsqueeze(0).expand(batch, -1, -1)  # (batch, count, embed_dim)
        attended_dec, _ = self.dec_cross_attn(query, kv_dec, kv_dec)
        attended_dec = attended_dec.squeeze(1)  # (batch, embed_dim)

        # Stage 3: synthesis cross-attention over [h, attended_h, attended_dec].
        context_tokens = torch.stack(
            [h, attended_h, attended_dec], dim=1,
        )  # (batch, 3, embed_dim)
        synthesised, _ = self.synthesis_cross_attn(query, context_tokens, context_tokens)
        synthesised = synthesised.squeeze(1)  # (batch, embed_dim)

        raw = self.rule_proj(synthesised)  # (batch, proposer_out)

        # Unpack proposed rule parameters.
        e, r = self.embed_dim, self.rank
        key = raw[:, :e]  # (batch, embed_dim)
        a_flat = raw[:, e : e + e * r]
        b_flat = raw[:, e + e * r : e + 2 * e * r]

        A = a_flat.view(-1, e, r)  # (batch, embed_dim, rank)
        B = b_flat.view(-1, r, e)  # (batch, rank, embed_dim)

        # Soft commit weight from cosine similarity between proposed key and h.
        similarity = nn.functional.cosine_similarity(key, h, dim=-1)  # (batch,)
        threshold = torch.sigmoid(self.commit_threshold_logit)
        temperature = self.commit_temperature.clamp(min=0.01)
        commit_weight = torch.sigmoid(
            (similarity - threshold) * temperature,
        ).unsqueeze(-1)  # (batch, 1)

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

        proposal = self.propose_rule(h)

        return logits_rule, confidence, correction, proposal
