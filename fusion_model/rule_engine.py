"""RuleGenerator — ephemeral corrections *and* persistent rule proposals.

**Concept:**  The RuleGenerator serves two roles:

1. **Ephemeral correction** — a small MLP takes the pooled embedding ``h``
   and outputs a one-shot low-rank correction applied **per spatial token**:
   ``A_new @ (B_new @ x_token)`` plus a confidence scalar.

2. **Rule proposal** — the generator maintains a circular history buffer of
   recent ``(h, decision, outcome)`` triples.  The proposal pipeline has
   three stages of cross-attention that operate *entirely on history* — the
   current input is deliberately excluded so that proposed rules reflect
   what actually worked rather than what the model is currently looking at.

This two-tier design lets the model invent hypotheses on the fly *and*
distill recurring patterns into the long-term rule bank.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RuleGenerator(nn.Module):
    """Generate ephemeral low-rank corrections and propose persistent rules.

    Now operates on **per-token spatial embeddings**: the MLP produces A, B
    matrices from the pooled ``h``, but the correction is applied to every
    token in the spatial sequence ``x``.

    :param embed_dim: Dimensionality of the shared input embedding.
    :param num_colours: Number of per-cell colour classes (10 for ARC).
    :param rank: Inner rank of the generated low-rank matrices.
    :param hidden_dim: Width of the internal MLP.
    :param history_size: Capacity of the circular buffer.
    :param min_history: Minimum entries before rule proposal is attempted.
    :param decision_vocab_size: Vocabulary size for decision embeddings
        stored in history.
    :param decision_embed_dim: Embedding dimension for stored decision
        indices before projection to ``embed_dim``.
    :param outcome_proj_dim: Hidden dimension for projecting scalar outcome
        signals to ``embed_dim``.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_colours: int = 10,
        rank: int = 8,
        hidden_dim: int = 256,
        history_size: int = 512,
        min_history: int = 64,
        decision_vocab_size: int = 9000,
        decision_embed_dim: int = 32,
        outcome_proj_dim: int = 64,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.rank = rank
        self.num_colours = num_colours
        self.decision_vocab_size = decision_vocab_size
        self.history_size = history_size
        self.min_history = min_history
        self.decision_embed_dim = decision_embed_dim

        # ── Ephemeral correction MLP ────────────────────────
        # Produces A (embed_dim x rank) and B (rank x embed_dim) + confidence.
        out_dim = (rank * embed_dim) + (embed_dim * rank) + 1
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, out_dim),
        )
        # Per-token classification head: embed_dim → num_colours.
        self.head = nn.Linear(embed_dim, num_colours)

        # ── History buffer (non-gradient circular buffer) ────────────────
        self.register_buffer("history_h", torch.zeros(history_size, embed_dim))
        self.register_buffer(
            "history_decisions", torch.full((history_size,), -1, dtype=torch.long),
        )
        self.register_buffer("history_outcomes", torch.zeros(history_size))
        self.register_buffer("history_ptr", torch.tensor(0, dtype=torch.long))
        self.register_buffer("history_count", torch.tensor(0, dtype=torch.long))

        # ── Rule proposer ────────────────────────────────────────────────
        self.decision_embed = nn.Embedding(decision_vocab_size, decision_embed_dim)

        self.input_dec_cross_attn = nn.MultiheadAttention(
            embed_dim, num_heads=4, batch_first=True,
        )
        self.dec_proj = nn.Linear(decision_embed_dim, embed_dim)

        self.outcome_proj = nn.Sequential(
            nn.Linear(1, outcome_proj_dim),
            nn.GELU(),
            nn.Linear(outcome_proj_dim, embed_dim),
        )
        self.outcome_cross_attn = nn.MultiheadAttention(
            embed_dim, num_heads=4, batch_first=True,
        )

        self.synthesis_query = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.synthesis_cross_attn = nn.MultiheadAttention(
            embed_dim, num_heads=4, batch_first=True,
        )

        proposer_out = embed_dim + (embed_dim * rank) + (rank * embed_dim)
        self.rule_proj = nn.Linear(embed_dim, proposer_out)

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
        """Append a batch of ``(h, decision, outcome)`` triples to the buffer."""
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

        Returns ``None`` when the history buffer has fewer than
        ``min_history`` entries.
        """
        count = self.history_count.item()
        if count < self.min_history:
            return None

        valid_h = self.history_h[:count].clone()
        valid_dec = self.history_decisions[:count].clone()
        valid_out = self.history_outcomes[:count].clone()

        hist_h = valid_h.unsqueeze(0).expand(batch_size, -1, -1)

        # Stage 1: historical inputs cross-attend over historical decisions.
        dec_emb = self.dec_proj(self.decision_embed(valid_dec))
        kv_dec = dec_emb.unsqueeze(0).expand(batch_size, -1, -1)
        attended_input_dec, _ = self.input_dec_cross_attn(hist_h, kv_dec, kv_dec)

        # Stage 2: input→decision result cross-attends over outcome embeddings.
        outcome_emb = self.outcome_proj(valid_out.unsqueeze(-1))
        kv_out = outcome_emb.unsqueeze(0).expand(batch_size, -1, -1)
        attended_outcome, _ = self.outcome_cross_attn(
            attended_input_dec, kv_out, kv_out,
        )

        attended_input_dec_pooled = attended_input_dec.mean(dim=1)
        attended_outcome_pooled = attended_outcome.mean(dim=1)

        # Stage 3: learned synthesis query attends over the two pooled results.
        context_tokens = torch.stack(
            [attended_input_dec_pooled, attended_outcome_pooled], dim=1,
        )
        syn_query = self.synthesis_query.expand(batch_size, -1, -1)
        synthesised, _ = self.synthesis_cross_attn(
            syn_query, context_tokens, context_tokens,
        )
        synthesised = synthesised.squeeze(1)

        raw = self.rule_proj(synthesised)

        e, r = self.embed_dim, self.rank
        key = raw[:, :e]
        a_flat = raw[:, e : e + e * r]
        b_flat = raw[:, e + e * r : e + 2 * e * r]

        A = a_flat.view(-1, e, r)
        B = b_flat.view(-1, r, e)

        mean_hist = valid_h.mean(dim=0, keepdim=True).expand(batch_size, -1)
        similarity = nn.functional.cosine_similarity(key, mean_hist, dim=-1)
        threshold = torch.sigmoid(self.commit_threshold_logit)
        temperature = self.commit_temperature.clamp(min=0.01)
        commit_weight = torch.sigmoid(
            (similarity - threshold) * temperature,
        ).unsqueeze(-1)

        return {
            "key": key,
            "A": A,
            "B": B,
            "commit_weight": commit_weight,
        }

    # ── Forward pass ─────────────────────────────────────────────────────

    def forward(
        self, x: torch.Tensor, h: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor] | None]:
        """Produce per-token ephemeral rule logits, confidence, repr, and a rule proposal.

        :param x: Spatial token embeddings ``(batch, seq, embed_dim)``.
        :param h: Pooled embedding ``(batch, embed_dim)`` for the MLP and
            history/proposal machinery.
        :return: Tuple ``(logits_rule, confidence, rule_repr, proposal)``
            where ``logits_rule`` has shape ``(batch, seq, num_colours)``,
            ``rule_repr`` is ``(batch, embed_dim)`` for the router.
        """
        batch = h.shape[0]
        params = self.mlp(h)

        b_size = self.rank * self.embed_dim
        a_size = self.embed_dim * self.rank
        b_flat = params[:, :b_size]
        a_flat = params[:, b_size : b_size + a_size]
        confidence = torch.sigmoid(params[:, -1:])

        b_mat = b_flat.view(batch, self.rank, self.embed_dim)       # (B, r, E)
        a_mat = a_flat.view(batch, self.embed_dim, self.rank)       # (B, E, r)

        # Per-token correction: A @ (B @ x_token) for every token.
        # compressed: (B, r, E) @ (B, E, seq) → (B, r, seq)
        compressed = torch.bmm(b_mat, x.transpose(1, 2))
        # correction: (B, E, r) @ (B, r, seq) → (B, E, seq) → transpose → (B, seq, E)
        correction = torch.bmm(a_mat, compressed).transpose(1, 2)

        # Per-token classification.
        logits_rule = self.head(correction)  # (B, seq, num_colours)

        # Router representation: mean-pool the per-token corrections.
        rule_repr = correction.mean(dim=1)  # (B, E)

        proposal = self.propose_rule(batch)

        return logits_rule, confidence, rule_repr, proposal
