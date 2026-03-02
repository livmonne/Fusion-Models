"""RuleGenerator — ephemeral corrections *and* persistent rule proposals.

**Concept:**  The RuleGenerator serves two roles:

1. **Ephemeral correction** — a small MLP takes the shared embedding ``h``
   and outputs a one-shot low-rank correction (``A_new @ (B_new @ h)``) plus
   a confidence scalar.  This correction is used for the current forward pass
   only.

2. **Rule proposal** — the generator maintains a circular history buffer of
   recent ``(h, prediction)`` pairs.  Using scaled-dot-product attention over
   this buffer, it produces a *proposed persistent rule* — a ``(key, A, B)``
   triple together with a proposal confidence.  When that confidence exceeds
   a **learnable commitment threshold**, the rule is committed to the
   :class:`~fusion_model.memory.RuleMemory` bank (handled by the orchestrator
   in :class:`~fusion_model.model.FusionModel`).

This two-tier design lets the model invent hypotheses on the fly *and*
distill recurring patterns into the long-term rule bank.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RuleGenerator(nn.Module):
    """Generate ephemeral low-rank corrections and propose persistent rules.

    :param embed_dim: Dimensionality of the shared input embedding.
    :param num_classes: Number of output classes (CLEVR answers).
    :param rank: Inner rank of the generated low-rank matrices.
    :param hidden_dim: Width of the internal MLP.
    :param history_size: Capacity of the circular ``(h, decision)`` buffer.
    :param min_history: Minimum entries in the buffer before rule proposal
        is attempted.
    :param decision_embed_dim: Embedding dimension for stored decision indices.
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

        # ── Ephemeral correction MLP (unchanged) ────────────────────────
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
            "history_decisions", torch.full((history_size,), -1, dtype=torch.long)
        )
        self.register_buffer("history_ptr", torch.tensor(0, dtype=torch.long))
        self.register_buffer("history_count", torch.tensor(0, dtype=torch.long))

        # ── Rule proposer ────────────────────────────────────────────────
        self.decision_embed = nn.Embedding(num_classes, decision_embed_dim)

        proposer_in = embed_dim + embed_dim + decision_embed_dim
        proposer_out = embed_dim + (embed_dim * rank) + (rank * embed_dim) + 1
        self.rule_proposer = nn.Sequential(
            nn.Linear(proposer_in, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, proposer_out),
        )

        # Learnable commitment threshold (sigmoid → 0.5 initially).
        self.commit_threshold_logit = nn.Parameter(torch.tensor(0.0))

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
        self.history_count.fill_(
            min(self.history_count.item() + batch, self.history_size)
        )

    # ── Rule proposal ────────────────────────────────────────────────────

    def propose_rule(
        self, h: torch.Tensor
    ) -> dict[str, torch.Tensor] | None:
        """Propose a persistent rule from current input + history context.

        Returns ``None`` when the history buffer has fewer than
        ``min_history`` entries.

        :param h: Shared embedding ``(batch, embed_dim)``.
        :return: Dict with ``key``, ``A``, ``B``, ``confidence`` (per batch
            element) and scalar ``commit_threshold``, or ``None``.
        """
        count = self.history_count.item()
        if count < self.min_history:
            return None

        valid_h = self.history_h[:count].clone()  # (count, embed_dim)
        valid_dec = self.history_decisions[:count].clone()  # (count,)

        # Attend over history using current h as query.
        attn_scores = torch.matmul(h, valid_h.t()) / (self.embed_dim**0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)  # (batch, count)

        attended_h = torch.matmul(attn_weights, valid_h)  # (batch, embed_dim)
        dec_emb = self.decision_embed(valid_dec)  # (count, dec_embed_dim)
        attended_dec = torch.matmul(attn_weights, dec_emb)  # (batch, dec_embed_dim)

        combined = torch.cat([h, attended_h, attended_dec], dim=-1)
        raw = self.rule_proposer(combined)  # (batch, proposer_out)

        # Unpack proposed rule parameters.
        e, r = self.embed_dim, self.rank
        key = raw[:, :e]  # (batch, embed_dim)
        a_flat = raw[:, e : e + e * r]
        b_flat = raw[:, e + e * r : e + 2 * e * r]
        confidence = torch.sigmoid(raw[:, -1:])  # (batch, 1)

        A = a_flat.view(-1, e, r)  # (batch, embed_dim, rank)
        B = b_flat.view(-1, r, e)  # (batch, rank, embed_dim)

        threshold = torch.sigmoid(self.commit_threshold_logit)

        return {
            "key": key,
            "A": A,
            "B": B,
            "confidence": confidence,
            "commit_threshold": threshold,
        }

    # ── Forward pass ─────────────────────────────────────────────────────

    def forward(
        self, h: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor] | None]:
        """Produce ephemeral rule logits, confidence, and a rule proposal.

        :param h: Shared embedding ``(batch, embed_dim)``.
        :return: Tuple ``(logits_rule, confidence, proposal)`` where
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

        return logits_rule, confidence, proposal
