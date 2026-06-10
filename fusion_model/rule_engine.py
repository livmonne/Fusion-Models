"""RuleGenerator — ephemeral corrections *and* a routed history-based proposer.

**Concept:**  The RuleGenerator hosts two of the four expert pathways:

1. **Ephemeral generator** (pathway "rule") — a small MLP takes the pooled
   task embedding ``t`` and outputs a one-shot low-rank correction applied
   **per spatial token**: ``A @ (B @ x_token)``.  The correction is added
   back onto the token stream (``x + correction``) and decoded by a shared
   head — the LoRA-style base path.  It exists only for this forward pass.

2. **History proposer** (pathway "prop") — maintains a circular history
   buffer of recent ``(task embedding, decision, outcome)`` triples and
   synthesises a rule from *history alone* via three stages of
   cross-attention.  Crucially, the proposed rule is **also applied to the
   current input** and produces its own per-cell logits that the decision
   router can select.  This closes the learning loop: the proposer "tries"
   its rule on every forward pass, the task loss grades the attempt, and
   gradients flow back through the entire history-attention pipeline.
   (Earlier revisions only used proposals for no-grad commits into the
   memory bank, which left the proposer's rule content untrained.)

**History entries** store three views of a past attempt:

- ``task`` — the pooled task embedding ``t`` (what the task looked like);
- ``decision`` — a compact feature vector of what the model *did*: the
  colour histogram of its prediction concatenated with the routing weights
  ``alpha`` (which pathway it trusted).  This replaces an earlier
  content-free hash of the predicted cells;
- ``outcome`` — the per-sample task loss (how well it went).  Outcomes are
  standardised over the buffer at read time so the proposer sees relative
  quality, independent of the absolute loss scale, which shrinks over
  training.

**Commitment** into the persistent memory bank is *outcome-gated*: the
training loop measures the proposal pathway's actual per-sample loss and
calls :meth:`should_commit`, which only approves a commit when the best
sample in the batch beat an exponential moving average of recent proposal
losses — i.e. rules are stored because they demonstrably *worked*, not
because a similarity heuristic fired.  A learned commit weight (regularised
toward a target rate) controls how strongly the rule is blended in.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from .common import masked_mean


def apply_low_rank(
    x: torch.Tensor, A: torch.Tensor, B: torch.Tensor
) -> torch.Tensor:
    """Apply a batched low-rank correction ``A @ (B @ x_token)`` per token.

    :param x: Token embeddings ``(batch, seq, embed_dim)``.
    :param A: Low-rank factor ``(batch, embed_dim, rank)``.
    :param B: Low-rank factor ``(batch, rank, embed_dim)``.
    :return: Correction tensor ``(batch, seq, embed_dim)``.
    """
    compressed = torch.bmm(B, x.transpose(1, 2))        # (B, r, seq)
    return torch.bmm(A, compressed).transpose(1, 2)     # (B, seq, E)


class RuleGenerator(nn.Module):
    """Generate ephemeral low-rank corrections and propose history-based rules.

    Both sub-pathways operate on **per-token spatial embeddings**: rule
    parameters are produced from pooled context (the task embedding ``t``
    for the generator; the history buffer for the proposer), but the
    corrections are applied to every token in the spatial sequence ``x``
    and decoded through residual base paths.

    :param embed_dim: Dimensionality of the shared input embedding.
    :param num_colours: Number of per-cell colour classes (10 for ARC).
    :param rank: Inner rank of the generated low-rank matrices.
    :param hidden_dim: Width of the internal MLP.
    :param history_size: Capacity of the circular buffer.
    :param min_history: Minimum entries before rule proposal is attempted.
    :param num_pathways: Number of expert pathways in the full model (the
        routing weights stored as decision features have this length).
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
        num_pathways: int = 4,
        outcome_proj_dim: int = 64,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.rank = rank
        self.num_colours = num_colours
        self.history_size = history_size
        self.min_history = min_history
        # Decision feature = predicted colour histogram + routing weights.
        self.decision_feature_dim = num_colours + num_pathways

        # ── Ephemeral generator (pathway "rule") ─────────────────────────
        # Produces A (embed_dim x rank) and B (rank x embed_dim) from t.
        out_dim = 2 * rank * embed_dim
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, out_dim),
        )
        # Shared per-token head reading the residual stream (x + correction).
        self.gen_norm = nn.LayerNorm(embed_dim)
        self.gen_head = nn.Linear(embed_dim, num_colours)
        self.rule_repr_norm = nn.LayerNorm(embed_dim)

        # ── History buffer (non-gradient circular buffer) ────────────────
        self.history_task: torch.Tensor
        self.register_buffer("history_task", torch.zeros(history_size, embed_dim))
        self.history_decisions: torch.Tensor
        self.register_buffer(
            "history_decisions",
            torch.zeros(history_size, self.decision_feature_dim),
        )
        self.history_outcomes: torch.Tensor
        self.register_buffer("history_outcomes", torch.zeros(history_size))
        self.history_ptr: torch.Tensor
        self.register_buffer("history_ptr", torch.tensor(0, dtype=torch.long))
        self.history_count: torch.Tensor
        self.register_buffer("history_count", torch.tensor(0, dtype=torch.long))

        # ── History proposer (pathway "prop") ────────────────────────────
        self.dec_proj = nn.Linear(self.decision_feature_dim, embed_dim)

        self.input_dec_cross_attn = nn.MultiheadAttention(
            embed_dim, num_heads=4, batch_first=True,
        )

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

        proposer_out = embed_dim + 2 * embed_dim * rank
        self.rule_proj = nn.Linear(embed_dim, proposer_out)

        # Proposal pathway decode head (base path, like the generator's).
        self.prop_norm = nn.LayerNorm(embed_dim)
        self.prop_head = nn.Linear(embed_dim, num_colours)
        self.prop_repr_norm = nn.LayerNorm(embed_dim)

        # Router representation used while the history buffer is still too
        # small for proposals (the router masks this pathway out, but its
        # key/value token should not be degenerate zeros).
        self.inactive_repr = nn.Parameter(torch.randn(embed_dim) * 0.02)

        self.commit_threshold_logit = nn.Parameter(torch.tensor(0.0))
        self.commit_temperature = nn.Parameter(torch.tensor(1.0))

        # EMA of the proposal pathway's per-sample loss — the outcome gate
        # for commits ("only store rules that beat the running average").
        self.prop_loss_ema: torch.Tensor
        self.register_buffer("prop_loss_ema", torch.tensor(0.0))
        self.prop_loss_ema_init: torch.Tensor
        self.register_buffer("prop_loss_ema_init", torch.tensor(False))

    # ── History management ───────────────────────────────────────────────

    @torch.no_grad()
    def update_history(
        self,
        task: torch.Tensor,
        decisions: torch.Tensor,
        outcomes: torch.Tensor,
    ) -> None:
        """Append a batch of ``(task, decision, outcome)`` triples to the buffer.

        :param task: Pooled task embeddings ``(batch, embed_dim)``.
        :param decisions: Decision feature vectors
            ``(batch, decision_feature_dim)``.
        :param outcomes: Per-sample outcome signal ``(batch,)``.
        """
        batch = task.shape[0]
        task_det = task.detach()
        dec_det = decisions.detach()
        out_det = outcomes.detach()

        ptr = int(self.history_ptr.item())
        end = ptr + batch

        if end <= self.history_size:
            self.history_task[ptr:end] = task_det
            self.history_decisions[ptr:end] = dec_det
            self.history_outcomes[ptr:end] = out_det
        else:
            first = self.history_size - ptr
            self.history_task[ptr:] = task_det[:first]
            self.history_decisions[ptr:] = dec_det[:first]
            self.history_outcomes[ptr:] = out_det[:first]
            self.history_task[: end - self.history_size] = task_det[first:]
            self.history_decisions[: end - self.history_size] = dec_det[first:]
            self.history_outcomes[: end - self.history_size] = out_det[first:]

        self.history_ptr.fill_(end % self.history_size)
        self.history_count.fill_(
            min(int(self.history_count.item()) + batch, self.history_size),
        )

    # ── Commit gating ────────────────────────────────────────────────────

    @torch.no_grad()
    def should_commit(self, prop_loss: torch.Tensor) -> int | None:
        """Outcome-gate for rule commits.

        Called by the training loop with the proposal pathway's *measured*
        per-sample task loss.  Approves a commit (returning the best
        sample's index) only when that sample beat the running average of
        recent proposal losses, then updates the running average.

        :param prop_loss: Per-sample CE of the proposal pathway ``(batch,)``.
        :return: Index of the batch sample whose rule should be committed,
            or ``None`` when no sample beat the running average (or on the
            very first call, which only initialises the average).
        """
        batch_mean = float(prop_loss.mean().item())
        if not bool(self.prop_loss_ema_init.item()):
            self.prop_loss_ema.fill_(batch_mean)
            self.prop_loss_ema_init.fill_(True)
            return None

        best = int(prop_loss.argmin().item())
        approved = float(prop_loss[best].item()) < float(self.prop_loss_ema.item())
        self.prop_loss_ema.mul_(0.99).add_(0.01 * batch_mean)
        return best if approved else None

    # ── Rule proposal ────────────────────────────────────────────────────

    def propose_rule(self, batch_size: int) -> dict[str, torch.Tensor] | None:
        """Propose a rule from historical context only.

        The pipeline deliberately excludes the current input so that
        proposed rules reflect what *worked in the past* rather than what
        the model is currently looking at:

        1. Historical task embeddings cross-attend over historical
           decision features.
        2. That result cross-attends over (standardised) outcome signals.
        3. A learned synthesis query fuses the two pooled results.

        Returns ``None`` when the history buffer has fewer than
        ``min_history`` entries.
        """
        count = int(self.history_count.item())
        if count < self.min_history:
            return None

        valid_task = self.history_task[:count].clone()
        valid_dec = self.history_decisions[:count].clone()
        valid_out = self.history_outcomes[:count].clone()

        # Standardise outcomes over the buffer: the proposer should react
        # to *relative* quality, not the absolute loss scale (which decays
        # over training).
        out_std = (valid_out - valid_out.mean()) / (valid_out.std() + 1e-5)

        hist_task = valid_task.unsqueeze(0).expand(batch_size, -1, -1)

        # Stage 1: historical tasks cross-attend over historical decisions.
        dec_emb = self.dec_proj(valid_dec)
        kv_dec = dec_emb.unsqueeze(0).expand(batch_size, -1, -1)
        attended_input_dec, _ = self.input_dec_cross_attn(
            hist_task, kv_dec, kv_dec, need_weights=False,
        )

        # Stage 2: task→decision result cross-attends over outcome embeddings.
        outcome_emb = self.outcome_proj(out_std.unsqueeze(-1))
        kv_out = outcome_emb.unsqueeze(0).expand(batch_size, -1, -1)
        attended_outcome, _ = self.outcome_cross_attn(
            attended_input_dec, kv_out, kv_out, need_weights=False,
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

        mean_hist = valid_task.mean(dim=0, keepdim=True).expand(batch_size, -1)
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

    # ── Pathway applications ─────────────────────────────────────────────

    def ephemeral_logits(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        pad_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate a one-shot rule from ``t`` and apply it to ``x``.

        :param x: Spatial token embeddings ``(batch, seq, embed_dim)``.
        :param t: Pooled task embedding ``(batch, embed_dim)``.
        :param pad_mask: Boolean validity mask ``(batch, seq)``.
        :return: ``(logits_rule, rule_repr)`` with shapes
            ``(batch, seq, num_colours)`` and ``(batch, embed_dim)``.
        """
        batch = t.shape[0]
        params = self.mlp(t)

        size = self.rank * self.embed_dim
        b_mat = params[:, :size].view(batch, self.rank, self.embed_dim)
        a_mat = params[:, size:].view(batch, self.embed_dim, self.rank)

        correction = apply_low_rank(x, a_mat, b_mat)
        logits_rule = self.gen_head(self.gen_norm(x + correction))
        rule_repr = self.rule_repr_norm(masked_mean(correction, pad_mask))
        return logits_rule, rule_repr

    def proposal_logits(
        self,
        x: torch.Tensor,
        proposal: dict[str, torch.Tensor],
        pad_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply a proposed rule to ``x`` and decode per-cell logits.

        This is what makes the proposer a *routed pathway*: the same rule
        that may later be committed to memory is tried on the current
        input, so the task loss directly grades (and trains) the proposal
        machinery.

        :param x: Spatial token embeddings ``(batch, seq, embed_dim)``.
        :param proposal: Output of :meth:`propose_rule`.
        :param pad_mask: Boolean validity mask ``(batch, seq)``.
        :return: ``(logits_prop, prop_repr)`` with shapes
            ``(batch, seq, num_colours)`` and ``(batch, embed_dim)``.
        """
        correction = apply_low_rank(x, proposal["A"], proposal["B"])
        logits_prop = self.prop_head(self.prop_norm(x + correction))
        prop_repr = self.prop_repr_norm(masked_mean(correction, pad_mask))
        return logits_prop, prop_repr

    # ── Forward pass ─────────────────────────────────────────────────────

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        pad_mask: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        """Run both sub-pathways.

        :param x: Spatial token embeddings ``(batch, seq, embed_dim)``.
        :param t: Pooled task embedding ``(batch, embed_dim)``.
        :param pad_mask: Boolean validity mask ``(batch, seq)``.
        :return: Dict with keys:

            - ``logits_rule`` ``(batch, seq, num_colours)`` — ephemeral
              generator logits.
            - ``rule_repr`` ``(batch, embed_dim)`` — generator router repr.
            - ``proposal`` — dict from :meth:`propose_rule`, or ``None``.
            - ``prop_active`` — bool; whether a proposal was produced.
            - ``logits_prop`` ``(batch, seq, num_colours)`` or ``None``.
            - ``prop_repr`` ``(batch, embed_dim)`` — proposal router repr
              (a learned placeholder while inactive).
        """
        batch = t.shape[0]
        logits_rule, rule_repr = self.ephemeral_logits(x, t, pad_mask)

        proposal = self.propose_rule(batch)
        if proposal is not None:
            logits_prop, prop_repr = self.proposal_logits(x, proposal, pad_mask)
            prop_active = True
        else:
            logits_prop = None
            prop_repr = self.inactive_repr.unsqueeze(0).expand(batch, -1)
            prop_active = False

        return {
            "logits_rule": logits_rule,
            "rule_repr": rule_repr,
            "proposal": proposal,
            "prop_active": prop_active,
            "logits_prop": logits_prop,
            "prop_repr": prop_repr,
        }
