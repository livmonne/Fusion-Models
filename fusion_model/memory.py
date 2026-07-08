"""RuleMemory — a fixed bank of learnable low-rank rules with differentiable retrieval.

**High-level idea:**  Instead of modifying the base network's weights to
encode every pattern, we store a set of small *corrections* ("rules") in a
memory bank.  At inference time the model looks up which rules are relevant
to the current task and blends their corrections together.

Each *rule slot* stores:

1. **key** — a trigger embedding that determines *when* the rule fires.
   **Multi-head cross-attention** between the pooled **task embedding**
   ``t`` (derived from the demo pairs, *not* the test input — the rule
   lives in the demonstrations) and every key allows different heads to
   specialise on different aspects of relevance (e.g. spatial layout vs
   colour transformation).
2. **A, B** — a pair of small matrices whose product ``A @ (B @ x)`` forms a
   low-rank correction to the per-token embeddings.  This is the same idea
   as LoRA (Hu et al., 2021): rather than storing a full weight matrix we
   only store two thin factors, keeping memory usage small.

The score-weighted blend of slot corrections is added back onto the token
stream (``x + blended``) — exactly like LoRA's base path — and decoded by a
**shared** classification head.  Earlier revisions decoded each slot's
correction in isolation, which forced every per-cell prediction through a
rank-``r`` bottleneck; the residual base path removes that bottleneck and
also avoids materialising a ``(batch, seq, slots, embed)`` tensor, keeping
peak memory low enough for laptop-class GPUs.

Retrieval uses **multi-head** scaled-dot-product cross-attention so that
gradients flow through the memory bank and the whole system is end-to-end
trainable.  A learned head-combination vector merges per-head attention
distributions into the final per-slot retrieval scores.

**Memory strength (biologically-inspired decay & reinforcement):**

Each slot carries a *strength* scalar in ``[0, 1]`` that modulates its
influence during retrieval — weak memories contribute less, strong ones
dominate.  Strength is derived from two independent signals:

- **Frequency** — how often the slot is activated (cumulative retrieval
  score, normalised to ``[0, 1]``).
- **Recency** — how recently the slot was last strongly activated, with a
  learnable half-life.

The final strength is ``frequency_score * recency_score``.  The **decay
rate** and **reinforcement rate** are *learnable parameters*; their
initial values are derived from the slot count so that an average slot
(softmax scores sum to 1, so the mean score is ``1/num_slots``) settles at
an equilibrium frequency of ~0.5 instead of hovering at the prune
threshold.  Similarly, the recency-activation threshold is *relative* to
the uniform score ``1/num_slots`` rather than a fixed constant.  Earlier
fixed defaults were tuned for ~16 slots and caused permanent prune/respawn
churn at 64+ slots.

During each forward pass the new frequency is computed *differentiably*
from the current rate parameters and used in the strength gating, providing
the gradient path ``loss → scores → strength → new_freq → rate logits``.
The resulting frequency value is then detached and stored in a buffer for
the next step.

Slots whose strength falls below a configurable threshold are considered
"forgotten" and recycled: their parameters are re-initialised with small
random values (``B`` reset to zero so the recycled slot starts as a no-op)
and given a moderate starting strength.  Pruning is deliberately
infrequent (every 500 steps by default) because it resets parameters that
the optimiser still holds momentum for.

Rules proposed by the :class:`~fusion_model.rule_engine.RuleGenerator` are
**soft-blended** into the weakest slot via :meth:`commit_rule`.  Commits
are invoked by the *training loop* after the backward pass (see
``FusionModel.apply_outcomes``) — never mid-step — so the in-place
parameter update can no longer corrupt gradients computed in the same
forward pass.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import masked_mean


class MultiHeadMemoryCrossAttention(nn.Module):
    """Multi-head cross-attention for memory slot retrieval.

    This module replaces a simple single-head dot-product between the pooled
    task embedding ``t`` and the memory slot keys.  Instead, it projects both
    the query (``t``) and the keys/values (slot key bank) into multiple
    independent subspaces ("heads"), computes scaled-dot-product attention in
    each subspace, and combines the results.

    **Why multiple heads?**

    A single dot-product compresses the notion of "relevance" into one
    number per slot.  With *H* heads, the model gets *H* independent
    channels to assess relevance — one head might focus on colour
    transformations, another on spatial layout, a third on symmetry, etc.
    The per-head attention maps are then combined through a *learned*
    head-combination vector into a single set of retrieval scores.

    **Outputs:**

    1. ``scores (batch, num_slots)`` — final retrieval weights (sum to 1
       over slots), ready to be gated by memory strength.
    2. ``context (batch, embed_dim)`` — a rich vector summarising what was
       retrieved, used to enrich the memory pathway's representation for
       the decision router.
    3. ``head_attn (batch, num_heads, num_slots)`` — raw per-head attention
       maps, useful for visualisation and debugging.

    :param embed_dim: Embedding dimensionality.  Must be divisible by
        ``num_heads`` so that each head operates on a ``head_dim``-sized
        subspace.
    :param num_heads: Number of parallel attention heads.  More heads
        give richer retrieval at the cost of a smaller per-head subspace.
        Typical values: 4 (head_dim=64 at embed_dim=256) or 8
        (head_dim=32).
    """

    def __init__(self, embed_dim: int = 256, num_heads: int = 4) -> None:
        super().__init__()
        assert embed_dim % num_heads == 0, (
            f"embed_dim ({embed_dim}) must be divisible by "
            f"num_heads ({num_heads})"
        )
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Linear projections that map the query and keys/values into
        # multi-head subspaces.  No bias — following standard practice
        # for attention Q/K/V projections.
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        # Output projection: recombines the concatenated per-head
        # retrieved vectors back into a single embed_dim vector.
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # Learned head-combination weights.  Initialised to uniform
        # (all ones → softmax gives 1/H each) so that early training
        # uses all heads equally.
        self.head_combine = nn.Parameter(torch.ones(num_heads))

    def forward(
        self, t: torch.Tensor, keys: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute multi-head cross-attention over the memory slot bank.

        :param t: Pooled task embedding of shape ``(batch, embed_dim)``.
            This serves as the *query* — "what kind of rules does the
            current task need?"
        :param keys: Memory slot key bank of shape ``(num_slots, embed_dim)``.
            Each row is a learnable trigger embedding.  These serve as
            both *keys* (for matching) and *values* (for retrieval
            content), each through their own projection.
        :return: A 3-tuple ``(scores, context, head_attn)``:

            - **scores** ``(batch, num_slots)`` — combined retrieval
              weights that sum to 1 over the slot dimension.
            - **context** ``(batch, embed_dim)`` — the standard multi-head
              attention output (weighted sum of value projections), added
              to the memory pathway's router representation.
            - **head_attn** ``(batch, num_heads, num_slots)`` — raw
              per-head softmax attention distributions, logged for
              visualisation (e.g. checking whether heads specialised).
        """
        B = t.shape[0]
        S = keys.shape[0]
        H = self.num_heads
        D = self.head_dim

        # ── Project into multi-head subspaces ──────────────────────────
        Q = self.q_proj(t).view(B, H, D)       # (B, H, D)
        K = self.k_proj(keys).view(S, H, D)    # (S, H, D)
        V = self.v_proj(keys).view(S, H, D)    # (S, H, D)

        # ── Scaled dot-product attention per head ──────────────────────
        attn_logits = torch.einsum("bhd,shd->bhs", Q, K) / (D ** 0.5)
        head_attn = F.softmax(attn_logits, dim=-1)  # (B, H, S)

        # ── Retrieve values via attention-weighted sum ─────────────────
        retrieved = torch.einsum("bhs,shd->bhd", head_attn, V)  # (B, H, D)
        retrieved = retrieved.reshape(B, self.embed_dim)        # (B, E)
        context = self.out_proj(retrieved)                      # (B, E)

        # ── Combine per-head attention into final slot scores ──────────
        head_w = F.softmax(self.head_combine, dim=0)            # (H,)
        scores = torch.einsum("bhs,h->bs", head_attn, head_w)   # (B, S)

        return scores, context, head_attn


class RuleMemory(nn.Module):
    """Fixed-size bank of learnable low-rank rule slots with biologically-inspired
    memory strength dynamics.

    Operates on **per-token spatial embeddings**: retrieval scores are
    computed once per sample from the pooled task embedding ``t`` via
    multi-head cross-attention over the slot key bank, while the low-rank
    corrections are applied independently to every spatial token.  The
    score-blended correction is added onto the token stream (``x +
    blended``) and decoded by a shared classification head — the LoRA-style
    base path.

    :param embed_dim: Dimensionality of the shared input embedding.
    :param num_colours: Number of per-cell colour classes (10 for ARC).
    :param num_slots: How many rule slots to allocate.
    :param rank: Inner rank of each rule's low-rank decomposition ``A @ (B @ x)``.
    :param num_retrieval_heads: Number of attention heads for multi-head
        memory retrieval.  Must evenly divide ``embed_dim``.
    :param prune_threshold: Strength below which a slot is considered
        "forgotten" and eligible for recycling.
    :param prune_every_n_steps: How often (in training forward passes) to
        run the pruning sweep.  Set to 0 to disable automatic pruning.
        Kept deliberately large: pruning re-initialises parameters the
        optimiser still has momentum for, so frequent pruning causes churn.
    :param recency_rel_threshold: A slot counts as "recently activated"
        (resetting its recency timer) when its batch-mean retrieval score
        exceeds ``recency_rel_threshold / num_slots`` — i.e. this many
        times the uniform share.  Relative scaling keeps the dynamics
        sensible across slot counts.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_colours: int = 10,
        num_slots: int = 64,
        rank: int = 16,
        num_retrieval_heads: int = 4,
        prune_threshold: float = 0.05,
        prune_every_n_steps: int = 500,
        recency_rel_threshold: float = 2.0,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_colours = num_colours
        self.num_slots = num_slots
        self.rank = rank
        self.num_retrieval_heads = num_retrieval_heads
        self.prune_threshold = prune_threshold
        self.prune_every_n_steps = prune_every_n_steps
        # Absolute activation threshold derived from the uniform score.
        self.recency_activation_threshold = recency_rel_threshold / num_slots

        # --- Learnable rule bank ---
        self.keys = nn.Parameter(torch.randn(num_slots, embed_dim) * 0.02)

        # Low-rank factors: correction = A @ (B @ x_token).  B starts at
        # zero so every slot begins as a no-op correction (standard LoRA
        # initialisation).
        self.B = nn.Parameter(torch.zeros(num_slots, rank, embed_dim))
        self.A = nn.Parameter(torch.randn(num_slots, embed_dim, rank) * 0.02)

        # Shared classification head reading the residual stream
        # (x + blended correction).  A LayerNorm keeps the residual sum
        # well-scaled regardless of how large the corrections grow.
        self.head_norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_colours)

        # ── Multi-head cross-attention for retrieval ─────────────────────
        self.retrieval_attn = MultiHeadMemoryCrossAttention(
            embed_dim=embed_dim,
            num_heads=num_retrieval_heads,
        )

        # LayerNorm stabilises the residual sum of the pooled blended
        # correction and the cross-attention context vector before it's
        # fed to the decision router.
        self.repr_norm = nn.LayerNorm(embed_dim)

        # Running utility score per slot (not trained — purely diagnostic).
        self.utility: torch.Tensor
        self.register_buffer("utility", torch.zeros(num_slots))

        # ── Memory strength: frequency + recency ─────────────────────────
        # Initial rates are derived from the slot count: with softmax
        # scores the mean per-slot score is 1/num_slots, so equilibrium
        # frequency = reinforce * mean_score / (1 - decay).  We pick the
        # reinforcement rate so an average slot equilibrates at ~0.5.
        decay_init = 0.999
        reinforce_init = min(0.5, max(1e-3, 0.5 * (1.0 - decay_init) * num_slots))
        self.decay_rate_logit = nn.Parameter(
            torch.tensor(math.log(decay_init / (1.0 - decay_init)))
        )
        self.reinforce_rate_logit = nn.Parameter(
            torch.tensor(math.log(reinforce_init / (1.0 - reinforce_init)))
        )
        self.recency_halflife_log = nn.Parameter(torch.tensor(math.log(500.0)))

        self.frequency: torch.Tensor
        self.register_buffer("frequency", torch.full((num_slots,), 0.5))
        self.steps_since_activation: torch.Tensor
        self.register_buffer("steps_since_activation", torch.zeros(num_slots))
        self.step_counter: torch.Tensor
        self.register_buffer("step_counter", torch.tensor(0, dtype=torch.long))

    # ── Strength computation ────────────────────────────────────────────

    def get_strength(
        self, freq_override: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Compute per-slot memory strength as ``frequency_score * recency_score``.

        :param freq_override: If provided, use this tensor instead of
            ``self.frequency`` for the frequency component.  This allows
            callers to pass a differentiable frequency tensor so that
            gradients flow back to the decay/reinforce rate parameters.
        :return: Strength tensor of shape ``(num_slots,)`` in ``[0, 1]``.
        """
        freq_score = (
            freq_override if freq_override is not None else self.frequency
        ).clamp(0.0, 1.0)

        half_life = self.recency_halflife_log.exp().clamp(min=1.0)
        recency_score = torch.exp(-math.log(2.0) * self.steps_since_activation / half_life)

        return freq_score * recency_score

    # ── Forward pass ──────────────────────────────────────────────────────

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        pad_mask: torch.Tensor | None = None,
        update_state: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Retrieve relevant rules and produce per-cell memory-pathway logits.

        :param x: Spatial token embeddings ``(batch, seq, embed_dim)``.
        :param t: Pooled task embedding ``(batch, embed_dim)`` used as the
            retrieval query.
        :param pad_mask: Boolean validity mask ``(batch, seq)``; True for
            real cells.  Used for the pooled router representation.
        :param update_state: Whether to update the frequency/recency
            buffers (training only).  The demo-verification pass sets this
            to False so each optimiser step counts as exactly one memory
            step.
        :return: Tuple of ``(logits_mem, mem_repr, retrieval_info)`` where
            ``logits_mem`` has shape ``(batch, seq, num_colours)``,
            ``mem_repr`` is a pooled representation ``(batch, embed_dim)``
            for the router, and ``retrieval_info`` carries ``scores``,
            ``strength``, and ``head_attn``.
        """
        # -- 1. Multi-head cross-attention retrieval from pooled t --
        raw_scores, mem_context, head_attn = self.retrieval_attn(t, self.keys)

        # -- 2. Compute differentiable frequency & gate by memory strength --
        track = self.training and update_state
        if track:
            batch_mean_scores = raw_scores.detach().mean(dim=0)  # (S,)
            decay_rate = torch.sigmoid(self.decay_rate_logit)
            reinforce_rate = torch.sigmoid(self.reinforce_rate_logit)
            new_freq = (
                self.frequency.detach().clone() * decay_rate
                + reinforce_rate * batch_mean_scores
            ).clamp(max=1.0)
            strength = self.get_strength(freq_override=new_freq)  # (S,)
        else:
            strength = self.get_strength()  # (S,)

        gated_scores = raw_scores * strength.unsqueeze(0)
        scores = gated_scores / (gated_scores.sum(dim=-1, keepdim=True) + 1e-8)  # (B, S)

        # -- 3. Score-blended low-rank correction --
        # compressed(b, t, s, r) = B(s, r, e) · x(b, t, e) — the rank-r
        # bottleneck per slot.  Scaling by the retrieval scores *before*
        # contracting with A avoids materialising the much larger
        # (batch, seq, slots, embed) per-slot correction tensor.
        compressed = torch.einsum("sre, bte -> btsr", self.B, x)
        weighted = compressed * scores.unsqueeze(1).unsqueeze(-1)  # (B, T, S, r)
        blended = torch.einsum("ser, btsr -> bte", self.A, weighted)  # (B, T, E)

        # -- 4. Decode from the residual stream (base path + correction) --
        logits_mem = self.head(self.head_norm(x + blended))  # (B, T, C)

        # -- 5. Router representation --
        # (a) masked mean of the blended correction — what the rules did;
        # (b) mem_context — which slots were selected (attention output).
        mem_repr = self.repr_norm(masked_mean(blended, pad_mask) + mem_context)

        # -- 6. Persist frequency state & update recency (training only) --
        if track:
            with torch.no_grad():
                self.frequency.copy_(new_freq.detach())

                self.steps_since_activation += 1
                activated = batch_mean_scores > self.recency_activation_threshold
                self.steps_since_activation[activated] = 0.0

                self.utility = self.utility * 0.99 + 0.01 * batch_mean_scores

                self.step_counter += 1
                if (
                    self.prune_every_n_steps > 0
                    and self.step_counter.item() % self.prune_every_n_steps == 0
                ):
                    self.prune_weak_slots()

        retrieval_info: dict[str, torch.Tensor] = {
            "scores": scores,
            "strength": strength,
            "head_attn": head_attn,
        }
        return logits_mem, mem_repr, retrieval_info

    # ── Slot selection ──────────────────────────────────────────────────

    def get_weakest_slot(self) -> int:
        """Return the index of the slot with the lowest combined strength."""
        strength = self.get_strength()
        combined = strength + 1e-6 * self.utility
        return int(combined.argmin().item())

    # ── Rule commitment ──────────────────────────────────────────────────

    @torch.no_grad()
    def commit_rule(
        self,
        slot_idx: int,
        key: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        commit_weight: float = 1.0,
    ) -> None:
        """Soft-blend a proposed rule into a slot.

        Must only be called *between* optimiser steps (the training loop
        does this via ``FusionModel.apply_outcomes``) — never between a
        forward and backward pass, where the in-place update would corrupt
        the gradients of the affected slot.

        :param slot_idx: Target slot index in ``[0, num_slots)``.
        :param key: Trigger embedding ``(embed_dim,)``.
        :param A: Low-rank factor ``(embed_dim, rank)``.
        :param B: Low-rank factor ``(rank, embed_dim)``.
        :param commit_weight: Blend weight in ``[0, 1]``.
        """
        w = commit_weight
        self.keys.data[slot_idx].lerp_(key, w)
        self.A.data[slot_idx].lerp_(A, w)
        self.B.data[slot_idx].lerp_(B, w)
        self.utility[slot_idx] *= 1.0 - w

        self.frequency[slot_idx] = max(w, self.frequency[slot_idx].item())
        self.steps_since_activation[slot_idx] = 0.0

    # ── Pruning ──────────────────────────────────────────────────────────

    @torch.no_grad()
    def prune_weak_slots(self, threshold: float | None = None) -> int:
        """Recycle slots whose strength has decayed below *threshold*.

        Recycled slots get a fresh random key, a small random ``A``, and a
        **zero** ``B`` so they restart as no-op corrections (matching the
        initial LoRA-style state).

        :param threshold: Override for ``self.prune_threshold``.
        :return: Number of slots that were pruned.
        """
        thresh = threshold if threshold is not None else self.prune_threshold
        strength = self.get_strength()
        dead = strength < thresh

        n_pruned = int(dead.sum().item())
        if n_pruned > 0:
            self.keys.data[dead] = torch.randn_like(self.keys.data[dead]) * 0.02
            self.A.data[dead] = torch.randn_like(self.A.data[dead]) * 0.02
            self.B.data[dead] = 0.0
            self.frequency[dead] = 0.5
            self.steps_since_activation[dead] = 0.0
            self.utility[dead] = 0.0

        return n_pruned
