"""RuleMemory — a fixed bank of learnable low-rank rules with differentiable retrieval.

**High-level idea:**  Instead of modifying the base network's weights to
encode every pattern, we store a set of small *corrections* ("rules") in a
memory bank.  At inference time the model looks up which rules are relevant
to the current input and blends their corrections together.

Each *rule slot* stores three things:

1. **key** — a trigger embedding that determines *when* the rule fires.
   The model computes cosine-like attention between the input embedding and
   every key to decide relevance.
2. **A, B** — a pair of small matrices whose product ``A @ (B @ x)`` forms a
   low-rank correction to the per-token embeddings.  This is the same idea
   as LoRA (Hu et al., 2021): rather than storing a full weight matrix we
   only store two thin factors, keeping memory usage small.
3. **head** — a tiny linear projection that converts the correction into
   per-cell colour logits.

Retrieval uses scaled-dot-product soft attention so that gradients flow
through the memory bank and the whole system is end-to-end trainable.

**Memory strength (biologically-inspired decay & reinforcement):**

Each slot carries a *strength* scalar in ``[0, 1]`` that modulates its
influence during retrieval — weak memories contribute less, strong ones
dominate.  Strength is derived from two independent signals:

- **Frequency** — how often the slot is activated (cumulative retrieval
  score, normalised to ``[0, 1]``).  Frequently triggered memories build
  up a high frequency score.
- **Recency** — how recently the slot was last strongly activated.  A
  per-slot step counter tracks the last activation time, and a learnable
  recency half-life controls how quickly the recency signal decays.

The final strength is ``frequency_score * recency_score``.  Both the
**decay rate** (which governs how fast frequency fades each step) and the
**reinforcement rate** (which governs how much a retrieval boosts
frequency) are *learnable parameters* — the model can discover its own
optimal forgetting/consolidation dynamics via gradient descent.

Slots whose strength falls below a configurable threshold are considered
"forgotten" and are recycled: their parameters are re-initialised with
small random values and given a moderate starting strength, making room
for newly proposed rules.

The module also maintains a running *utility* estimate (exponential moving
average of each slot's mean retrieval score).  Low-utility slots are
recycled by the :class:`~fusion_model.rule_engine.RuleGenerator` commitment
mechanism: when the generator proposes a rule, it is **soft-blended** into
the lowest-utility slot via :meth:`commit_rule` using a learned commit
weight, giving the bank a warm start for newly discovered patterns while
preserving existing slot content proportionally.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RuleMemory(nn.Module):
    """Fixed-size bank of learnable low-rank rule slots with biologically-inspired
    memory strength dynamics.

    Now operates on **per-token spatial embeddings** rather than a single
    pooled vector.  Retrieval scores are computed from the pooled embedding
    ``h`` (one score vector per sample), while the low-rank corrections and
    classification heads are applied independently to every spatial token.

    :param embed_dim: Dimensionality of the shared input embedding.
    :param num_colours: Number of per-cell colour classes (10 for ARC).
    :param num_slots: How many rule slots to allocate.
    :param rank: Inner rank of each rule's low-rank decomposition ``A @ (B @ x)``.
    :param prune_threshold: Strength below which a slot is considered
        "forgotten" and eligible for recycling.
    :param prune_every_n_steps: How often (in forward passes) to run the
        pruning sweep.  Set to 0 to disable automatic pruning.
    :param recency_activation_threshold: Minimum batch-mean retrieval score
        for a slot to count as "recently activated" (resets its recency
        timer).
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_colours: int = 10,
        num_slots: int = 16,
        rank: int = 16,
        prune_threshold: float = 0.05,
        prune_every_n_steps: int = 100,
        recency_activation_threshold: float = 0.1,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_colours = num_colours
        self.num_slots = num_slots
        self.rank = rank
        self.prune_threshold = prune_threshold
        self.prune_every_n_steps = prune_every_n_steps
        self.recency_activation_threshold = recency_activation_threshold

        # --- Learnable rule bank ---
        self.keys = nn.Parameter(torch.randn(num_slots, embed_dim) * 0.02)

        # Low-rank factors: correction = A @ (B @ x_token).
        self.B = nn.Parameter(torch.zeros(num_slots, rank, embed_dim))
        self.A = nn.Parameter(torch.randn(num_slots, embed_dim, rank) * 0.02)

        # Per-slot classification head: maps embed_dim → num_colours.
        self.heads = nn.Linear(embed_dim, num_colours * num_slots, bias=False)

        # Running utility score per slot (not trained — purely diagnostic).
        self.utility: torch.Tensor
        self.register_buffer("utility", torch.zeros(num_slots))

        # ── Memory strength: frequency + recency ─────────────────────────
        self.decay_rate_logit = nn.Parameter(torch.tensor(math.log(0.999 / 0.001)))
        self.reinforce_rate_logit = nn.Parameter(torch.tensor(math.log(0.01 / 0.99)))
        self.recency_halflife_log = nn.Parameter(torch.tensor(math.log(500.0)))

        self.frequency: torch.Tensor
        self.register_buffer("frequency", torch.full((num_slots,), 0.5))
        self.steps_since_activation: torch.Tensor
        self.register_buffer("steps_since_activation", torch.zeros(num_slots))
        self.step_counter: torch.Tensor
        self.register_buffer("step_counter", torch.tensor(0, dtype=torch.long))

    # ── Strength computation ────────────────────────────────────────────

    def get_strength(self) -> torch.Tensor:
        """Compute per-slot memory strength as ``frequency_score * recency_score``.

        :return: Strength tensor of shape ``(num_slots,)`` in ``[0, 1]``.
        """
        freq_score = self.frequency.clamp(0.0, 1.0)

        half_life = self.recency_halflife_log.exp().clamp(min=1.0)
        recency_score = torch.exp(-math.log(2.0) * self.steps_since_activation / half_life)

        return freq_score * recency_score

    # ── Forward pass ──────────────────────────────────────────────────────

    def forward(
        self, x: torch.Tensor, h: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Retrieve relevant rules and produce per-cell memory-pathway logits.

        :param x: Spatial token embeddings ``(batch, seq, embed_dim)``.
        :param h: Pooled embedding ``(batch, embed_dim)`` used for retrieval
            key matching.
        :return: Tuple of ``(logits_mem, mem_repr, retrieval_info)`` where
            ``logits_mem`` has shape ``(batch, seq, num_colours)``,
            ``mem_repr`` is a pooled representation ``(batch, embed_dim)``
            for the router, and ``retrieval_info`` is a dict with
            ``scores`` and ``strength``.
        """
        # -- 1. Retrieval scores from pooled h --
        raw_scores = torch.matmul(h, self.keys.t()) / (self.embed_dim**0.5)
        raw_scores = F.softmax(raw_scores, dim=-1)  # (B, S)

        # -- 2. Gate by memory strength --
        strength = self.get_strength()  # (S,)
        gated_scores = raw_scores * strength.unsqueeze(0)
        scores = gated_scores / (gated_scores.sum(dim=-1, keepdim=True) + 1e-8)  # (B, S)

        # -- 3. Per-token low-rank corrections --
        # compressed(b, t, s, r) = B(s, r, e) · x(b, t, e)
        compressed = torch.einsum("sre, bte -> btsr", self.B, x)
        # correction(b, t, s, e) = A(s, e, r) · compressed(b, t, s, r)
        correction = torch.einsum("ser, btsr -> btse", self.A, compressed)

        # -- 4. Blend corrections using scores --
        # blended(b, t, e) = Σ_s scores(b, s) · correction(b, t, s, e)
        blended = torch.einsum("bs, btse -> bte", scores, correction)

        # -- 5. Per-token classification via per-slot heads --
        w_heads = self.heads.weight.view(self.num_slots, self.num_colours, self.embed_dim)
        # slot_logits(b, t, s, c) = correction(b, t, s, e) · W(s, c, e)
        slot_logits = torch.einsum("btse, sce -> btsc", correction, w_heads)
        # logits(b, t, c) = Σ_s scores(b, s) · slot_logits(b, t, s, c)
        logits_mem = torch.einsum("bs, btsc -> btc", scores, slot_logits)

        # -- 6. Router representation: mean-pool blended correction --
        mem_repr = blended.mean(dim=1)  # (B, E)

        # -- 7. Update strength signals (training only) --
        if self.training:
            with torch.no_grad():
                batch_mean_scores = raw_scores.mean(dim=0)  # (S,)

                decay_rate = torch.sigmoid(self.decay_rate_logit)
                reinforce_rate = torch.sigmoid(self.reinforce_rate_logit)
                self.frequency = (
                    self.frequency * decay_rate + reinforce_rate * batch_mean_scores
                ).clamp(max=1.0)

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
            "strength": strength.detach(),
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
