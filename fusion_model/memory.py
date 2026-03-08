"""RuleMemory — a fixed bank of learnable low-rank rules with differentiable retrieval.

**High-level idea:**  Instead of modifying the base network's weights to
encode every pattern, we store a set of small *corrections* ("rules") in a
memory bank.  At inference time the model looks up which rules are relevant
to the current input and blends their corrections together.

Each *rule slot* stores three things:

1. **key** — a trigger embedding that determines *when* the rule fires.
   The model computes cosine-like attention between the input embedding and
   every key to decide relevance.
2. **A, B** — a pair of small matrices whose product ``A @ (B @ h)`` forms a
   low-rank correction to the embedding.  This is the same idea as LoRA
   (Hu et al., 2021): rather than storing a full weight matrix we only
   store two thin factors, keeping memory usage small.
3. **head** — a tiny linear projection that converts the correction into
   classification logits.

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

    Slots are initialised randomly and refined via gradient descent.
    Additionally, the :class:`~fusion_model.rule_engine.RuleGenerator` may
    *commit* proposed rules into the bank by **soft-blending** them into the
    lowest-utility slot (see :meth:`commit_rule`), where the blend weight is
    determined by the generator's learned commit weight.

    **Memory strength** modulates each slot's influence during retrieval.
    It is the product of two independent signals:

    - *Frequency score* — a running accumulator of how often the slot is
      retrieved, decayed each step by a **learnable decay rate** and boosted
      by a **learnable reinforcement rate** proportional to the batch-mean
      retrieval score.
    - *Recency score* — an exponential decay based on how many steps have
      elapsed since the slot was last strongly activated, governed by a
      **learnable recency half-life**.

    Both rates are stored as unconstrained logits and mapped to ``(0, 1)``
    via sigmoid so that standard optimisers can tune them without constraint
    handling.

    :param embed_dim: Dimensionality of the shared input embedding.
    :param num_classes: Number of output classes (CLEVR answers).
    :param num_slots: How many rule slots to allocate.
    :param rank: Inner rank of each rule's low-rank decomposition ``A @ (B @ h)``.
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
        num_classes: int = 28,
        num_slots: int = 16,
        rank: int = 16,
        prune_threshold: float = 0.05,
        prune_every_n_steps: int = 100,
        recency_activation_threshold: float = 0.1,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_slots = num_slots
        self.rank = rank
        self.prune_threshold = prune_threshold
        self.prune_every_n_steps = prune_every_n_steps
        self.recency_activation_threshold = recency_activation_threshold

        # --- Learnable rule bank ---
        # Each key is a trigger vector that the input is compared against.
        self.keys = nn.Parameter(torch.randn(num_slots, embed_dim) * 0.02)

        # Low-rank factors for each slot: correction = A @ (B @ h).
        # B compresses the embedding; A expands the compressed representation.
        self.B = nn.Parameter(torch.zeros(num_slots, rank, embed_dim))
        self.A = nn.Parameter(torch.randn(num_slots, embed_dim, rank) * 0.02)

        # Per-slot classification head packed into a single Linear for efficiency.
        # Its weight has shape (num_classes * num_slots, embed_dim).
        self.heads = nn.Linear(embed_dim, num_classes * num_slots, bias=False)

        # Running utility score per slot (not trained — purely diagnostic).
        self.utility: torch.Tensor
        self.register_buffer("utility", torch.zeros(num_slots))

        # ── Memory strength: frequency + recency ─────────────────────────
        # Learnable decay rate for the frequency accumulator.  Stored as an
        # unconstrained logit; the effective rate is sigmoid(logit) which
        # lives in (0, 1).  Initialised so that sigmoid(logit) ≈ 0.999.
        self.decay_rate_logit = nn.Parameter(torch.tensor(math.log(0.999 / 0.001)))

        # Learnable reinforcement rate (same logit trick).
        # Initialised so that sigmoid(logit) ≈ 0.01.
        self.reinforce_rate_logit = nn.Parameter(torch.tensor(math.log(0.01 / 0.99)))

        # Learnable recency half-life (in steps).  Stored as log to keep it
        # positive; effective half-life = exp(logit).  Initialised to ~500
        # steps.
        self.recency_halflife_log = nn.Parameter(torch.tensor(math.log(500.0)))

        # Per-slot frequency accumulator (non-gradient running state).
        # Initialised to 0.5 so that fresh slots start with moderate strength
        # rather than being immediately eligible for pruning.
        self.frequency: torch.Tensor
        self.register_buffer("frequency", torch.full((num_slots,), 0.5))

        # Per-slot step counter since last strong activation.
        self.steps_since_activation: torch.Tensor
        self.register_buffer("steps_since_activation", torch.zeros(num_slots))

        # Global forward-pass counter for periodic pruning.
        self.step_counter: torch.Tensor
        self.register_buffer("step_counter", torch.tensor(0, dtype=torch.long))

    # ── Strength computation ────────────────────────────────────────────

    def get_strength(self) -> torch.Tensor:
        """Compute per-slot memory strength as ``frequency_score * recency_score``.

        - **Frequency score**: the raw frequency accumulator clamped to
          ``[0, 1]``.  Higher values mean the slot has been triggered more
          often (relative to decay).
        - **Recency score**: ``exp(-ln(2) * steps_since_activation / half_life)``
          — an exponential decay that halves every ``half_life`` steps.

        :return: Strength tensor of shape ``(num_slots,)`` in ``[0, 1]``.
        """
        freq_score = self.frequency.clamp(0.0, 1.0)

        half_life = self.recency_halflife_log.exp().clamp(min=1.0)
        recency_score = torch.exp(
            -math.log(2.0) * self.steps_since_activation / half_life
        )

        return freq_score * recency_score

    # ── Forward pass ──────────────────────────────────────────────────────

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Retrieve relevant rules and produce memory-pathway logits.

        Retrieval scores are **gated by memory strength** so that weak
        (infrequent / stale) memories contribute less to the output, and
        strong (frequent / recent) memories dominate.

        :param h: Shared input embedding of shape ``(batch, embed_dim)``.
        :return: Tuple of ``(logits_mem, blended_correction, retrieval_info)``
            where ``logits_mem`` has shape ``(batch, num_classes)``,
            ``blended_correction`` is the score-weighted sum of per-slot
            corrections ``(batch, embed_dim)`` before the classification
            heads (used by the :class:`~fusion_model.decision.DecisionRouter`),
            and ``retrieval_info`` is a dict with ``scores`` (the
            strength-gated, re-normalised retrieval distribution) and
            ``strength`` (per-slot strength values).
        """
        # -- 1. Compute relevance of every rule slot via scaled dot-product. --
        # Shape: (batch, num_slots)
        raw_scores = torch.matmul(h, self.keys.t()) / (self.embed_dim**0.5)
        raw_scores = F.softmax(raw_scores, dim=-1)

        # -- 2. Gate retrieval scores by memory strength. --
        # Weak slots are suppressed; strong slots are amplified.
        strength = self.get_strength()  # (num_slots,)
        gated_scores = raw_scores * strength.unsqueeze(0)
        scores = gated_scores / (gated_scores.sum(dim=-1, keepdim=True) + 1e-8)

        # -- 3. Apply every rule's low-rank correction in parallel. --
        # "einsum" lets us batch the per-slot matrix multiplies efficiently.
        #   compressed(b, s, r) = B(s, r, e) · h(b, e)
        compressed = torch.einsum("sre, be -> bsr", self.B, h)
        #   correction(b, s, e) = A(s, e, r) · compressed(b, s, r)
        correction = torch.einsum("ser, bsr -> bse", self.A, compressed)

        # -- 4. Blend corrections using strength-gated scores. --
        # blended(b, e) = Σ_s  scores(b, s) · correction(b, s, e)
        blended_correction = torch.einsum("bs, bse -> be", scores, correction)

        # -- 5. Project each slot's correction to answer logits. --
        # Reshape the packed weight into (num_slots, num_classes, embed_dim).
        w_heads = self.heads.weight.view(self.num_slots, -1, self.embed_dim)
        # slot_logits(b, s, c) = correction(b, s, e) · W(s, c, e)
        slot_logits = torch.einsum("bse, sce -> bsc", correction, w_heads)

        # -- 6. Blend slot logits using strength-gated scores. --
        # logits_mem(b, c) = Σ_s  scores(b, s) · slot_logits(b, s, c)
        logits_mem = torch.einsum("bs, bsc -> bc", scores, slot_logits)

        # -- 7. Update strength signals and utility (training only). --
        if self.training:
            with torch.no_grad():
                batch_mean_scores = raw_scores.mean(dim=0)  # (num_slots,)

                # Frequency: decay then reinforce.
                decay_rate = torch.sigmoid(self.decay_rate_logit)
                reinforce_rate = torch.sigmoid(self.reinforce_rate_logit)
                self.frequency = (
                    self.frequency * decay_rate + reinforce_rate * batch_mean_scores
                ).clamp(max=1.0)

                # Recency: increment all counters, reset strongly activated slots.
                self.steps_since_activation += 1
                activated = batch_mean_scores > self.recency_activation_threshold
                self.steps_since_activation[activated] = 0.0

                # Utility EMA (unchanged — still used for slot selection).
                self.utility = self.utility * 0.99 + 0.01 * batch_mean_scores

                # Step counter for periodic pruning.
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
        return logits_mem, blended_correction, retrieval_info

    # ── Slot selection ──────────────────────────────────────────────────

    def get_weakest_slot(self) -> int:
        """Return the index of the slot with the lowest combined strength.

        This is the preferred target for rule commitment — the weakest
        memory is the one most worth overwriting.  Falls back to utility
        as a tiebreaker when multiple slots have identical strength.
        """
        strength = self.get_strength()
        # Tiebreak with utility so that among equally-weak slots we pick
        # the least useful one.
        combined = strength + 1e-6 * self.utility
        return int(combined.argmin().item())

    def get_lowest_utility_slot(self) -> int:
        """Return the index of the rule slot with the lowest utility score.

        .. deprecated::
            Prefer :meth:`get_weakest_slot` which accounts for both
            frequency and recency via the strength mechanism.
        """
        return int(self.utility.argmin().item())

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

        When ``commit_weight`` is 1.0 the slot is fully overwritten (legacy
        behaviour).  For values in (0, 1) the slot parameters are linearly
        interpolated: ``slot = w * proposed + (1 - w) * slot``, preserving
        existing content proportionally.

        The committed slot's strength signals are also updated: frequency
        is set to at least the commit weight (giving the new rule a warm
        start), and the recency timer is reset so the slot is considered
        freshly activated.

        :param slot_idx: Target slot index in ``[0, num_slots)``.
        :param key: Trigger embedding ``(embed_dim,)``.
        :param A: Low-rank factor ``(embed_dim, rank)``.
        :param B: Low-rank factor ``(rank, embed_dim)``.
        :param commit_weight: Blend weight in ``[0, 1]``.  1.0 = full
            overwrite, 0.0 = no change.
        """
        w = commit_weight
        self.keys.data[slot_idx].lerp_(key, w)
        self.A.data[slot_idx].lerp_(A, w)
        self.B.data[slot_idx].lerp_(B, w)
        self.utility[slot_idx] *= 1.0 - w

        # Give the newly committed rule a fair starting strength.
        self.frequency[slot_idx] = max(w, self.frequency[slot_idx].item())
        self.steps_since_activation[slot_idx] = 0.0

    # ── Pruning ──────────────────────────────────────────────────────────

    @torch.no_grad()
    def prune_weak_slots(self, threshold: float | None = None) -> int:
        """Recycle slots whose strength has decayed below *threshold*.

        Pruned slots have their parameters re-initialised with small random
        values and are given a moderate starting frequency (0.5) so they
        are neither immediately dominant nor immediately pruned again.

        :param threshold: Override for the instance-level
            ``prune_threshold``.  If ``None``, uses ``self.prune_threshold``.
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
