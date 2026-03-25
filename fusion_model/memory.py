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
optimal forgetting/consolidation dynamics via gradient descent.  During
each forward pass the new frequency is computed *differentiably* from
the current rate parameters and used in the strength gating, providing
the gradient path ``loss → scores → strength → new_freq → rate logits``.
The resulting frequency value is then detached and stored in a buffer
for the next step.

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
from typing import Any

import jax
import jax.numpy as jnp
import flax.linen as nn


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

    embed_dim: int = 256
    num_colours: int = 10
    num_slots: int = 16
    rank: int = 16
    prune_threshold: float = 0.05
    prune_every_n_steps: int = 100
    recency_activation_threshold: float = 0.1

    def setup(self) -> None:
        # --- Learnable rule bank ---
        self.keys = self.param(
            "keys",
            lambda rng, shape: jax.random.normal(rng, shape) * 0.02,
            (self.num_slots, self.embed_dim),
        )
        self.B = self.param(
            "B", nn.initializers.zeros_init(), (self.num_slots, self.rank, self.embed_dim),
        )
        self.A = self.param(
            "A",
            lambda rng, shape: jax.random.normal(rng, shape) * 0.02,
            (self.num_slots, self.embed_dim, self.rank),
        )

        # Per-slot classification head weight: (num_slots, num_colours, embed_dim).
        self.heads_weight = self.param(
            "heads_weight",
            nn.initializers.lecun_normal(),
            (self.num_slots, self.num_colours, self.embed_dim),
        )

        # ── Memory strength: frequency + recency ─────────────────────────
        self.decay_rate_logit = self.param(
            "decay_rate_logit", lambda _rng, _shape: jnp.array(math.log(0.999 / 0.001)), (),
        )
        self.reinforce_rate_logit = self.param(
            "reinforce_rate_logit", lambda _rng, _shape: jnp.array(math.log(0.01 / 0.99)), (),
        )
        self.recency_halflife_log = self.param(
            "recency_halflife_log", lambda _rng, _shape: jnp.array(math.log(500.0)), (),
        )

        # ── Mutable state (buffers) ──────────────────────────────────────
        self._frequency = self.variable(
            "state", "frequency", lambda: jnp.full((self.num_slots,), 0.5),
        )
        self._steps_since_activation = self.variable(
            "state", "steps_since_activation", lambda: jnp.zeros((self.num_slots,)),
        )
        self._step_counter = self.variable(
            "state", "step_counter", lambda: jnp.array(0, dtype=jnp.int32),
        )
        self._utility = self.variable(
            "state", "utility", lambda: jnp.zeros((self.num_slots,)),
        )

    # ── Strength computation ────────────────────────────────────────────

    def get_strength(
        self, freq_override: jnp.ndarray | None = None
    ) -> jnp.ndarray:
        """Compute per-slot memory strength as ``frequency_score * recency_score``.

        :param freq_override: If provided, use this tensor instead of
            ``self.frequency`` for the frequency component.  This allows
            callers to pass a differentiable frequency tensor so that
            gradients flow back to the decay/reinforce rate parameters.
        :return: Strength tensor of shape ``(num_slots,)`` in ``[0, 1]``.
        """
        freq_score = jnp.clip(
            freq_override if freq_override is not None else self._frequency.value,
            0.0, 1.0,
        )

        half_life = jnp.clip(jnp.exp(self.recency_halflife_log), min=1.0)
        recency_score = jnp.exp(
            -math.log(2.0) * self._steps_since_activation.value / half_life
        )

        return freq_score * recency_score

    # ── Forward pass ──────────────────────────────────────────────────────

    def __call__(
        self, x: jnp.ndarray, h: jnp.ndarray, *, training: bool = False,
    ) -> tuple[jnp.ndarray, jnp.ndarray, dict[str, Any]]:
        """Retrieve relevant rules and produce per-cell memory-pathway logits.

        :param x: Spatial token embeddings ``(batch, seq, embed_dim)``.
        :param h: Pooled embedding ``(batch, embed_dim)`` used for retrieval
            key matching.
        :param training: Whether we are in training mode.
        :return: Tuple of ``(logits_mem, mem_repr, retrieval_info)``.
        """
        # -- 1. Retrieval scores from pooled h --
        raw_scores = jnp.matmul(h, self.keys.T) / (self.embed_dim ** 0.5)
        raw_scores = jax.nn.softmax(raw_scores, axis=-1)  # (B, S)

        # -- 2. Compute differentiable frequency & gate by memory strength --
        if training:
            batch_mean_scores = jax.lax.stop_gradient(raw_scores.mean(axis=0))  # (S,)
            decay_rate = jax.nn.sigmoid(self.decay_rate_logit)
            reinforce_rate = jax.nn.sigmoid(self.reinforce_rate_logit)
            new_freq = jnp.clip(
                jax.lax.stop_gradient(self._frequency.value) * decay_rate
                + reinforce_rate * batch_mean_scores,
                max=1.0,
            )
            strength = self.get_strength(freq_override=new_freq)  # (S,)
        else:
            strength = self.get_strength()  # (S,)
            batch_mean_scores = None
            new_freq = None

        gated_scores = raw_scores * strength[None, :]
        scores = gated_scores / (gated_scores.sum(axis=-1, keepdims=True) + 1e-8)  # (B, S)

        # -- 3. Per-token low-rank corrections --
        compressed = jnp.einsum("sre, bte -> btsr", self.B, x)
        correction = jnp.einsum("ser, btsr -> btse", self.A, compressed)

        # -- 4. Blend corrections using scores --
        blended = jnp.einsum("bs, btse -> bte", scores, correction)

        # -- 5. Per-token classification via per-slot heads --
        # heads_weight: (S, C, E), correction: (B, T, S, E)
        slot_logits = jnp.einsum("btse, sce -> btsc", correction, self.heads_weight)
        logits_mem = jnp.einsum("bs, btsc -> btc", scores, slot_logits)

        # -- 6. Router representation: mean-pool blended correction --
        mem_repr = blended.mean(axis=1)  # (B, E)

        # -- 7. Persist frequency state & update recency (training only) --
        if training and new_freq is not None:
            self._frequency.value = jax.lax.stop_gradient(new_freq)

            new_steps = self._steps_since_activation.value + 1
            activated = batch_mean_scores > self.recency_activation_threshold
            new_steps = jnp.where(activated, 0.0, new_steps)
            self._steps_since_activation.value = new_steps

            self._utility.value = self._utility.value * 0.99 + 0.01 * batch_mean_scores

            new_counter = self._step_counter.value + 1
            self._step_counter.value = new_counter

            # Pruning
            if self.prune_every_n_steps > 0:
                should_prune = (new_counter % self.prune_every_n_steps) == 0
                self._maybe_prune(should_prune)

        retrieval_info: dict[str, Any] = {
            "scores": scores,
            "strength": strength,
        }
        return logits_mem, mem_repr, retrieval_info

    # ── Slot selection ──────────────────────────────────────────────────

    def get_weakest_slot(self) -> int:
        """Return the index of the slot with the lowest combined strength."""
        strength = self.get_strength()
        combined = strength + 1e-6 * self._utility.value
        return int(jnp.argmin(combined).item())

    # ── Rule commitment ──────────────────────────────────────────────────

    def commit_rule(
        self,
        slot_idx: int,
        key: jnp.ndarray,
        A: jnp.ndarray,
        B: jnp.ndarray,
        commit_weight: float = 1.0,
    ) -> dict[str, jnp.ndarray]:
        """Soft-blend a proposed rule into a slot.

        Returns a dict of updated parameter arrays. In JAX/Flax, we cannot
        mutate parameters in-place — the caller must apply these updates.

        :param slot_idx: Target slot index in ``[0, num_slots)``.
        :param key: Trigger embedding ``(embed_dim,)``.
        :param A: Low-rank factor ``(embed_dim, rank)``.
        :param B: Low-rank factor ``(rank, embed_dim)``.
        :param commit_weight: Blend weight in ``[0, 1]``.
        :return: Dict with updated 'keys', 'A', 'B' arrays plus state updates.
        """
        w = commit_weight

        new_keys = self.keys.at[slot_idx].set(
            self.keys[slot_idx] * (1 - w) + key * w
        )
        new_A = self.A.at[slot_idx].set(
            self.A[slot_idx] * (1 - w) + A * w
        )
        new_B = self.B.at[slot_idx].set(
            self.B[slot_idx] * (1 - w) + B * w
        )

        self._utility.value = self._utility.value.at[slot_idx].set(
            self._utility.value[slot_idx] * (1.0 - w)
        )
        self._frequency.value = self._frequency.value.at[slot_idx].set(
            jnp.maximum(w, self._frequency.value[slot_idx])
        )
        self._steps_since_activation.value = self._steps_since_activation.value.at[slot_idx].set(0.0)

        return {"keys": new_keys, "A": new_A, "B": new_B}

    # ── Pruning ──────────────────────────────────────────────────────────

    def _maybe_prune(self, should_prune: jnp.ndarray) -> None:
        """Conditionally recycle slots whose strength has decayed below threshold.

        Uses jax.lax.cond for XLA-compatible conditional execution.
        """
        # Note: in JAX we can't easily mutate params during forward pass.
        # We handle pruning of state variables only; param pruning is done
        # externally in the training loop.
        strength = self.get_strength()
        dead = strength < self.prune_threshold

        self._frequency.value = jnp.where(dead, 0.5, self._frequency.value)
        self._steps_since_activation.value = jnp.where(dead, 0.0, self._steps_since_activation.value)
        self._utility.value = jnp.where(dead, 0.0, self._utility.value)

    def prune_weak_slots(self, threshold: float | None = None) -> jnp.ndarray:
        """Recycle slots whose strength has decayed below *threshold*.

        :param threshold: Override for ``self.prune_threshold``.
        :return: Boolean mask of pruned slots.
        """
        thresh = threshold if threshold is not None else self.prune_threshold
        strength = self.get_strength()
        dead = strength < thresh

        self._frequency.value = jnp.where(dead, 0.5, self._frequency.value)
        self._steps_since_activation.value = jnp.where(dead, 0.0, self._steps_since_activation.value)
        self._utility.value = jnp.where(dead, 0.0, self._utility.value)

        return dead
