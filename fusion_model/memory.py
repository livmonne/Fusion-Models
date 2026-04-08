"""RuleMemory — a fixed bank of learnable low-rank rules with differentiable retrieval.

**High-level idea:**  Instead of modifying the base network's weights to
encode every pattern, we store a set of small *corrections* ("rules") in a
memory bank.  At inference time the model looks up which rules are relevant
to the current input and blends their corrections together.

Each *rule slot* stores three things:

1. **key** — a trigger embedding that determines *when* the rule fires.
   **Multi-head cross-attention** between the pooled input embedding and
   every key allows different heads to specialise on different aspects of
   relevance (e.g. spatial layout vs colour transformation).
2. **A, B** — a pair of small matrices whose product ``A @ (B @ x)`` forms a
   low-rank correction to the per-token embeddings.  This is the same idea
   as LoRA (Hu et al., 2021): rather than storing a full weight matrix we
   only store two thin factors, keeping memory usage small.  During the
   forward pass, retrieval scores are folded into the A/B matrices *before*
   expanding over the spatial dimension, so peak memory is ``O(B × seq × E)``
   instead of the naïve ``O(B × seq × S × E)``.
3. **head** — a tiny linear projection that converts the correction into
   per-cell colour logits.

Retrieval uses **multi-head** scaled-dot-product cross-attention so that
gradients flow through the memory bank and the whole system is end-to-end
trainable.  A learned head-combination vector merges per-head attention
distributions into the final per-slot retrieval scores.

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

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadMemoryCrossAttention(nn.Module):
    """Multi-head cross-attention for memory slot retrieval.

    This module replaces a simple single-head dot-product between the pooled
    task embedding ``h`` and the memory slot keys.  Instead, it projects both
    the query (``h``) and the keys/values (slot key bank) into multiple
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

    **Architecture (per forward call):**

    ::

        h (B, E)       ──► q_proj ──► Q (B, H, D)  ─┐
                                                      ├─► attn_logits (B, H, S)
        keys (S, E)    ──► k_proj ──► K (S, H, D)  ─┘     │
                       ──► v_proj ──► V (S, H, D)         │
                                                      softmax
                                                           │
                                                    head_attn (B, H, S)
                                                      │         │
                                              einsum w/ V    einsum w/ head_combine
                                                      │         │
                                              retrieved (B,H,D)  scores (B, S)
                                                      │
                                              reshape → out_proj
                                                      │
                                              context (B, E)

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
        # for attention Q/K/V projections (bias can shift attention in
        # undesirable ways and adds no expressivity here).
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        # Output projection: recombines the concatenated per-head
        # retrieved vectors back into a single embed_dim vector.
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # Learned head-combination weights.  Initialised to uniform
        # (all ones → softmax gives 1/H each) so that early training
        # uses all heads equally.  The model can later learn to
        # up-weight the most informative heads.
        self.head_combine = nn.Parameter(torch.ones(num_heads))

    def forward(
        self, h: torch.Tensor, keys: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute multi-head cross-attention over the memory slot bank.

        :param h: Pooled task embedding of shape ``(batch, embed_dim)``.
            This serves as the *query* — "what kind of rules does the
            current input need?"
        :param keys: Memory slot key bank of shape ``(num_slots, embed_dim)``.
            Each row is a learnable trigger embedding.  These serve as
            both *keys* (for matching) and *values* (for retrieval
            content), each through their own projection.
        :return: A 3-tuple ``(scores, context, head_attn)``:

            - **scores** ``(batch, num_slots)`` — combined retrieval
              weights that sum to 1 over the slot dimension.  These
              replace the old single-head softmax scores.
            - **context** ``(batch, embed_dim)`` — a rich retrieved
              representation formed by the standard multi-head attention
              output (weighted sum of value projections, concatenated
              across heads, then linearly projected).  This is added to
              the memory pathway's router representation so the decision
              router gets a richer signal.
            - **head_attn** ``(batch, num_heads, num_slots)`` — the raw
              per-head softmax attention distributions.  Logged in the
              training metadata for visualisation and analysis (e.g.
              checking whether heads have specialised).
        """
        B = h.shape[0]
        S = keys.shape[0]
        H = self.num_heads
        D = self.head_dim

        # ── Project into multi-head subspaces ──────────────────────────
        # Each projection: (*, embed_dim) → (*, embed_dim), then reshaped
        # into (*, num_heads, head_dim).
        Q = self.q_proj(h).view(B, H, D)      # (B, H, D)
        K = self.k_proj(keys).view(S, H, D)    # (S, H, D)
        V = self.v_proj(keys).view(S, H, D)    # (S, H, D)

        # ── Scaled dot-product attention per head ──────────────────────
        # attn_logits[b, h, s] = (Q[b,h,:] · K[s,h,:]) / √D
        # Scaling by √head_dim prevents the dot products from growing
        # too large in magnitude, which would push softmax into
        # saturated regions with vanishing gradients.
        attn_logits = torch.einsum("bhd,shd->bhs", Q, K) / (D ** 0.5)
        head_attn = F.softmax(attn_logits, dim=-1)  # (B, H, S)

        # ── Retrieve values via attention-weighted sum ─────────────────
        # For each head, compute a weighted sum of value vectors across
        # all slots: retrieved[b, h, d] = Σ_s head_attn[b,h,s] · V[s,h,d]
        retrieved = torch.einsum("bhs,shd->bhd", head_attn, V)  # (B, H, D)

        # Concatenate heads back into full embed_dim and project.
        retrieved = retrieved.reshape(B, self.embed_dim)    # (B, E)
        context = self.out_proj(retrieved)                   # (B, E)

        # ── Combine per-head attention into final slot scores ──────────
        # head_w: (H,) — learned softmax weights over heads.
        # scores[b, s] = Σ_h head_w[h] · head_attn[b, h, s]
        # This produces a single attention distribution over slots that
        # is a convex combination of all heads' distributions.
        head_w = F.softmax(self.head_combine, dim=0)           # (H,)
        scores = torch.einsum("bhs,h->bs", head_attn, head_w)  # (B, S)

        return scores, context, head_attn


class RuleMemory(nn.Module):
    """Fixed-size bank of learnable low-rank rule slots with biologically-inspired
    memory strength dynamics.

    Now operates on **per-token spatial embeddings** rather than a single
    pooled vector.  Retrieval scores are computed from the pooled embedding
    ``h`` via **multi-head cross-attention** over the slot key bank (one
    combined score vector per sample), while the low-rank corrections and
    classification heads are applied independently to every spatial token.

    :param embed_dim: Dimensionality of the shared input embedding.
    :param num_colours: Number of per-cell colour classes (10 for ARC).
    :param num_slots: How many rule slots to allocate.
    :param rank: Inner rank of each rule's low-rank decomposition ``A @ (B @ x)``.
    :param num_retrieval_heads: Number of attention heads for multi-head
        memory retrieval.  Each head independently assesses slot relevance;
        a learned combination merges them into the final per-slot scores.
        Must evenly divide ``embed_dim``.
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
        num_retrieval_heads: int = 4,
        prune_threshold: float = 0.05,
        prune_every_n_steps: int = 100,
        recency_activation_threshold: float = 0.1,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_colours = num_colours
        self.num_slots = num_slots
        self.rank = rank
        self.num_retrieval_heads = num_retrieval_heads
        self.prune_threshold = prune_threshold
        self.prune_every_n_steps = prune_every_n_steps
        self.recency_activation_threshold = recency_activation_threshold

        # --- Learnable rule bank ---
        # Each slot's key is a trigger embedding.  The multi-head cross-
        # attention module projects these through learned K and V
        # transforms, decoupling "what triggers retrieval" from "what
        # information is retrieved."
        self.keys = nn.Parameter(torch.randn(num_slots, embed_dim) * 0.02)

        # Low-rank factors: correction = A @ (B @ x_token).
        self.B = nn.Parameter(torch.zeros(num_slots, rank, embed_dim))
        self.A = nn.Parameter(torch.randn(num_slots, embed_dim, rank) * 0.02)

        # Per-slot classification head: maps embed_dim → num_colours.
        self.heads = nn.Linear(embed_dim, num_colours * num_slots, bias=False)

        # ── Multi-head cross-attention for retrieval ─────────────────────
        # Replaces the old single-head dot-product (h @ keys.T / √d).
        # Each of the `num_retrieval_heads` heads independently assesses
        # which slots are relevant, then a learned combination merges them.
        self.retrieval_attn = MultiHeadMemoryCrossAttention(
            embed_dim=embed_dim,
            num_heads=num_retrieval_heads,
        )

        # LayerNorm stabilises the residual sum of the mean-pooled
        # blended correction and the cross-attention context vector
        # before it's fed to the decision router.
        self.repr_norm = nn.LayerNorm(embed_dim)

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
        # -- 1. Multi-head cross-attention retrieval from pooled h --
        # The cross-attention module projects h (query) and self.keys
        # (key/value) into multiple head subspaces, computes per-head
        # softmax attention, and merges them via a learned combination.
        # Returns:
        #   raw_scores  (B, S) — combined retrieval weights
        #   mem_context (B, E) — retrieved memory context vector
        #   head_attn   (B, H, S) — per-head attention maps
        raw_scores, mem_context, head_attn = self.retrieval_attn(h, self.keys)

        # -- 2. Compute differentiable frequency & gate by memory strength --
        # During training, compute new_freq through the learnable decay/reinforce
        # rate parameters so that gradients flow back to them.
        if self.training:
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

        # -- 3–5. Per-token low-rank corrections, blending, and classification --
        #
        # The naïve approach materialises a (B, seq, S, E) correction
        # tensor — with S=128, seq=900, E=256 this is ~471 MB per
        # micro-batch in float32.  Instead we contract scores into the
        # computation *before* expanding over the spatial dimension,
        # producing intermediates of at most (B, seq, E) or (S, R, E).
        #
        # Key identity: blended(b,t,e) = Σ_s scores(b,s) · A(s,e,r) · B(s,r,e') · x(b,t,e')
        #   = Σ_s A(s,e,r) · [ scores(b,s) · B(s,r,e') · x(b,t,e') ]
        #
        # Step 1: score-weighted B per sample → wB(b, r, e) = Σ_s scores(b,s) · B(s,r,e)
        wB = torch.einsum("bs, sre -> bre", scores, self.B)  # (B, R, E)
        # Step 2: compressed(b, r, t) = wB(b, r, e) · x(b, t, e)^T
        compressed = torch.bmm(wB, x.transpose(1, 2))  # (B, R, T)
        # Step 3: score-weighted A → wA(b, e, r) = Σ_s scores(b,s) · A(s,e,r)
        wA = torch.einsum("bs, ser -> ber", scores, self.A)  # (B, E, R)
        # Step 4: blended(b, t, e) = (wA @ compressed)^T
        blended = torch.bmm(wA, compressed).transpose(1, 2)  # (B, T, E)

        # Classification: similarly fold scores into the per-slot heads
        # before the spatial expansion.
        # w_heads: (S, C, E)  →  wH(b, c, e) = Σ_s scores(b,s) · W(s,c,e)
        w_heads = self.heads.weight.view(self.num_slots, self.num_colours, self.embed_dim)
        wH = torch.einsum("bs, sce -> bce", scores, w_heads)  # (B, C, E)
        # logits_mem(b, t, c) = blended(b, t, e) · wH(b, c, e)
        logits_mem = torch.einsum("bte, bce -> btc", blended, wH)  # (B, T, C)

        # -- 6. Router representation --
        # Combine two complementary signals for the decision router:
        #   (a) Mean-pooled blended correction — the actual per-token
        #       effect of the selected rules, averaged over positions.
        #   (b) mem_context — a summary of *which* slots were selected
        #       and their value projections, from the cross-attention.
        # The residual sum is stabilised by LayerNorm before being
        # passed to the router.
        mem_repr = self.repr_norm(blended.mean(dim=1) + mem_context)  # (B, E)

        # -- 7. Persist frequency state & update recency (training only) --
        if self.training:
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
            # Per-head attention maps (B, H, S) for visualisation /
            # analysis — lets you inspect whether heads have specialised.
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
