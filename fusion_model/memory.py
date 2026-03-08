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

The module maintains a running *utility* estimate (exponential moving average
of each slot's mean retrieval score).  Low-utility slots are recycled by the
:class:`~fusion_model.rule_engine.RuleGenerator` commitment mechanism: when
the generator proposes a high-confidence rule, it is written into the
lowest-utility slot via :meth:`commit_rule`, giving the bank a warm start
for newly discovered patterns.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RuleMemory(nn.Module):
    """Fixed-size bank of learnable low-rank rule slots.

    Slots are initialised randomly and refined via gradient descent.
    Additionally, the :class:`~fusion_model.rule_engine.RuleGenerator` may
    *commit* high-confidence proposed rules into the bank by overwriting the
    lowest-utility slot (see :meth:`commit_rule`).

    :param embed_dim: Dimensionality of the shared input embedding.
    :param num_classes: Number of output classes (CLEVR answers).
    :param num_slots: How many rule slots to allocate.
    :param rank: Inner rank of each rule's low-rank decomposition ``A @ (B @ h)``.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_classes: int = 28,
        num_slots: int = 16,
        rank: int = 16,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_slots = num_slots
        self.rank = rank

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

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Retrieve relevant rules and produce memory-pathway logits.

        :param h: Shared input embedding of shape ``(batch, embed_dim)``.
        :return: Tuple of ``(logits_mem, blended_correction, retrieval_info)``
            where ``logits_mem`` has shape ``(batch, num_classes)``,
            ``blended_correction`` is the score-weighted sum of per-slot
            corrections ``(batch, embed_dim)`` before the classification
            heads (used by the :class:`~fusion_model.decision.DecisionRouter`),
            and ``retrieval_info`` is a dict with ``scores``.
        """
        # -- 1. Compute relevance of every rule slot via scaled dot-product. --
        # Shape: (batch, num_slots)
        scores = torch.matmul(h, self.keys.t()) / (self.embed_dim**0.5)
        scores = F.softmax(scores, dim=-1)

        # -- 2. Apply every rule's low-rank correction in parallel. --
        # "einsum" lets us batch the per-slot matrix multiplies efficiently.
        #   compressed(b, s, r) = B(s, r, e) · h(b, e)
        compressed = torch.einsum("sre, be -> bsr", self.B, h)
        #   correction(b, s, e) = A(s, e, r) · compressed(b, s, r)
        correction = torch.einsum("ser, bsr -> bse", self.A, compressed)

        # -- 3. Blend corrections using relevance scores. --
        # blended(b, e) = Σ_s  scores(b, s) · correction(b, s, e)
        blended_correction = torch.einsum("bs, bse -> be", scores, correction)

        # -- 4. Project each slot's correction to answer logits. --
        # Reshape the packed weight into (num_slots, num_classes, embed_dim).
        w_heads = self.heads.weight.view(self.num_slots, -1, self.embed_dim)
        # slot_logits(b, s, c) = correction(b, s, e) · W(s, c, e)
        slot_logits = torch.einsum("bse, sce -> bsc", correction, w_heads)

        # -- 5. Blend slot logits using their relevance scores. --
        # logits_mem(b, c) = Σ_s  scores(b, s) · slot_logits(b, s, c)
        logits_mem = torch.einsum("bs, bsc -> bc", scores, slot_logits)

        # -- 6. Track per-slot utility (exponential moving average). --
        if self.training:
            with torch.no_grad():
                batch_utility = scores.mean(dim=0)
                self.utility = self.utility * 0.99 + 0.01 * batch_utility

        retrieval_info: dict[str, torch.Tensor] = {
            "scores": scores,
        }
        return logits_mem, blended_correction, retrieval_info

    # ── Rule commitment ──────────────────────────────────────────────────

    def get_lowest_utility_slot(self) -> int:
        """Return the index of the rule slot with the lowest utility score."""
        return int(self.utility.argmin().item())

    @torch.no_grad()
    def commit_rule(
        self,
        slot_idx: int,
        key: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
    ) -> None:
        """Overwrite a slot with a proposed rule.

        :param slot_idx: Target slot index in ``[0, num_slots)``.
        :param key: Trigger embedding ``(embed_dim,)``.
        :param A: Low-rank factor ``(embed_dim, rank)``.
        :param B: Low-rank factor ``(rank, embed_dim)``.
        """
        self.keys.data[slot_idx].copy_(key)
        self.A.data[slot_idx].copy_(A)
        self.B.data[slot_idx].copy_(B)
        self.utility[slot_idx] = 0.0
