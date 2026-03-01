"""FusionLoss — combined training objective for the Fusion Model.

The overall loss is a weighted sum of several terms, each addressing a
different failure mode of the mixture-of-experts architecture:

1. **Task loss** — standard cross-entropy on the combined output.  This is
   the main learning signal.

2. **Guess penalty** — ``mean(alpha_guess)`` pushes the router away from
   relying exclusively on the guess pathway, encouraging it to explore the
   rule-based pathways when they are helpful.

3. **Storage cost** — an approximate L0 over rule-memory slot utilisation.
   Penalises *uniform* usage (which wastes capacity) and rewards sparse
   specialisation.

4. **Negative entropy** — ``-entropy(alpha)`` encourages the router to
   *explore* all three pathways early in training rather than collapsing
   to one.

5. **Auxiliary losses** — independent cross-entropy on each pathway's own
   logits.  This ensures that every pathway receives gradient signal even
   when the router assigns it near-zero weight, preventing "dead" pathways.

6. **Commitment regularisation** — penalises the deviation of the soft
   commitment rate (from the RuleGenerator's proposal confidence vs. its
   learnable threshold) from a target rate.  This provides gradient signal
   to both the proposer's confidence head and the ``commit_threshold_logit``
   parameter, preventing the model from committing too aggressively or
   never committing at all.

All penalty weights (``lambda_*``) are hyperparameters that may need tuning
for different tasks.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


class FusionLoss(nn.Module):
    """Compute the multi-component Fusion Model loss.

    :param lambda_guess: Weight for the guess-pathway penalty.
    :param lambda_storage: Weight for the rule-storage cost.
    :param lambda_entropy: Weight for the (negative) routing-entropy bonus.
    :param lambda_aux: Weight for the auxiliary per-pathway losses.
    :param lambda_commit: Weight for commitment-rate regularisation.
    :param commit_target_rate: Desired fraction of batch elements whose
        proposal confidence exceeds the learnable threshold (soft target).
    """

    def __init__(
        self,
        lambda_guess: float = 0.005,
        lambda_storage: float = 0.001,
        lambda_entropy: float = 0.05,
        lambda_aux: float = 0.3,
        lambda_commit: float = 0.001,
        commit_target_rate: float = 0.1,
    ) -> None:
        super().__init__()
        self.lambda_guess = lambda_guess
        self.lambda_storage = lambda_storage
        self.lambda_entropy = lambda_entropy
        self.lambda_aux = lambda_aux
        self.lambda_commit = lambda_commit
        self.commit_target_rate = commit_target_rate
        self.ce = nn.CrossEntropyLoss()

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        alphas: torch.Tensor,
        retrieval_scores: torch.Tensor,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute the total loss and a breakdown dict.

        :param logits: Combined model logits ``(batch, num_classes)``.
        :param targets: Ground-truth class indices ``(batch,)``.
        :param alphas: Routing weights ``(batch, 3)``.
        :param retrieval_scores: Memory retrieval scores ``(batch, num_slots)``.
        :param metadata: Optional dict with per-pathway logits
            (``logits_mem``, ``logits_rule``, ``logits_guess``).
        :return: Tuple of ``(total_loss, loss_dict)`` where *loss_dict*
            maps component names to their scalar values.
        """
        # -- 1. Primary task loss (cross-entropy on the blended output). --
        task_loss = self.ce(logits, targets)

        # -- 2. Penalise over-reliance on the guess pathway. --
        guess_penalty = alphas[:, 2].mean()

        # -- 3. Encourage sparse rule-slot usage (approximate L0). --
        # slot_usage is the average attention each slot receives across the batch.
        # The product p*(1-p) peaks at 0.5 (uniform) and is zero at 0 or 1.
        slot_usage = retrieval_scores.mean(dim=0)
        storage_cost = (slot_usage * (1.0 - slot_usage)).sum()

        # -- 4. Negative entropy: reward uniform alpha early, prevent collapse. --
        eps = 1e-8
        entropy = -(alphas * (alphas + eps).log()).sum(dim=-1).mean()

        # -- 5. Auxiliary losses keep all three pathways learning. --
        aux_loss = torch.tensor(0.0, device=logits.device)
        if metadata is not None:
            for key in ("logits_mem", "logits_rule", "logits_guess"):
                if key in metadata:
                    aux_loss = aux_loss + self.ce(metadata[key], targets)
            aux_loss = aux_loss / 3.0

        # -- 6. Commitment rate regularisation. --
        # Nudges the soft commitment rate toward a target so the proposer
        # and the learnable threshold receive gradient signal.
        commit_reg = torch.tensor(0.0, device=logits.device)
        if metadata is not None:
            proposal = metadata.get("proposal")
            if proposal is not None:
                prop_conf = proposal["confidence"]  # (batch, 1)
                threshold = proposal["commit_threshold"]  # scalar
                tau = 10.0
                soft_commit = torch.sigmoid(tau * (prop_conf.squeeze(-1) - threshold))
                commit_rate = soft_commit.mean()
                commit_reg = (commit_rate - self.commit_target_rate) ** 2

        # -- Combine everything. --
        total_loss: torch.Tensor = (
            task_loss
            + self.lambda_guess * guess_penalty
            + self.lambda_storage * storage_cost
            - self.lambda_entropy * entropy  # subtract because we *maximise* entropy
            + self.lambda_aux * aux_loss
            + self.lambda_commit * commit_reg
        )

        loss_dict: dict[str, float] = {
            "total": total_loss.item(),
            "task": task_loss.item(),
            "guess_penalty": guess_penalty.item(),
            "storage_cost": storage_cost.item(),
            "entropy": entropy.item(),
            "aux": aux_loss.item(),
            "commit_reg": commit_reg.item(),
        }
        return total_loss, loss_dict
