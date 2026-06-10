"""FusionLoss — combined training objective for the Fusion Model.

The overall loss is a weighted sum of several terms, each addressing a
different failure mode of the mixture-of-experts architecture:

1. **Task loss** — standard per-cell cross-entropy on the combined output.

2. **Guess penalty** — ``mean(alpha_guess)`` pushes the router away from
   relying exclusively on the guess pathway.

3. **Storage cost** — an approximate L0 over rule-memory slot utilisation.

4. **Negative entropy** — ``-entropy(alpha)`` encourages the router to
   *explore* all three pathways early in training.

5. **Auxiliary losses** — independent per-cell cross-entropy on each
   pathway's own logits, ensuring every pathway receives gradient signal.

6. **Commitment regularisation** — penalises the deviation of the mean
   soft commit weight from a target rate.

7. **Strength regularisation** — penalises the deviation of the mean
   memory-slot strength from a target occupancy.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


class FusionLoss(nn.Module):
    """Compute the multi-component Fusion Model loss.

    Now accepts **per-cell** logits ``(B, seq, C)`` and targets ``(B, seq)``
    with a ``pad_value`` for masking, instead of the previous flattened
    format.

    :param pad_value: Target value to ignore in cross-entropy (default -1).
    :param lambda_guess: Weight for the guess-pathway penalty.
    :param lambda_storage: Weight for the rule-storage cost.
    :param lambda_entropy: Weight for the (negative) routing-entropy bonus.
    :param lambda_aux: Weight for the auxiliary per-pathway losses.
    :param lambda_commit: Weight for commitment-rate regularisation.
    :param commit_target_rate: Desired mean commit weight.
    :param lambda_strength: Weight for strength occupancy regulariser.
    :param strength_target: Desired mean slot strength.
    """

    def __init__(
        self,
        pad_value: int = -1,
        lambda_guess: float = 0.01,
        lambda_storage: float = 0.001,
        lambda_entropy: float = 0.05,
        lambda_aux: float = 0.3,
        lambda_commit: float = 0.001,
        commit_target_rate: float = 0.1,
        lambda_strength: float = 0.001,
        strength_target: float = 0.5,
    ) -> None:
        super().__init__()
        self.pad_value = pad_value
        self.lambda_guess = lambda_guess
        self.lambda_storage = lambda_storage
        self.lambda_entropy = lambda_entropy
        self.lambda_aux = lambda_aux
        self.lambda_commit = lambda_commit
        self.commit_target_rate = commit_target_rate
        self.lambda_strength = lambda_strength
        self.strength_target = strength_target
        self.ce = nn.CrossEntropyLoss(ignore_index=pad_value)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        alphas: torch.Tensor,
        retrieval_scores: torch.Tensor,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute the total loss and a breakdown dict.

        :param logits: Per-cell logits ``(batch, seq, num_colours)``.
        :param targets: Ground-truth cell values ``(batch, seq)`` with
            ``pad_value`` for padding.
        :param alphas: Routing weights ``(batch, 3)``.
        :param retrieval_scores: Memory retrieval scores ``(batch, num_slots)``.
        :param metadata: Dict with per-pathway logits and other info.
        :return: Tuple of ``(total_loss, loss_dict)``.
        """
        B, S, C = logits.shape

        # -- 1. Primary task loss (per-cell cross-entropy). --
        task_loss = self.ce(logits.reshape(-1, C), targets.reshape(-1))

        # -- 2. Penalise over-reliance on the guess pathway. --
        guess_penalty = alphas[:, 2].mean()

        # -- 3. Encourage sparse rule-slot usage (approximate L0). --
        slot_usage = retrieval_scores.mean(dim=0)
        storage_cost = (slot_usage * (1.0 - slot_usage)).sum()

        # -- 4. Negative entropy: reward uniform alpha early. --
        eps = 1e-8
        entropy = -(alphas * (alphas + eps).log()).sum(dim=-1).mean()

        # -- 5. Auxiliary losses: per-cell CE for each pathway. --
        aux_loss = torch.tensor(0.0, device=logits.device)
        aux_mem = torch.tensor(0.0, device=logits.device)
        aux_rule = torch.tensor(0.0, device=logits.device)
        aux_guess = torch.tensor(0.0, device=logits.device)
        if metadata is not None:
            targets_flat = targets.reshape(-1)
            aux_parts: dict[str, torch.Tensor] = {}
            for key in ("logits_mem", "logits_rule", "logits_guess"):
                if key in metadata:
                    pathway_logits = metadata[key]  # (B, seq, num_colours)
                    aux_parts[key] = self.ce(
                        pathway_logits.reshape(-1, C), targets_flat,
                    )
            if aux_parts:
                aux_mem = aux_parts.get("logits_mem", aux_mem)
                aux_rule = aux_parts.get("logits_rule", aux_rule)
                aux_guess = aux_parts.get("logits_guess", aux_guess)
                aux_loss = sum(aux_parts.values()) / len(aux_parts)

        # -- 6. Commitment rate regularisation. --
        commit_reg = torch.tensor(0.0, device=logits.device)
        if metadata is not None:
            proposal = metadata.get("proposal")
            if proposal is not None:
                commit_weight = proposal["commit_weight"]
                mean_weight = commit_weight.squeeze(-1).mean()
                commit_reg = (mean_weight - self.commit_target_rate) ** 2

        # -- 7. Memory-strength occupancy regularisation. --
        strength_reg = torch.tensor(0.0, device=logits.device)
        if metadata is not None:
            strength = metadata.get("memory_strength")
            if strength is not None:
                mean_strength = strength.mean()
                strength_reg = (mean_strength - self.strength_target) ** 2

        # -- Combine everything. --
        total_loss: torch.Tensor = (
            task_loss
            + self.lambda_guess * guess_penalty
            + self.lambda_storage * storage_cost
            - self.lambda_entropy * entropy
            + self.lambda_aux * aux_loss
            + self.lambda_commit * commit_reg
            + self.lambda_strength * strength_reg
        )

        loss_dict: dict[str, float] = {
            "total": total_loss.item(),
            "task": task_loss.item(),
            "guess_penalty": guess_penalty.item(),
            "storage_cost": storage_cost.item(),
            "entropy": entropy.item(),
            "aux": aux_loss.item(),
            "aux_mem": aux_mem.item(),
            "aux_rule": aux_rule.item(),
            "aux_guess": aux_guess.item(),
            "commit_reg": commit_reg.item(),
            "strength_reg": strength_reg.item(),
        }
        return total_loss, loss_dict
