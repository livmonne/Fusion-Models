"""FusionLoss — combined training objective for the Fusion Model.

The overall loss is a weighted sum of several terms, each addressing a
different failure mode of the mixture-of-experts architecture:

1. **Task loss** — standard per-cell cross-entropy on the blended output.

2. **Size loss** — cross-entropy on the predicted output grid height and
   width.  Without it the model can only handle tasks whose output has
   the same shape as the input.

3. **Verification loss** — per-pathway cross-entropy on the held-out
   demonstration (leave-one-out).  This is real training signal: every
   pathway must be able to reproduce demonstrations it did not see, which
   is exactly what "knowing the rule" means.

4. **Guess penalty** — ``mean(alpha_guess)`` pushes the router away from
   relying exclusively on the guess pathway.

5. **Storage cost** — mean entropy of the memory retrieval distribution.
   Low entropy = peaky retrieval = specialised slots.  (An earlier
   ``u·(1−u)`` formulation was minimised by *binarising* slot usage,
   which encouraged winner-take-all collapse rather than sparsity.)

6. **Negative routing entropy** — ``-entropy(alpha)`` keeps all pathways
   alive early in training.

7. **Auxiliary losses** — independent per-cell cross-entropy on each
   pathway's own logits (including the proposal pathway when active),
   ensuring every pathway receives gradient signal even when the router
   ignores it.

8. **Commitment regularisation** — penalises the deviation of the mean
   soft commit weight from a target rate.

9. **Strength regularisation** — penalises the deviation of the mean
   memory-slot strength from a target occupancy.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from .decision import GUESS_INDEX, NUM_PATHWAYS, PATHWAY_NAMES, PROP_INDEX


class FusionLoss(nn.Module):
    """Compute the multi-component Fusion Model loss.

    Accepts **per-cell** logits ``(B, seq, C)`` and targets ``(B, seq)``
    with a ``pad_value`` for masking.

    :param pad_value: Target value to ignore in cross-entropy (default -1).
    :param lambda_size: Weight for the output-size prediction loss.
    :param lambda_verify: Weight for the leave-one-out verification loss.
    :param lambda_guess: Weight for the guess-pathway penalty.
    :param lambda_storage: Weight for the retrieval-entropy storage cost.
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
        lambda_size: float = 0.1,
        lambda_verify: float = 0.3,
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
        self.lambda_size = lambda_size
        self.lambda_verify = lambda_verify
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
        size_targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute the total loss and a breakdown dict.

        :param logits: Per-cell logits ``(batch, seq, num_colours)``.
        :param targets: Ground-truth cell values ``(batch, seq)`` with
            ``pad_value`` for padding.
        :param alphas: Routing weights ``(batch, NUM_PATHWAYS)``.
        :param retrieval_scores: Memory retrieval scores ``(batch, num_slots)``.
        :param metadata: Dict with per-pathway logits and other info
            (see ``FusionModel.forward``).
        :param size_targets: Ground-truth output sizes ``(batch, 2)`` as
            ``[height, width]`` in cells; entries ``<= 0`` are ignored.
        :return: Tuple of ``(total_loss, loss_dict)``.
        """
        B, S, C = logits.shape
        device = logits.device

        # -- 1. Primary task loss (per-cell cross-entropy). --
        task_loss = self.ce(logits.reshape(-1, C), targets.reshape(-1))

        # -- 2. Output size loss. --
        size_loss = torch.tensor(0.0, device=device)
        if metadata is not None and size_targets is not None:
            size_logits = metadata.get("size_logits")
            if size_logits is not None:
                # Class k corresponds to size k+1; sizes <= 0 are unknown.
                size_cls = size_targets.long() - 1
                size_cls[size_targets <= 0] = self.pad_value
                size_loss = self.ce(
                    size_logits.reshape(-1, size_logits.shape[-1]),
                    size_cls.reshape(-1),
                )

        # -- 3. Leave-one-out verification loss. --
        verify_loss = torch.tensor(0.0, device=device)
        if metadata is not None:
            verify_ce = metadata.get("verify_ce")
            verify_valid = metadata.get("verify_valid")
            if verify_ce is not None and verify_valid is not None and bool(
                verify_valid.any()
            ):
                cols = list(range(NUM_PATHWAYS))
                if not metadata.get("prop_active", False):
                    cols.remove(PROP_INDEX)
                verify_loss = verify_ce[verify_valid][:, cols].mean()

        # -- 4. Penalise over-reliance on the guess pathway. --
        guess_penalty = alphas[:, GUESS_INDEX].mean()

        # -- 5. Encourage peaky (specialised) rule retrieval. --
        eps = 1e-8
        storage_cost = (
            -(retrieval_scores * (retrieval_scores + eps).log()).sum(dim=-1).mean()
        )

        # -- 6. Negative routing entropy: reward exploration early. --
        entropy = -(alphas * (alphas + eps).log()).sum(dim=-1).mean()

        # -- 7. Auxiliary losses: per-cell CE for each pathway. --
        aux_loss = torch.tensor(0.0, device=device)
        aux_parts: dict[str, torch.Tensor] = {}
        if metadata is not None:
            targets_flat = targets.reshape(-1)
            for name in PATHWAY_NAMES:
                pathway_logits = metadata.get(f"logits_{name}")
                if pathway_logits is None:
                    continue
                aux_parts[name] = self.ce(
                    pathway_logits.reshape(-1, C), targets_flat,
                )
            if aux_parts:
                aux_loss = torch.stack(list(aux_parts.values())).mean()

        # -- 8. Commitment rate regularisation. --
        commit_reg = torch.tensor(0.0, device=device)
        if metadata is not None:
            proposal = metadata.get("proposal")
            if proposal is not None:
                commit_weight = proposal["commit_weight"]
                mean_weight = commit_weight.squeeze(-1).mean()
                commit_reg = (mean_weight - self.commit_target_rate) ** 2

        # -- 9. Memory-strength occupancy regularisation. --
        strength_reg = torch.tensor(0.0, device=device)
        if metadata is not None:
            strength = metadata.get("memory_strength")
            if strength is not None:
                mean_strength = strength.mean()
                strength_reg = (mean_strength - self.strength_target) ** 2

        # -- Combine everything. --
        total_loss: torch.Tensor = (
            task_loss
            + self.lambda_size * size_loss
            + self.lambda_verify * verify_loss
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
            "size": size_loss.item(),
            "verify": verify_loss.item(),
            "guess_penalty": guess_penalty.item(),
            "storage_cost": storage_cost.item(),
            "entropy": entropy.item(),
            "aux": aux_loss.item(),
            "commit_reg": commit_reg.item(),
            "strength_reg": strength_reg.item(),
        }
        for name in PATHWAY_NAMES:
            part = aux_parts.get(name)
            loss_dict[f"aux_{name}"] = part.item() if part is not None else 0.0
        return total_loss, loss_dict
