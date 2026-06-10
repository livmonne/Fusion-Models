"""Shared helpers used by several Fusion Model components.

Small, dependency-free utilities live here so that the expert pathways
(:mod:`~fusion_model.memory`, :mod:`~fusion_model.rule_engine`,
:mod:`~fusion_model.guess`) and the orchestrator
(:mod:`~fusion_model.model`) can share masking-aware pooling and
per-sample loss computation without circular imports.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_mean(x: torch.Tensor, pad_mask: torch.Tensor | None) -> torch.Tensor:
    """Mean-pool a token sequence, ignoring padding positions.

    :param x: Token features ``(batch, seq, embed_dim)``.
    :param pad_mask: Boolean validity mask ``(batch, seq)`` where *True*
        marks a **real** (non-padding) token.  ``None`` falls back to a
        plain mean over the sequence dimension.
    :return: Pooled features ``(batch, embed_dim)``.
    """
    if pad_mask is None:
        return x.mean(dim=1)
    mask = pad_mask.float()
    denom = mask.sum(dim=-1, keepdim=True).clamp(min=1.0)
    return (x * mask.unsqueeze(-1)).sum(dim=1) / denom


def per_sample_ce(
    logits: torch.Tensor, targets: torch.Tensor, pad_value: int = -1
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-sample mean cross-entropy over valid (non-padding) cells.

    :param logits: Per-cell logits ``(batch, seq, num_classes)``.
    :param targets: Ground-truth classes ``(batch, seq)`` with
        ``pad_value`` marking cells to ignore.
    :return: Tuple ``(loss, valid)`` where *loss* is ``(batch,)`` (zero
        for samples without any valid cell) and *valid* is a ``(batch,)``
        boolean mask of samples that had at least one valid cell.
    """
    B, _, C = logits.shape
    ce = F.cross_entropy(
        logits.reshape(-1, C),
        targets.reshape(-1),
        ignore_index=pad_value,
        reduction="none",
    ).view(B, -1)
    valid_cells = (targets != pad_value).float()
    counts = valid_cells.sum(dim=-1)
    loss = (ce * valid_cells).sum(dim=-1) / counts.clamp(min=1.0)
    return loss, counts > 0
