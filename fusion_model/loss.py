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

import jax
import jax.numpy as jnp
import optax


def cross_entropy_with_ignore(
    logits: jnp.ndarray, targets: jnp.ndarray, ignore_index: int = -1,
) -> jnp.ndarray:
    """Compute mean cross-entropy ignoring positions where target == ignore_index."""
    mask = targets != ignore_index
    safe_targets = jnp.where(mask, targets, 0)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    nll = -jnp.take_along_axis(log_probs, safe_targets[..., None], axis=-1).squeeze(-1)
    nll = jnp.where(mask, nll, 0.0)
    return nll.sum() / jnp.maximum(mask.sum(), 1)


def fusion_loss(
    logits: jnp.ndarray,
    targets: jnp.ndarray,
    alphas: jnp.ndarray,
    retrieval_scores: jnp.ndarray,
    metadata: dict[str, Any] | None = None,
    *,
    pad_value: int = -1,
    lambda_guess: float = 0.01,
    lambda_storage: float = 0.001,
    lambda_entropy: float = 0.05,
    lambda_aux: float = 0.3,
    lambda_commit: float = 0.001,
    commit_target_rate: float = 0.1,
    lambda_strength: float = 0.001,
    strength_target: float = 0.5,
) -> tuple[jnp.ndarray, dict[str, float]]:
    """Compute the multi-component Fusion Model loss.

    :param logits: Per-cell logits ``(batch, seq, num_colours)``.
    :param targets: Ground-truth cell values ``(batch, seq)`` with
        ``pad_value`` for padding.
    :param alphas: Routing weights ``(batch, 3)``.
    :param retrieval_scores: Memory retrieval scores ``(batch, num_slots)``.
    :param metadata: Dict with per-pathway logits and other info.
    :return: Tuple of ``(total_loss, loss_dict)`` where loss_dict values are
        raw jnp scalars (JIT-compatible). Convert to float outside JIT.
    """
    B, S, C = logits.shape

    # -- 1. Primary task loss (per-cell cross-entropy). --
    task_loss = cross_entropy_with_ignore(
        logits.reshape(-1, C), targets.reshape(-1), ignore_index=pad_value,
    )

    # -- 2. Penalise over-reliance on the guess pathway. --
    guess_penalty = alphas[:, 2].mean()

    # -- 3. Encourage sparse rule-slot usage (approximate L0). --
    slot_usage = retrieval_scores.mean(axis=0)
    storage_cost = (slot_usage * (1.0 - slot_usage)).sum()

    # -- 4. Negative entropy: reward uniform alpha early. --
    eps = 1e-8
    entropy = -(alphas * jnp.log(alphas + eps)).sum(axis=-1).mean()

    # -- 5. Auxiliary losses: per-cell CE for each pathway. --
    aux_loss = jnp.array(0.0)
    if metadata is not None:
        targets_flat = targets.reshape(-1)
        for key in ("logits_mem", "logits_rule", "logits_guess"):
            if key in metadata:
                pathway_logits = metadata[key]  # (B, seq, num_colours)
                aux_loss = aux_loss + cross_entropy_with_ignore(
                    pathway_logits.reshape(-1, C), targets_flat, ignore_index=pad_value,
                )
        aux_loss = aux_loss / 3.0

    # -- 6. Commitment rate regularisation. --
    commit_reg = jnp.array(0.0)
    if metadata is not None:
        proposal = metadata.get("proposal")
        if proposal is not None:
            commit_weight = proposal["commit_weight"]
            mean_weight = commit_weight.squeeze(-1).mean()
            commit_reg = (mean_weight - commit_target_rate) ** 2

    # -- 7. Memory-strength occupancy regularisation. --
    strength_reg = jnp.array(0.0)
    if metadata is not None:
        strength = metadata.get("memory_strength")
        if strength is not None:
            mean_strength = strength.mean()
            strength_reg = (mean_strength - strength_target) ** 2

    # -- Combine everything. --
    total_loss = (
        task_loss
        + lambda_guess * guess_penalty
        + lambda_storage * storage_cost
        - lambda_entropy * entropy
        + lambda_aux * aux_loss
        + lambda_commit * commit_reg
        + lambda_strength * strength_reg
    )

    loss_dict = {
        "total": total_loss,
        "task": task_loss,
        "guess_penalty": guess_penalty,
        "storage_cost": storage_cost,
        "entropy": entropy,
        "aux": aux_loss,
        "commit_reg": commit_reg,
        "strength_reg": strength_reg,
    }
    return total_loss, loss_dict
