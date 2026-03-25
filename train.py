"""Training script for the Fusion Model on the ARC-AGI-2 dataset (JAX/Flax).

Usage::

    python train.py --data_root data --epochs 40
    python train.py --parquet_dir /path/to/parquets --epochs 40
    python train.py --parquet_files a.parquet b.parquet --epochs 40

The script:

1. Constructs data loaders for the training and evaluation splits of ARC-AGI-2.
2. Instantiates the :class:`~fusion_model.FusionModel` with Flax.
3. Trains with AdamW + linear warmup + cosine-annealing LR via Optax,
   printing metrics every epoch.
4. Saves the best model weights (by validation accuracy) and a JSON
   training history for later analysis.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training import train_state

from fusion_model import FusionModel
from fusion_model.loss import fusion_loss
from tasks.arc import (
    NUM_COLOURS,
    PAD_VALUE,
    ARCDataset,
    ParquetARCDataset,
    collate_batch,
    data_loader,
)

# ── Helpers ──────────────────────────────────────────────────────────────────


def count_params(params: Any) -> int:
    """Return the number of trainable parameters."""
    return sum(x.size for x in jax.tree.leaves(params))


def compute_cell_accuracy(
    logits: jnp.ndarray,
    targets: jnp.ndarray,
    pad_value: int = PAD_VALUE,
) -> tuple[int, int]:
    """Compute per-cell accuracy ignoring padded positions."""
    preds = logits.argmax(axis=-1)
    valid = targets != pad_value
    correct = int(((preds == targets) & valid).sum())
    total = int(valid.sum())
    return correct, total


# ── Train state ──────────────────────────────────────────────────────────────


class FusionTrainState(train_state.TrainState):
    """Extended train state that carries mutable model state (buffers)."""
    model_state: Any = None
    rng: jax.Array = None


# ── Training step ────────────────────────────────────────────────────────────


def train_step(
    state: FusionTrainState,
    batch: dict[str, jnp.ndarray],
    pad_value: int = PAD_VALUE,
) -> tuple[FusionTrainState, dict[str, Any]]:
    """Execute one training step.

    :param state: Current training state (params, optimizer, model_state).
    :param batch: Dict of batched arrays from the data loader.
    :param pad_value: Padding value for targets.
    :return: Updated state and metrics dict.
    """
    rng, dropout_rng = jax.random.split(state.rng)

    def loss_fn(params):
        variables = {"params": params}
        if state.model_state is not None:
            variables["state"] = state.model_state

        (logits, alphas, meta), mutated = state.apply_fn(
            variables,
            batch["demo_inputs"],
            batch["demo_outputs"],
            batch["demo_mask"],
            batch["test_input"],
            training=True,
            rngs={"dropout": dropout_rng},
            mutable=["state"],
        )

        B = batch["test_output"].shape[0]
        targets_flat = batch["test_output"].reshape(B, -1)
        max_cells = logits.shape[1]
        targets_flat = targets_flat[:, :max_cells]

        total_loss, loss_dict = fusion_loss(
            logits, targets_flat, alphas,
            meta["retrieval_scores"],
            metadata=meta,
            pad_value=pad_value,
        )

        return total_loss, (loss_dict, logits, alphas, targets_flat, mutated)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (loss_dict, logits, alphas, targets_flat, mutated)), grads = grad_fn(state.params)

    # Clip gradients.
    grads = optax.clip_by_global_norm(1.0).update(grads, state.opt_state)[0]

    state = state.apply_gradients(grads=grads)
    state = state.replace(
        model_state=mutated.get("state"),
        rng=rng,
    )

    # Compute accuracy.
    preds = logits.argmax(axis=-1)
    valid = targets_flat != pad_value
    correct = ((preds == targets_flat) & valid).sum()
    total = valid.sum()

    metrics = {
        "loss": loss,
        "correct": correct,
        "total": total,
        "alphas": alphas.mean(axis=0),
        **{k: v for k, v in loss_dict.items()},
    }
    return state, metrics


# ── Evaluation step ──────────────────────────────────────────────────────────


def eval_step(
    state: FusionTrainState,
    batch: dict[str, jnp.ndarray],
    pad_value: int = PAD_VALUE,
) -> tuple[int, int]:
    """Execute one evaluation step (no gradients)."""
    variables = {"params": state.params}
    if state.model_state is not None:
        variables["state"] = state.model_state

    logits, _, _ = state.apply_fn(
        variables,
        batch["demo_inputs"],
        batch["demo_outputs"],
        batch["demo_mask"],
        batch["test_input"],
        training=False,
    )

    B = batch["test_output"].shape[0]
    targets_flat = batch["test_output"].reshape(B, -1)
    max_cells = logits.shape[1]
    targets_flat = targets_flat[:, :max_cells]

    return compute_cell_accuracy(logits, targets_flat, pad_value)


def evaluate(state: FusionTrainState, dataset: Any, batch_size: int) -> float:
    """Compute per-cell accuracy on a dataset."""
    correct = 0
    total = 0
    for batch in data_loader(dataset, batch_size=batch_size, shuffle=False):
        batch_jax = {k: jnp.array(v) for k, v in batch.items()}
        c, t = eval_step(state, batch_jax)
        correct += c
        total += t
    return correct / total if total > 0 else 0.0


# ── Training loop ────────────────────────────────────────────────────────────


def train_fusion(
    state: FusionTrainState,
    train_ds: Any,
    val_ds: Any,
    args: argparse.Namespace,
) -> tuple[FusionTrainState, dict[str, list[float]]]:
    """Run the full training loop for the Fusion Model."""
    best_val_acc = 0.0
    patience_counter = 0
    history: dict[str, list[float]] = {"train_loss": [], "train_acc": [], "val_acc": []}
    rng = np.random.default_rng(args.seed)

    accum_steps = args.grad_accum // args.batch_size

    print(
        f"Effective batch size: {args.grad_accum} "
        f"(micro={args.batch_size} x accum={accum_steps})"
    )

    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        step = 0

        for batch in data_loader(train_ds, batch_size=args.batch_size, shuffle=True, rng=rng):
            batch_jax = {k: jnp.array(v) for k, v in batch.items()}
            state, metrics = train_step(state, batch_jax)

            B = batch_jax["test_output"].shape[0]
            epoch_loss += float(metrics["loss"]) * B
            epoch_correct += int(metrics["correct"])
            epoch_total += int(metrics["total"])
            step += 1

        train_loss = epoch_loss / max(epoch_total, 1)
        train_acc = epoch_correct / max(epoch_total, 1)
        val_acc = evaluate(state, val_ds, args.batch_size)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        alpha_mean = metrics["alphas"]
        print(
            f"Epoch {epoch:3d}/{args.epochs}  "
            f"loss={train_loss:.4f}  train_acc={train_acc:.3f}  "
            f"val_acc={val_acc:.3f}  "
            f"alpha(mem={float(alpha_mean[0]):.3f}  "
            f"rule={float(alpha_mean[1]):.3f}  "
            f"guess={float(alpha_mean[2]):.3f})"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            # Save checkpoint.
            import pickle
            ckpt = {
                "params": jax.device_get(state.params),
                "model_state": jax.device_get(state.model_state),
            }
            with open(os.path.join(args.out_dir, "fusion_best.pkl"), "wb") as f:
                pickle.dump(ckpt, f)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch} (patience={args.patience})")
                break

    return state, history


# ── Entry point ──────────────────────────────────────────────────────────────


def main() -> None:
    """Parse arguments, build datasets, and launch training."""
    parser = argparse.ArgumentParser(description="Train Fusion Model on ARC-AGI-2 (JAX)")
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--parquet_dir", type=str, default=None)
    parser.add_argument("--parquet_files", type=str, nargs="+", default=None)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument(
        "--grad_accum", type=int, default=None,
        help="Effective batch size for gradient accumulation.",
    )
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--out_dir", type=str, default="outputs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_grid_size", type=int, default=30)
    parser.add_argument("--max_demos", type=int, default=5)
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--num_encoder_layers", type=int, default=4)
    parser.add_argument("--num_cross_attn_layers", type=int, default=4)
    args = parser.parse_args()

    # ── Validate gradient accumulation ────────────────────────────────────
    if args.grad_accum is None:
        args.grad_accum = args.batch_size
    if args.grad_accum < args.batch_size:
        print(f"Error: --grad_accum ({args.grad_accum}) must be >= --batch_size ({args.batch_size}).", file=sys.stderr)
        sys.exit(1)
    if args.grad_accum % args.batch_size != 0:
        print(f"Error: --grad_accum ({args.grad_accum}) must be divisible by --batch_size ({args.batch_size}).", file=sys.stderr)
        sys.exit(1)

    # ── Validate parquet args ─────────────────────────────────────────────
    if args.parquet_dir is not None and args.parquet_files is not None:
        print("Error: --parquet_dir and --parquet_files are mutually exclusive.", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"JAX devices: {jax.devices()}")
    print(f"JAX backend: {jax.default_backend()}")

    # ── Construct datasets ───────────────────────────────────────────────
    parquet_paths: list[str] | None = None
    if args.parquet_dir is not None:
        parquet_paths = sorted(
            os.path.join(args.parquet_dir, f)
            for f in os.listdir(args.parquet_dir)
            if f.endswith(".parquet")
        )
        if not parquet_paths:
            print(f"Error: no .parquet files found in {args.parquet_dir}.", file=sys.stderr)
            sys.exit(1)
    elif args.parquet_files is not None:
        parquet_paths = args.parquet_files

    if parquet_paths is not None:
        train_ds = ParquetARCDataset(
            parquet_paths,
            max_grid_size=args.max_grid_size,
            max_demos=args.max_demos,
            max_samples=args.max_samples,
        )
    else:
        train_dir = os.path.join(args.data_root, "training")
        train_ds = ARCDataset(
            train_dir,
            max_grid_size=args.max_grid_size,
            max_demos=args.max_demos,
            max_samples=args.max_samples,
        )

    eval_dir = os.path.join(args.data_root, "evaluation")
    val_ds = ARCDataset(
        eval_dir,
        max_grid_size=args.max_grid_size,
        max_demos=args.max_demos,
        max_samples=args.max_samples,
    )

    print(f"Train samples: {len(train_ds):,}  |  Val samples: {len(val_ds):,}")

    # ── Instantiate the Fusion Model ─────────────────────────────────────
    model = FusionModel(
        embed_dim=args.embed_dim,
        num_colours=NUM_COLOURS,
        max_grid_size=args.max_grid_size,
        num_encoder_layers=args.num_encoder_layers,
        num_cross_attn_layers=args.num_cross_attn_layers,
    )

    # Initialise parameters with a dummy batch.
    rng = jax.random.PRNGKey(args.seed)
    rng, init_rng, dropout_rng = jax.random.split(rng, 3)

    G = args.max_grid_size
    D = args.max_demos if hasattr(args, "max_demos") else 5
    dummy_batch = {
        "demo_inputs": jnp.zeros((1, D, G, G), dtype=jnp.int32),
        "demo_outputs": jnp.zeros((1, D, G, G), dtype=jnp.int32),
        "demo_mask": jnp.ones((1, D), dtype=jnp.bool_),
        "test_input": jnp.zeros((1, G, G), dtype=jnp.int32),
    }

    variables = model.init(
        {"params": init_rng, "dropout": dropout_rng},
        dummy_batch["demo_inputs"],
        dummy_batch["demo_outputs"],
        dummy_batch["demo_mask"],
        dummy_batch["test_input"],
        training=True,
    )
    params = variables["params"]
    model_state = variables.get("state")

    print(f"Fusion Model: {count_params(params):,} trainable parameters")

    # ── Optimizer: AdamW + linear warmup + cosine decay ──────────────────
    warmup_epochs = min(5, args.epochs // 4)
    total_steps = args.epochs * (len(train_ds) // args.batch_size + 1)
    warmup_steps = warmup_epochs * (len(train_ds) // args.batch_size + 1)

    schedule = optax.join_schedules(
        schedules=[
            optax.linear_schedule(init_value=args.lr * 0.01, end_value=args.lr, transition_steps=max(1, warmup_steps)),
            optax.cosine_decay_schedule(init_value=args.lr, decay_steps=max(1, total_steps - warmup_steps)),
        ],
        boundaries=[warmup_steps],
    )

    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=schedule, weight_decay=args.weight_decay),
    )

    state = FusionTrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer,
        model_state=model_state,
        rng=rng,
    )

    print("Training...")
    state, history = train_fusion(state, train_ds, val_ds, args)

    history_path = os.path.join(args.out_dir, "fusion_history.json")
    with open(history_path, "w", encoding="utf-8") as fh:
        json.dump(history, fh)
    print(f"History saved to {history_path}")


if __name__ == "__main__":
    main()
