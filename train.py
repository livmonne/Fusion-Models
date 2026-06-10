"""Training script for the Fusion Model on ARC-style datasets.

Usage::

    python train.py --data_root data --epochs 40
    python train.py --parquet_dir /path/to/parquets --epochs 40
    python train.py --parquet_files a.parquet b.parquet --epochs 40

The script:

1. Constructs PyTorch ``DataLoader`` instances for the training and
   evaluation splits.
2. Instantiates the :class:`~fusion_model.FusionModel` and
   :class:`~fusion_model.FusionLoss`.
3. Trains with AdamW + **per-step** linear warmup + cosine-annealing LR
   (per-epoch scheduling is useless on million-sample datasets where a
   single epoch is hundreds of thousands of steps).
4. After every backward pass, feeds the measured per-sample losses back
   into the model (``model.apply_outcomes``): this updates the rule
   proposer's history buffer and commits proposed rules to the memory
   bank when they beat the running average — commits only happen on
   optimiser-step boundaries so in-place slot updates never race a
   pending ``optimizer.step()``.
5. Tracks **exact-match solve rate** (cells *and* predicted size correct)
   alongside per-cell accuracy, and keeps the checkpoint with the best
   ``(solve_rate, cell_acc)``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from contextlib import nullcontext
from typing import Any

import torch
from torch.utils.data import DataLoader

from fusion_model import PATHWAY_NAMES, FusionModel
from fusion_model.common import per_sample_ce
from fusion_model.loss import FusionLoss
from tasks.arc import NUM_COLOURS, PAD_VALUE, ARCDataset, ParquetARCDataset, arc_collate_fn

# ── Helpers ──────────────────────────────────────────────────────────────────


def count_params(model: torch.nn.Module) -> int:
    """Return the number of trainable parameters in *model*."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def compute_cell_accuracy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pad_value: int = PAD_VALUE,
) -> tuple[int, int]:
    """Compute per-cell accuracy ignoring padded positions.

    :param logits: ``(batch, max_cells, num_colours)`` predicted logits.
    :param targets: ``(batch, max_cells)`` ground-truth cell values.
    :param pad_value: Value used for padding (ignored in accuracy).
    :return: ``(correct, total)`` counts.
    """
    preds = logits.argmax(dim=-1)  # (B, max_cells)
    valid = targets != pad_value
    correct = ((preds == targets) & valid).sum().item()
    total = valid.sum().item()
    return int(correct), int(total)


def flatten_targets(
    test_output: torch.Tensor, max_cells: int
) -> torch.Tensor:
    """Flatten ``(B, H, W)`` targets to ``(B, max_cells)`` matching the logits."""
    targets = test_output.view(test_output.size(0), -1)
    return targets[:, :max_cells]


# ── Evaluation ───────────────────────────────────────────────────────────────


@torch.no_grad()
def evaluate(
    model: FusionModel,
    loader: DataLoader,  # type: ignore[type-arg]
    device: torch.device,
) -> dict[str, float]:
    """Evaluate cell accuracy, exact-match solve rate, and size accuracy.

    A sample counts as **solved** only when the predicted output size
    matches the ground truth *and* every non-padding cell is correct —
    the honest ARC metric.  Per-cell accuracy is reported as a smoother
    diagnostic signal (it is very flattering on its own: 90% cell
    accuracy can still be a 0% solve rate).

    :param model: The Fusion Model (switched to eval mode here).
    :param loader: DataLoader yielding sample dicts.
    :param device: Torch device.
    :return: Dict with ``cell_acc``, ``solve_rate``, and ``size_acc``.
    """
    model.eval()
    correct = 0
    total = 0
    solved = 0
    size_correct = 0
    n_samples = 0

    for batch in loader:
        demo_inputs = batch["demo_inputs"].to(device)
        demo_outputs = batch["demo_outputs"].to(device)
        demo_mask = batch["demo_mask"].to(device)
        test_input = batch["test_input"].to(device)
        test_output = batch["test_output"].to(device)
        output_size = batch["output_size"].to(device)

        logits, _, meta = model(demo_inputs, demo_outputs, demo_mask, test_input)
        targets = flatten_targets(test_output, logits.size(1))

        c, t = compute_cell_accuracy(logits, targets)
        correct += c
        total += t

        preds = logits.argmax(dim=-1)
        valid = targets != PAD_VALUE
        cells_ok = ((preds == targets) | ~valid).all(dim=1) & valid.any(dim=1)

        size_pred = meta["size_logits"].argmax(dim=-1) + 1  # (B, 2)
        has_size = (output_size > 0).all(dim=1)
        size_ok = (size_pred == output_size).all(dim=1) & has_size

        solved += int((cells_ok & size_ok).sum().item())
        size_correct += int(size_ok.sum().item())
        n_samples += int(has_size.sum().item())

    return {
        "cell_acc": correct / total if total > 0 else 0.0,
        "solve_rate": solved / n_samples if n_samples > 0 else 0.0,
        "size_acc": size_correct / n_samples if n_samples > 0 else 0.0,
    }


# ── Training loop ────────────────────────────────────────────────────────────


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_steps: int,
    min_factor: float = 0.05,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Per-step linear warmup followed by cosine annealing to ``min_factor``."""

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(1.0, progress)
        return max(min_factor, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_fusion(
    model: FusionModel,
    criterion: FusionLoss,
    train_loader: DataLoader,  # type: ignore[type-arg]
    val_loader: DataLoader,  # type: ignore[type-arg]
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, list[float]]:
    """Run the full training loop for the Fusion Model.

    :param model: The Fusion Model to train.
    :param criterion: The :class:`~fusion_model.FusionLoss` instance.
    :param train_loader: Training data loader.
    :param val_loader: Validation data loader.
    :param device: Torch device.
    :param args: Parsed command-line arguments.
    :return: Dictionary with per-epoch metric lists.
    """
    print("Configuring optimizer & LR schedule...")
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    accum_steps = args.grad_accum // args.batch_size
    steps_per_epoch = max(1, math.ceil(len(train_loader) / accum_steps))
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = (
        args.warmup_steps
        if args.warmup_steps is not None
        else min(2000, max(10, int(0.03 * total_steps)))
    )
    scheduler = build_scheduler(optimizer, total_steps, warmup_steps)

    use_amp = args.amp and device.type == "cuda"

    best_metric = (-1.0, -1.0)  # (solve_rate, cell_acc)
    patience_counter = 0
    history: dict[str, list[float]] = {
        "train_loss": [], "train_acc": [],
        "val_acc": [], "val_solve": [], "val_size_acc": [],
    }

    print(
        f"Optimiser steps: {total_steps} total, {warmup_steps} warmup | "
        f"Effective batch size: {args.grad_accum} "
        f"(micro={args.batch_size} x accum={accum_steps}) | "
        f"AMP: {use_amp}"
    )

    global_step = 0
    stop = False
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_samples = 0
        epoch_correct = 0
        epoch_total = 0
        epoch_commits = 0
        alpha_sum = torch.zeros(len(PATHWAY_NAMES))
        alpha_batches = 0
        last_loss_dict: dict[str, float] = {}

        optimizer.zero_grad()

        step = -1
        for step, batch in enumerate(train_loader):
            demo_inputs = batch["demo_inputs"].to(device)
            demo_outputs = batch["demo_outputs"].to(device)
            demo_mask = batch["demo_mask"].to(device)
            test_input = batch["test_input"].to(device)
            test_output = batch["test_output"].to(device)
            output_size = batch["output_size"].to(device)

            amp_ctx = (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if use_amp
                else nullcontext()
            )
            with amp_ctx:
                logits, alphas, meta = model(
                    demo_inputs, demo_outputs, demo_mask, test_input,
                )
                targets = flatten_targets(test_output, logits.size(1))

                loss, loss_dict = criterion(
                    logits, targets, alphas,
                    meta["retrieval_scores"],
                    metadata=meta,
                    size_targets=output_size,
                )

            # Per-sample outcome signals for the rule proposer: the
            # blended loss feeds the history buffer; the proposal
            # pathway's own loss gates rule commits.
            with torch.no_grad():
                sample_loss, _ = per_sample_ce(logits.detach(), targets, PAD_VALUE)
                if meta["prop_active"]:
                    prop_loss, _ = per_sample_ce(
                        meta["logits_prop"].detach(), targets, PAD_VALUE,
                    )
                else:
                    prop_loss = None

            scaled_loss = loss / accum_steps
            scaled_loss.backward()

            did_step = (step + 1) % accum_steps == 0
            if did_step:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1

            # History update every micro-batch; commits only between
            # optimiser steps (no pending gradients).
            commit_info = model.apply_outcomes(
                sample_loss, prop_loss, allow_commit=did_step,
            )
            if commit_info["committed"]:
                epoch_commits += 1

            # Track metrics.
            B = test_output.size(0)
            epoch_loss += loss_dict["task"] * B
            epoch_samples += B
            c, t = compute_cell_accuracy(logits, targets)
            epoch_correct += c
            epoch_total += t
            alpha_sum += alphas.detach().mean(dim=0).cpu()
            alpha_batches += 1
            last_loss_dict = loss_dict

            # Optional mid-epoch validation for huge datasets.
            if (
                args.val_every > 0
                and did_step
                and global_step % args.val_every == 0
            ):
                val = evaluate(model, val_loader, device)
                model.train()
                metric = (val["solve_rate"], val["cell_acc"])
                marker = ""
                if metric > best_metric:
                    best_metric = metric
                    patience_counter = 0
                    torch.save(
                        model.state_dict(),
                        os.path.join(args.out_dir, "fusion_best.pt"),
                    )
                    marker = "  (saved)"
                else:
                    patience_counter += 1
                print(
                    f"  [step {global_step}] val_solve={val['solve_rate']:.3f}  "
                    f"val_acc={val['cell_acc']:.3f}  "
                    f"val_size_acc={val['size_acc']:.3f}{marker}"
                )
                if patience_counter >= args.patience:
                    print(f"Early stopping at step {global_step} (patience={args.patience})")
                    stop = True
                    break

        # Flush any remaining accumulated gradients at end of epoch.
        if step >= 0 and (step + 1) % accum_steps != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            global_step += 1

        train_loss = epoch_loss / max(epoch_samples, 1)
        train_acc = epoch_correct / max(epoch_total, 1)
        val = evaluate(model, val_loader, device)
        model.train()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val["cell_acc"])
        history["val_solve"].append(val["solve_rate"])
        history["val_size_acc"].append(val["size_acc"])

        current_lr = optimizer.param_groups[0]["lr"]
        alpha_mean = alpha_sum / max(alpha_batches, 1)
        alpha_str = "  ".join(
            f"{name}={alpha_mean[i]:.3f}" for i, name in enumerate(PATHWAY_NAMES)
        )
        aux_str = "  ".join(
            f"{name}={last_loss_dict.get(f'aux_{name}', 0.0):.4f}"
            for name in PATHWAY_NAMES
        )
        print(
            f"Epoch {epoch:3d}/{args.epochs}  "
            f"loss={train_loss:.4f}  train_acc={train_acc:.3f}  "
            f"val_acc={val['cell_acc']:.3f}  val_solve={val['solve_rate']:.3f}  "
            f"val_size_acc={val['size_acc']:.3f}  lr={current_lr:.2e}"
        )
        print(f"  alpha({alpha_str})  commits={epoch_commits}")
        print(
            f"  pathway losses: {aux_str}  "
            f"verify={last_loss_dict.get('verify', 0.0):.4f}  "
            f"size={last_loss_dict.get('size', 0.0):.4f}"
        )

        if stop:
            break

        metric = (val["solve_rate"], val["cell_acc"])
        if metric > best_metric:
            best_metric = metric
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(args.out_dir, "fusion_best.pt"))
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch} (patience={args.patience})")
                break

    return history


# ── Entry point ──────────────────────────────────────────────────────────────


def main() -> None:
    """Parse arguments, build datasets, and launch training."""
    parser = argparse.ArgumentParser(description="Train the Fusion Model")
    parser.add_argument(
        "--data_root",
        type=str,
        default="data",
        help="Path to the data directory containing training/ and evaluation/.",
    )
    parser.add_argument(
        "--parquet_dir",
        type=str,
        default=None,
        help="Path to a directory of .parquet files for training data. "
        "All .parquet files in the directory will be loaded.",
    )
    parser.add_argument(
        "--parquet_files",
        type=str,
        nargs="+",
        default=None,
        help="One or more .parquet file paths for training data.",
    )
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument(
        "--grad_accum",
        type=int,
        default=None,
        help="Effective batch size for gradient accumulation. Must be >= batch_size "
        "and divisible by batch_size. Defaults to batch_size (no accumulation).",
    )
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Validation events without improvement before early stopping.",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=None,
        help="LR warmup in optimiser steps. Default: min(2000, 3%% of total).",
    )
    parser.add_argument(
        "--val_every",
        type=int,
        default=0,
        help="Validate every N optimiser steps (0 = once per epoch). "
        "Strongly recommended for million-sample parquet datasets.",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Use bfloat16 autocast on CUDA (recommended on A40-class GPUs).",
    )
    parser.add_argument(
        "--no_verify",
        action="store_true",
        help="Disable leave-one-out demo verification (saves ~40%% step time, "
        "loses the grounded routing signal).",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="outputs",
        help="Directory for saved weights and training history.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Limit the number of samples per split (useful for debugging).",
    )
    parser.add_argument("--max_grid_size", type=int, default=30)
    parser.add_argument("--max_demos", type=int, default=5)
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--num_encoder_layers", type=int, default=4)
    parser.add_argument("--num_cross_attn_layers", type=int, default=4)
    parser.add_argument("--num_rule_slots", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    # ── Validate gradient accumulation ────────────────────────────────────
    if args.grad_accum is None:
        args.grad_accum = args.batch_size
    if args.grad_accum < args.batch_size:
        print(
            f"Error: --grad_accum ({args.grad_accum}) must be >= "
            f"--batch_size ({args.batch_size}).",
            file=sys.stderr,
        )
        sys.exit(1)
    if args.grad_accum % args.batch_size != 0:
        print(
            f"Error: --grad_accum ({args.grad_accum}) must be divisible by "
            f"--batch_size ({args.batch_size}).",
            file=sys.stderr,
        )
        sys.exit(1)

    # ── Validate parquet args ─────────────────────────────────────────────
    if args.parquet_dir is not None and args.parquet_files is not None:
        print(
            "Error: --parquet_dir and --parquet_files are mutually exclusive.",
            file=sys.stderr,
        )
        sys.exit(1)

    torch.manual_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device: {device}")
    print("Loading datasets...")

    # ── Construct datasets and loaders ───────────────────────────────────
    parquet_paths: list[str] | None = None
    if args.parquet_dir is not None:
        parquet_paths = sorted(
            os.path.join(args.parquet_dir, f)
            for f in os.listdir(args.parquet_dir)
            if f.endswith(".parquet")
        )
        if not parquet_paths:
            print(
                f"Error: no .parquet files found in {args.parquet_dir}.",
                file=sys.stderr,
            )
            sys.exit(1)
    elif args.parquet_files is not None:
        parquet_paths = args.parquet_files

    train_ds: Any
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

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=arc_collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=arc_collate_fn,
    )
    print(f"Train samples: {len(train_ds):,}  |  Val samples: {len(val_ds):,}")

    # ── Instantiate the Fusion Model ─────────────────────────────────────
    print("Building model...")
    model = FusionModel(
        embed_dim=args.embed_dim,
        num_colours=NUM_COLOURS,
        max_grid_size=args.max_grid_size,
        num_encoder_layers=args.num_encoder_layers,
        num_cross_attn_layers=args.num_cross_attn_layers,
        num_rule_slots=args.num_rule_slots,
        max_demos=max(args.max_demos, 1),
        verify_demos=not args.no_verify,
    ).to(device)
    criterion = FusionLoss(pad_value=PAD_VALUE)

    print(f"Fusion Model: {count_params(model):,} trainable parameters")

    history = train_fusion(model, criterion, train_loader, val_loader, device, args)

    history_path = os.path.join(args.out_dir, "fusion_history.json")
    with open(history_path, "w", encoding="utf-8") as fh:
        json.dump(history, fh)
    print(f"History saved to {history_path}")


if __name__ == "__main__":
    main()
