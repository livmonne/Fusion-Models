"""Training script for the Fusion Model on the ARC-AGI-2 dataset.

Usage::

    python train.py --data_root data --epochs 40
    python train.py --parquet_dir /path/to/parquets --epochs 40
    python train.py --parquet_files a.parquet b.parquet --epochs 40

The script:

1. Constructs PyTorch ``DataLoader`` instances for the training and
   evaluation splits of ARC-AGI-2.
2. Instantiates the :class:`~fusion_model.FusionModel` and
   :class:`~fusion_model.FusionLoss`.
3. Trains with AdamW + linear warmup + cosine-annealing LR, printing
   metrics every epoch.
4. Saves the best model weights (by validation accuracy) and a JSON
   training history for later analysis.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch
from torch.utils.data import DataLoader

from fusion_model import FusionModel
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


# ── Evaluation ───────────────────────────────────────────────────────────────


@torch.no_grad()
def evaluate(
    model: FusionModel,
    loader: DataLoader,  # type: ignore[type-arg]
    device: torch.device,
    max_grid_size: int = 30,
) -> float:
    """Compute per-cell accuracy on a data loader.

    :param model: The Fusion Model in eval mode.
    :param loader: DataLoader yielding sample dicts.
    :param device: Torch device.
    :param max_grid_size: Grid size used for flattening targets.
    :return: Accuracy as a float in ``[0, 1]``.
    """
    model.eval()
    correct = 0
    total = 0
    for batch in loader:
        demo_inputs = batch["demo_inputs"].to(device)
        demo_outputs = batch["demo_outputs"].to(device)
        demo_mask = batch["demo_mask"].to(device)
        test_input = batch["test_input"].to(device)
        test_output = batch["test_output"].to(device)

        logits, _, _ = model(demo_inputs, demo_outputs, demo_mask, test_input)

        targets_flat = test_output.view(test_output.size(0), -1)
        max_cells = logits.size(1)
        targets_flat = targets_flat[:, :max_cells]

        c, t = compute_cell_accuracy(logits, targets_flat)
        correct += c
        total += t

    return correct / total if total > 0 else 0.0


# ── Training loop ────────────────────────────────────────────────────────────


def train_fusion(
    model: FusionModel,
    criterion: FusionLoss,
    train_loader: DataLoader,  # type: ignore[type-arg]
    val_loader: DataLoader,  # type: ignore[type-arg]
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, list[float]]:
    """Run the full training loop for the Fusion Model.

    Uses AdamW with a linear warmup phase followed by cosine annealing.
    Supports gradient accumulation via ``args.grad_accum`` (effective batch
    size; must be a multiple of ``args.batch_size``).

    :param model: The Fusion Model to train.
    :param criterion: The :class:`~fusion_model.FusionLoss` instance.
    :param train_loader: Training data loader.
    :param val_loader: Validation data loader.
    :param device: Torch device.
    :param args: Parsed command-line arguments.
    :return: Dictionary with ``train_loss``, ``train_acc``, and ``val_acc``
        lists (one entry per epoch).
    """
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # ── LR schedule: linear warmup → cosine annealing ────────────────────
    warmup_epochs = min(5, args.epochs // 4)
    if warmup_epochs > 0:
        warmup_sched = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, total_iters=warmup_epochs,
        )
        cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, args.epochs - warmup_epochs),
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_sched, cosine_sched],
            milestones=[warmup_epochs],
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs,
        )

    # ── Gradient accumulation ────────────────────────────────────────────
    accum_steps = args.grad_accum // args.batch_size

    best_val_acc = 0.0
    patience_counter = 0
    history: dict[str, list[float]] = {"train_loss": [], "train_acc": [], "val_acc": []}

    ce_per_sample = torch.nn.CrossEntropyLoss(
        ignore_index=PAD_VALUE, reduction="none",
    )

    print(
        f"Warmup: {warmup_epochs} epochs | "
        f"Effective batch size: {args.grad_accum} "
        f"(micro={args.batch_size} x accum={accum_steps})"
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0

        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            demo_inputs = batch["demo_inputs"].to(device)
            demo_outputs = batch["demo_outputs"].to(device)
            demo_mask = batch["demo_mask"].to(device)
            test_input = batch["test_input"].to(device)
            test_output = batch["test_output"].to(device)

            logits, alphas, meta = model(
                demo_inputs, demo_outputs, demo_mask, test_input,
            )

            # Flatten targets to match logits shape: (B, seq).
            B = test_output.size(0)
            targets_flat = test_output.view(B, -1)  # (B, G*G)
            max_cells = logits.size(1)
            targets_flat = targets_flat[:, :max_cells]

            # Per-sample loss for the rule generator's history buffer.
            C = logits.size(-1)
            per_cell_loss = ce_per_sample(
                logits.reshape(-1, C),
                targets_flat.reshape(-1),
            ).view(B, -1)
            valid_mask = (targets_flat != PAD_VALUE).float()
            per_sample_loss = (
                (per_cell_loss * valid_mask).sum(dim=-1)
                / valid_mask.sum(dim=-1).clamp(min=1.0)
            ).detach()

            # Feed outcome signal to the rule generator's history.
            model.update_rule_history(per_sample_loss)

            # Compute combined loss (task CE + all regularisation terms).
            loss, loss_dict = criterion(
                logits, targets_flat, alphas,
                meta["retrieval_scores"],
                metadata=meta,
            )

            # Scale loss for gradient accumulation and backward.
            scaled_loss = loss / accum_steps
            scaled_loss.backward()

            # Optimizer step every accum_steps micro-batches.
            if (step + 1) % accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

            # Track unscaled metrics for logging.
            epoch_loss += loss_dict["task"] * B
            c, t = compute_cell_accuracy(logits, targets_flat)
            epoch_correct += c
            epoch_total += t

        # Flush any remaining accumulated gradients at end of epoch.
        if (step + 1) % accum_steps != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        scheduler.step()

        train_loss = epoch_loss / max(epoch_total, 1)
        train_acc = epoch_correct / max(epoch_total, 1)
        val_acc = evaluate(model, val_loader, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        current_lr = optimizer.param_groups[0]["lr"]
        alpha_mean = (
            f"mem={alphas[:, 0].mean():.3f}  "
            f"rule={alphas[:, 1].mean():.3f}  "
            f"guess={alphas[:, 2].mean():.3f}"
        )
        print(
            f"Epoch {epoch:3d}/{args.epochs}  "
            f"loss={train_loss:.4f}  train_acc={train_acc:.3f}  "
            f"val_acc={val_acc:.3f}  lr={current_lr:.2e}  "
            f"alpha({alpha_mean})"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
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
    parser = argparse.ArgumentParser(description="Train Fusion Model on ARC-AGI-2")
    parser.add_argument(
        "--data_root",
        type=str,
        default="data",
        help="Path to the ARC-AGI-2 data directory containing training/ and evaluation/.",
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
    parser.add_argument("--patience", type=int, default=10)
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
    model = FusionModel(
        embed_dim=args.embed_dim,
        num_colours=NUM_COLOURS,
        max_grid_size=args.max_grid_size,
        num_encoder_layers=args.num_encoder_layers,
        num_cross_attn_layers=args.num_cross_attn_layers,
    ).to(device)
    criterion = FusionLoss(pad_value=PAD_VALUE)

    print(f"Fusion Model: {count_params(model):,} trainable parameters")
    print("Training...")

    history = train_fusion(model, criterion, train_loader, val_loader, device, args)

    history_path = os.path.join(args.out_dir, "fusion_history.json")
    with open(history_path, "w", encoding="utf-8") as fh:
        json.dump(history, fh)
    print(f"History saved to {history_path}")


if __name__ == "__main__":
    main()
