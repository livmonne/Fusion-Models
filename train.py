"""Training script for the Fusion Model on the CLEVR VQA dataset.

Usage::

    python train.py --clevr_root CLEVR_v1.0 --epochs 20

The script:

1. Builds a word vocabulary from the CLEVR training questions.
2. Constructs PyTorch ``DataLoader`` instances for the train and val splits.
3. Instantiates the :class:`~fusion_model.FusionModel` and
   :class:`~fusion_model.FusionLoss`.
4. Trains with Adam + cosine-annealing LR, printing metrics every epoch.
5. Saves the best model weights (by validation accuracy) and a JSON
   training history for later analysis.
"""

from __future__ import annotations

import argparse
import json
import os
import random

import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset

from fusion_model import FusionModel
from fusion_model.loss import FusionLoss
from tasks.clevr import NUM_ANSWERS, CLEVRDataset, build_question_vocab

# ── Helpers ──────────────────────────────────────────────────────────────────


def count_params(model: torch.nn.Module) -> int:
    """Return the number of trainable parameters in *model*.

    :param model: Any ``torch.nn.Module``.
    :return: Total number of parameters with ``requires_grad=True``.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def stratified_split(
    dataset: CLEVRDataset,
    fraction: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    """Split dataset indices into *keep* / *leftover* sets, stratified by answer.

    Every answer class contributes approximately *fraction* of its samples to
    the keep set (at least one sample per class).  Indices within each class
    are shuffled deterministically using *seed*.

    :param dataset: A :class:`CLEVRDataset` whose ``.entries`` have an
        ``"answer"`` key.
    :param fraction: Fraction of samples to keep, in ``(0, 1]``.
    :param seed: RNG seed for reproducibility.
    :return: ``(keep_indices, leftover_indices)`` — two disjoint lists whose
        union covers every index in the dataset.
    """
    label_to_indices: dict[str, list[int]] = {}
    for i, entry in enumerate(dataset.entries):
        label_to_indices.setdefault(entry["answer"], []).append(i)

    rng = random.Random(seed)
    keep: list[int] = []
    leftover: list[int] = []

    for label in sorted(label_to_indices):
        indices = label_to_indices[label]
        rng.shuffle(indices)
        n_keep = max(1, round(len(indices) * fraction))
        keep.extend(indices[:n_keep])
        leftover.extend(indices[n_keep:])

    return keep, leftover


@torch.no_grad()
def evaluate(
    model: FusionModel,
    loader: DataLoader,  # type: ignore[type-arg]
    device: torch.device,
) -> float:
    """Compute top-1 accuracy on a data loader.

    :param model: The Fusion Model in eval mode.
    :param loader: DataLoader yielding ``(images, questions, answers)`` batches.
    :param device: Torch device to run on.
    :return: Accuracy as a float in ``[0, 1]``.
    """
    model.eval()
    correct = 0
    total = 0
    for images, questions, answers in loader:
        images = images.to(device)
        questions = questions.to(device)
        answers = answers.to(device)

        logits, _, _ = model(images, questions)
        preds = logits.argmax(dim=-1)
        correct += (preds == answers).sum().item()
        total += answers.size(0)
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

    :param model: The Fusion Model to train.
    :param criterion: The :class:`~fusion_model.FusionLoss` instance.
    :param train_loader: Training data loader.
    :param val_loader: Validation data loader.
    :param device: Torch device.
    :param args: Parsed command-line arguments (learning rate, epochs, etc.).
    :return: Dictionary with ``train_loss``, ``train_acc``, and ``val_acc``
        lists (one entry per epoch).
    """
    optimizer = torch.optim.Adam(
        # Only optimise parameters that require gradients (backbone is frozen).
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_acc = 0.0
    patience_counter = 0
    history: dict[str, list[float]] = {"train_loss": [], "train_acc": [], "val_acc": []}

    out_dir: str = args.out_dir

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0

        for images, questions, answers in train_loader:
            images = images.to(device)
            questions = questions.to(device)
            answers = answers.to(device)

            # -- Forward pass through the Fusion Model. --
            logits, alphas, meta = model(images, questions)

            # -- Compute the composite loss. --
            loss, loss_dict = criterion(
                logits, answers, alphas, meta["retrieval_scores"], metadata=meta
            )

            # -- Back-propagate and update weights. --
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # -- Accumulate epoch statistics. --
            epoch_loss += loss_dict["total"] * answers.size(0)
            preds = logits.argmax(dim=-1)
            epoch_correct += (preds == answers).sum().item()
            epoch_total += answers.size(0)

        scheduler.step()

        train_loss = epoch_loss / epoch_total
        train_acc = epoch_correct / epoch_total
        val_acc = evaluate(model, val_loader, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        # -- Print a progress summary. --
        alpha_mean = (
            f"mem={alphas[:, 0].mean():.3f}  "
            f"rule={alphas[:, 1].mean():.3f}  "
            f"guess={alphas[:, 2].mean():.3f}"
        )
        print(
            f"Epoch {epoch:3d}/{args.epochs}  "
            f"loss={train_loss:.4f}  train_acc={train_acc:.3f}  "
            f"val_acc={val_acc:.3f}  alpha({alpha_mean})"
        )

        # -- Early stopping: save the best model so far. --
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(out_dir, "fusion_best.pt"))
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch} (patience={args.patience})")
                break

    return history


# ── Entry point ──────────────────────────────────────────────────────────────


def main() -> None:
    """Parse arguments, build datasets, and launch training."""
    parser = argparse.ArgumentParser(description="Train Fusion Model on CLEVR")
    parser.add_argument(
        "--clevr_root",
        type=str,
        default="CLEVR_v1.0",
        help="Path to the CLEVR_v1.0 directory containing images/ and questions/.",
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=5)
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
    parser.add_argument(
        "--train_fraction",
        type=float,
        default=1.0,
        help=(
            "Fraction of training data to use (0.0-1.0, default 1.0). "
            "The split is stratified by answer class for balanced sub-sampling."
        ),
    )
    parser.add_argument(
        "--val_fraction",
        type=float,
        default=1.0,
        help=(
            "Fraction of validation data to use (0.0–1.0, default 1.0). "
            "The split is stratified by answer class for balanced sub-sampling."
        ),
    )
    parser.add_argument(
        "--extend_val",
        action="store_true",
        help="When --train_fraction < 1.0, add unused training samples to the validation set.",
    )
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    # -- Select the best available accelerator. --
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device: {device}")

    # ── Build question vocabulary from the training split ────────────────
    train_q_path = os.path.join(args.clevr_root, "questions", "CLEVR_train_questions.json")
    question_vocab: dict[str, int] = build_question_vocab(train_q_path)
    vocab_size = len(question_vocab)
    print(f"Question vocabulary size: {vocab_size}")

    # ── Construct datasets and loaders ───────────────────────────────────
    train_ds = CLEVRDataset(
        args.clevr_root,
        split="train",
        question_vocab=question_vocab,
        max_samples=args.max_samples,
    )
    val_ds = CLEVRDataset(
        args.clevr_root,
        split="val",
        question_vocab=question_vocab,
        max_samples=args.max_samples,
    )

    # ── Optionally sub-sample training data (stratified by answer) ──────
    train_set: Dataset = train_ds  # type: ignore[type-arg]
    val_set: Dataset = val_ds  # type: ignore[type-arg]

    if args.train_fraction < 1.0:
        keep_idx, leftover_idx = stratified_split(train_ds, args.train_fraction, args.seed)
        train_set = Subset(train_ds, keep_idx)
        print(
            f"Sub-sampled training set: {len(keep_idx):,} of {len(train_ds):,} samples "
            f"({args.train_fraction:.0%})"
        )
        if args.extend_val and leftover_idx:
            val_set = ConcatDataset([val_ds, Subset(train_ds, leftover_idx)])
            print(f"  + {len(leftover_idx):,} leftover samples appended to validation set")

    if args.val_fraction < 1.0:
        val_keep_idx, _ = stratified_split(val_ds, args.val_fraction, args.seed)
        val_set = Subset(val_ds, val_keep_idx)
        print(
            f"Sub-sampled validation set: {len(val_keep_idx):,} of {len(val_ds):,} samples "
            f"({args.val_fraction:.0%})"
        )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    print(f"Train samples: {len(train_set):,}  |  Val samples: {len(val_set):,}")

    # ── Instantiate the Fusion Model ─────────────────────────────────────
    model = FusionModel(
        vocab_size=vocab_size,
        num_classes=NUM_ANSWERS,
    ).to(device)
    criterion = FusionLoss()

    print(f"Fusion Model: {count_params(model):,} trainable parameters")
    print("Training...")

    history = train_fusion(model, criterion, train_loader, val_loader, device, args)

    # ── Save training history for the evaluation script ──────────────────
    history_path = os.path.join(args.out_dir, "fusion_history.json")
    with open(history_path, "w", encoding="utf-8") as fh:
        json.dump(history, fh)
    print(f"History saved to {history_path}")


if __name__ == "__main__":
    main()
