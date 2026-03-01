"""Evaluation and visualisation for the Fusion Model on CLEVR.

Run this after ``train.py`` to generate:

1. **Accuracy table** — overall and per-answer-type.
2. **Router behaviour bar chart** — mean routing weights broken down by
   question type (count / compare / exist / query).
3. **Alpha heatmap** — per-answer average routing weights.
4. **Training curves** — loss and accuracy over epochs.
5. **Rule utility histogram** — memory slot utilisation.
"""

from __future__ import annotations

import argparse
import json
import os
import re

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from fusion_model import FusionModel
from tasks.clevr import ANSWER_VOCAB, NUM_ANSWERS, CLEVRDataset, build_question_vocab

# Use the non-interactive Agg backend so plots can be saved headlessly.
matplotlib.use("Agg")

# ── Question-type heuristic ─────────────────────────────────────────────────
# CLEVR questions can be roughly categorised by their first few words.

_TYPE_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("count", re.compile(r"^how many\b", re.I)),
    ("exist", re.compile(r"^(is there|are there)\b", re.I)),
    (
        "compare",
        re.compile(
            r"^(is the .+ the same|do the .+ have the same|is the .+ (bigger|smaller))", re.I
        ),
    ),
    ("query", re.compile(r"^what (color|material|size|shape|number)\b", re.I)),
]


def classify_question(question: str) -> str:
    """Return a coarse question type based on simple regex patterns.

    :param question: Raw question string.
    :return: One of ``"count"``, ``"exist"``, ``"compare"``, ``"query"``,
        or ``"other"``.
    """
    for name, pattern in _TYPE_PATTERNS:
        if pattern.search(question):
            return name
    return "other"


# ── Gather predictions ──────────────────────────────────────────────────────


def gather_predictions(
    model: FusionModel,
    loader: DataLoader,  # type: ignore[type-arg]
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run inference and collect predictions, targets, and routing alphas.

    :param model: Trained Fusion Model.
    :param loader: DataLoader yielding ``(images, questions, answers)`` batches.
    :param device: Torch device.
    :return: Tuple of NumPy arrays ``(preds, targets, alphas)`` with shapes
        ``(N,)``, ``(N,)``, ``(N, 3)``.
    """
    all_preds: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []
    all_alphas: list[torch.Tensor] = []
    model.eval()

    with torch.no_grad():
        for images, questions, answers in loader:
            images = images.to(device)
            questions = questions.to(device)
            logits, alphas, _ = model(images, questions)
            all_preds.append(logits.argmax(dim=-1).cpu())
            all_targets.append(answers)
            all_alphas.append(alphas.cpu())

    preds_arr: np.ndarray = torch.cat(all_preds).numpy()
    targets_arr: np.ndarray = torch.cat(all_targets).numpy()
    alphas_arr: np.ndarray = torch.cat(all_alphas).numpy()
    return preds_arr, targets_arr, alphas_arr


# ── Visualisations ──────────────────────────────────────────────────────────


def print_accuracy_table(preds: np.ndarray, targets: np.ndarray) -> None:
    """Print overall and per-answer accuracy.

    :param preds: Predicted class indices.
    :param targets: Ground-truth class indices.
    """
    overall = (preds == targets).mean()
    print(f"\nOverall accuracy: {overall:.4f}")
    print(f"\n{'Answer':<12} {'Accuracy':>8} {'Count':>8}")
    print("-" * 32)
    for idx, name in enumerate(ANSWER_VOCAB):
        mask = targets == idx
        if mask.sum() == 0:
            continue
        acc = (preds[mask] == targets[mask]).mean()
        print(f"{name:<12} {acc:>8.3f} {int(mask.sum()):>8}")


def plot_router_by_qtype(
    alphas: np.ndarray,
    questions_raw: list[str],
    out_dir: str,
) -> None:
    """Bar chart of mean routing weights per question type.

    :param alphas: Array of shape ``(N, 3)`` with routing weights.
    :param questions_raw: List of raw question strings parallel to *alphas*.
    :param out_dir: Directory in which to save the plot.
    """
    # Group alphas by question type.
    groups: dict[str, list[np.ndarray]] = {}
    for alpha_row, q in zip(alphas, questions_raw, strict=False):
        qtype = classify_question(q)
        groups.setdefault(qtype, []).append(alpha_row)

    type_names = sorted(groups)
    means = np.array([np.mean(groups[t], axis=0) for t in type_names])

    x = np.arange(len(type_names))
    width = 0.25
    labels = [r"$\alpha_{mem}$", r"$\alpha_{rule}$", r"$\alpha_{guess}$"]

    fig, ax = plt.subplots(figsize=(8, 4))
    for i, label in enumerate(labels):
        ax.bar(x + i * width, means[:, i], width, label=label)
    ax.set_xticks(x + width)
    ax.set_xticklabels(type_names)
    ax.set_ylabel("Mean routing weight")
    ax.set_title("Router Behaviour by Question Type")
    ax.legend()
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "router_by_qtype.png"), dpi=150)
    plt.close(fig)
    print("Saved router_by_qtype.png")


def plot_alpha_heatmap(alphas: np.ndarray, targets: np.ndarray, out_dir: str) -> None:
    """Heatmap of average routing weights per answer class.

    :param alphas: Array of shape ``(N, 3)``.
    :param targets: Ground-truth class indices ``(N,)``.
    :param out_dir: Output directory.
    """
    class_alphas = np.zeros((NUM_ANSWERS, 3))
    for c in range(NUM_ANSWERS):
        mask = targets == c
        if mask.sum() > 0:
            class_alphas[c] = alphas[mask].mean(axis=0)

    fig, ax = plt.subplots(figsize=(5, 10))
    im = ax.imshow(class_alphas, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels([r"$\alpha_{mem}$", r"$\alpha_{rule}$", r"$\alpha_{guess}$"])
    ax.set_yticks(range(NUM_ANSWERS))
    ax.set_yticklabels(ANSWER_VOCAB)
    ax.set_title("Routing Weights per Answer")

    for i in range(NUM_ANSWERS):
        for j in range(3):
            ax.text(j, i, f"{class_alphas[i, j]:.2f}", ha="center", va="center", fontsize=7)

    fig.colorbar(im, ax=ax, shrink=0.6)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "alpha_heatmap.png"), dpi=150)
    plt.close(fig)
    print("Saved alpha_heatmap.png")


def plot_training_curves(out_dir: str) -> None:
    """Plot loss and accuracy curves from the saved training history.

    :param out_dir: Directory containing ``fusion_history.json``.
    """
    with open(os.path.join(out_dir, "fusion_history.json"), encoding="utf-8") as fh:
        hist: dict[str, list[float]] = json.load(fh)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(hist["train_loss"])
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Training Loss")
    axes[0].set_title("Training Loss")

    axes[1].plot(hist["train_acc"], label="train")
    axes[1].plot(hist["val_acc"], label="val")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "training_curves.png"), dpi=150)
    plt.close(fig)
    print("Saved training_curves.png")


def plot_rule_utility(model: FusionModel, out_dir: str) -> None:
    """Bar chart showing how much each memory slot is used.

    :param model: Trained Fusion Model (contains the utility buffer).
    :param out_dir: Output directory.
    """
    utility_tensor = model.memory.utility.cpu()
    utility: np.ndarray = utility_tensor.numpy()

    fig, ax = plt.subplots(figsize=(7, 3))
    ax.bar(range(len(utility)), utility)
    ax.set_xlabel("Rule Slot Index")
    ax.set_ylabel("Utility (EMA of retrieval weight)")
    ax.set_title("Rule Memory Slot Utilisation")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "rule_utility.png"), dpi=150)
    plt.close(fig)
    print("Saved rule_utility.png")


# ── Entry point ──────────────────────────────────────────────────────────────


def main() -> None:
    """Load the best model, run evaluation, and generate all plots."""
    parser = argparse.ArgumentParser(description="Evaluate Fusion Model on CLEVR")
    parser.add_argument("--clevr_root", type=str, default="CLEVR_v1.0")
    parser.add_argument("--out_dir", type=str, default="outputs")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    # ── Build vocab and dataset ──────────────────────────────────────────
    train_q_path = os.path.join(args.clevr_root, "questions", "CLEVR_train_questions.json")
    question_vocab: dict[str, int] = build_question_vocab(train_q_path)
    vocab_size = len(question_vocab)

    val_ds = CLEVRDataset(
        args.clevr_root,
        split="val",
        question_vocab=question_vocab,
        max_samples=args.max_samples,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    # ── Load trained model ───────────────────────────────────────────────
    model = FusionModel(vocab_size=vocab_size, num_classes=NUM_ANSWERS).to(device)
    ckpt_path = os.path.join(args.out_dir, "fusion_best.pt")
    model.load_state_dict(torch.load(ckpt_path, weights_only=True, map_location=device))
    print(f"Loaded weights from {ckpt_path}")

    # ── Predictions ──────────────────────────────────────────────────────
    preds, targets, alphas = gather_predictions(model, val_loader, device)

    # ── 1. Accuracy table ────────────────────────────────────────────────
    print_accuracy_table(preds, targets)

    # ── 2. Router by question type ───────────────────────────────────────
    # We need the raw question strings to classify by type.
    questions_raw: list[str] = [e["question"] for e in val_ds.entries]
    if args.max_samples is not None:
        questions_raw = questions_raw[: args.max_samples]
    plot_router_by_qtype(alphas, questions_raw, args.out_dir)

    # ── 3. Alpha heatmap ─────────────────────────────────────────────────
    plot_alpha_heatmap(alphas, targets, args.out_dir)

    # ── 4. Training curves ───────────────────────────────────────────────
    plot_training_curves(args.out_dir)

    # ── 5. Rule utility ──────────────────────────────────────────────────
    plot_rule_utility(model, args.out_dir)

    print(f"\nAll plots saved to {args.out_dir}/")


if __name__ == "__main__":
    main()
