"""Evaluation and visualisation for the Fusion Model on ARC-AGI-2.

Run this after ``train.py`` to generate:

1. **Accuracy summary** — overall per-cell accuracy and per-task solve rate.
2. **Router behaviour** — mean routing weights across tasks.
3. **Training curves** — loss and accuracy over epochs.
4. **Rule utility histogram** — memory slot utilisation.
5. **Grid visualisation** — side-by-side input / predicted / ground-truth
   grids for a sample of tasks.
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap
from torch.utils.data import DataLoader

from fusion_model import FusionModel
from tasks.arc import NUM_COLOURS, PAD_VALUE, ARCDataset, arc_collate_fn

matplotlib.use("Agg")

# ARC colour palette (matches the official ARC visualiser).
ARC_COLOURS = [
    "#000000",  # 0: black
    "#0074D9",  # 1: blue
    "#FF4136",  # 2: red
    "#2ECC40",  # 3: green
    "#FFDC00",  # 4: yellow
    "#AAAAAA",  # 5: grey
    "#F012BE",  # 6: magenta
    "#FF851B",  # 7: orange
    "#7FDBFF",  # 8: cyan
    "#B10DC9",  # 9: maroon
]
ARC_CMAP = ListedColormap(ARC_COLOURS)


# ── Gather predictions ──────────────────────────────────────────────────────


def gather_predictions(
    model: FusionModel,
    loader: DataLoader,  # type: ignore[type-arg]
    device: torch.device,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], np.ndarray]:
    """Run inference and collect per-sample predictions.

    :return: Tuple of ``(pred_grids, target_grids, input_grids, alphas)``
        where grids are lists of 2-D arrays and alphas is ``(N, 3)``.
    """
    all_preds: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    all_inputs: list[np.ndarray] = []
    all_alphas: list[torch.Tensor] = []
    model.eval()

    with torch.no_grad():
        for batch in loader:
            demo_inputs = batch["demo_inputs"].to(device)
            demo_outputs = batch["demo_outputs"].to(device)
            demo_mask = batch["demo_mask"].to(device)
            test_input = batch["test_input"].to(device)
            test_output = batch["test_output"]
            output_size = batch["output_size"]
            input_size = batch["input_size"]

            logits, alphas, _ = model(demo_inputs, demo_outputs, demo_mask, test_input)
            preds = logits.argmax(dim=-1).cpu()  # (B, max_cells)
            all_alphas.append(alphas.cpu())

            B = test_input.size(0)
            H, W = test_input.size(1), test_input.size(2)
            for i in range(B):
                oh, ow = output_size[i].tolist()
                ih, iw = input_size[i].tolist()

                pred_grid = preds[i].view(H, W)[:oh, :ow].numpy() if oh > 0 and ow > 0 else np.zeros((1, 1), dtype=int)
                tgt_grid = test_output[i][:oh, :ow].numpy() if oh > 0 and ow > 0 else np.zeros((1, 1), dtype=int)
                inp_grid = test_input[i].cpu()[:ih, :iw].numpy()

                all_preds.append(pred_grid)
                all_targets.append(tgt_grid)
                all_inputs.append(inp_grid)

    alphas_arr: np.ndarray = torch.cat(all_alphas).numpy()
    return all_preds, all_targets, all_inputs, alphas_arr


# ── Visualisations ──────────────────────────────────────────────────────────


def print_accuracy_summary(
    preds: list[np.ndarray], targets: list[np.ndarray]
) -> None:
    """Print per-cell accuracy and task solve rate."""
    total_correct = 0
    total_cells = 0
    tasks_solved = 0

    for pred, tgt in zip(preds, targets, strict=True):
        mask = tgt >= 0
        correct = (pred[mask] == tgt[mask]).sum()
        cells = mask.sum()
        total_correct += correct
        total_cells += cells
        if correct == cells and cells > 0:
            tasks_solved += 1

    cell_acc = total_correct / total_cells if total_cells > 0 else 0.0
    solve_rate = tasks_solved / len(preds) if preds else 0.0
    print(f"\nPer-cell accuracy: {cell_acc:.4f}")
    print(f"Task solve rate:   {solve_rate:.4f} ({tasks_solved}/{len(preds)})")


def plot_grid(
    ax: plt.Axes, grid: np.ndarray, title: str
) -> None:
    """Plot a single ARC grid on a matplotlib axis."""
    ax.imshow(grid, cmap=ARC_CMAP, vmin=0, vmax=9, interpolation="nearest")
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("gray")


def plot_sample_grids(
    inputs: list[np.ndarray],
    preds: list[np.ndarray],
    targets: list[np.ndarray],
    out_dir: str,
    num_samples: int = 8,
) -> None:
    """Visualise input / predicted / ground-truth grids side by side."""
    n = min(num_samples, len(inputs))
    fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    for i in range(n):
        plot_grid(axes[i, 0], inputs[i], "Input")
        plot_grid(axes[i, 1], preds[i], "Predicted")
        plot_grid(axes[i, 2], targets[i], "Ground Truth")

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "sample_grids.png"), dpi=150)
    plt.close(fig)
    print("Saved sample_grids.png")


def plot_router_summary(alphas: np.ndarray, out_dir: str) -> None:
    """Bar chart of mean routing weights."""
    means = alphas.mean(axis=0)
    labels = [r"$\alpha_{mem}$", r"$\alpha_{rule}$", r"$\alpha_{guess}$"]

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar(labels, means, color=["#0074D9", "#FF4136", "#2ECC40"])
    ax.set_ylabel("Mean routing weight")
    ax.set_title("Router Behaviour (ARC-AGI-2)")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "router_summary.png"), dpi=150)
    plt.close(fig)
    print("Saved router_summary.png")


def plot_training_curves(out_dir: str) -> None:
    """Plot loss and accuracy curves from the saved training history."""
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
    axes[1].set_ylabel("Per-Cell Accuracy")
    axes[1].set_title("Accuracy")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "training_curves.png"), dpi=150)
    plt.close(fig)
    print("Saved training_curves.png")


def plot_rule_utility(model: FusionModel, out_dir: str) -> None:
    """Bar chart showing how much each memory slot is used."""
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
    parser = argparse.ArgumentParser(description="Evaluate Fusion Model on ARC-AGI-2")
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--out_dir", type=str, default="outputs")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_grid_size", type=int, default=30)
    parser.add_argument("--max_demos", type=int, default=5)
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    # ── Build dataset ────────────────────────────────────────────────────
    eval_dir = os.path.join(args.data_root, "evaluation")
    val_ds = ARCDataset(
        eval_dir,
        max_grid_size=args.max_grid_size,
        max_demos=args.max_demos,
        max_samples=args.max_samples,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=arc_collate_fn,
    )

    # ── Load trained model ───────────────────────────────────────────────
    model = FusionModel(
        embed_dim=args.embed_dim,
        num_colours=NUM_COLOURS,
        max_grid_size=args.max_grid_size,
    ).to(device)
    ckpt_path = os.path.join(args.out_dir, "fusion_best.pt")
    state = torch.load(ckpt_path, weights_only=True, map_location=device)
    # Compatibility: old checkpoints have pos_encoding as (G*G, E) instead of (G, G, E).
    pe = state.get("pos_encoding")
    if pe is not None and pe.dim() == 2:
        G = int(pe.size(0) ** 0.5)
        state["pos_encoding"] = pe.view(G, G, -1)
    model.load_state_dict(state)
    print(f"Loaded weights from {ckpt_path}")

    # ── Predictions ──────────────────────────────────────────────────────
    preds, targets, inputs, alphas = gather_predictions(model, val_loader, device)

    # ── 1. Accuracy summary ──────────────────────────────────────────────
    print_accuracy_summary(preds, targets)

    # ── 2. Router summary ────────────────────────────────────────────────
    plot_router_summary(alphas, args.out_dir)

    # ── 3. Sample grid visualisations ────────────────────────────────────
    plot_sample_grids(inputs, preds, targets, args.out_dir)

    # ── 4. Training curves ───────────────────────────────────────────────
    plot_training_curves(args.out_dir)

    # ── 5. Rule utility ──────────────────────────────────────────────────
    plot_rule_utility(model, args.out_dir)

    print(f"\nAll plots saved to {args.out_dir}/")


if __name__ == "__main__":
    main()
