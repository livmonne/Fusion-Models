"""Evaluation and visualisation for the Fusion Model on ARC-AGI-2 (JAX/Flax).

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
import pickle

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp
from matplotlib.colors import ListedColormap

from fusion_model import FusionModel
from tasks.arc import NUM_COLOURS, PAD_VALUE, ARCDataset, data_loader

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
    params: dict,
    model_state: dict | None,
    dataset: ARCDataset,
    batch_size: int,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], np.ndarray, dict[str, np.ndarray]]:
    """Run inference and collect per-sample predictions and metadata."""
    all_preds: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    all_inputs: list[np.ndarray] = []
    all_alphas: list[np.ndarray] = []
    all_retrieval_scores: list[np.ndarray] = []
    all_strength: list[np.ndarray] = []
    all_rule_confidence: list[np.ndarray] = []

    for batch in data_loader(dataset, batch_size=batch_size, shuffle=False):
        batch_jax = {k: jnp.array(v) for k, v in batch.items()}

        variables = {"params": params}
        if model_state is not None:
            variables["state"] = model_state

        logits, alphas, meta = model.apply(
            variables,
            batch_jax["demo_inputs"],
            batch_jax["demo_outputs"],
            batch_jax["demo_mask"],
            batch_jax["test_input"],
            training=False,
        )

        preds = np.array(logits.argmax(axis=-1))  # (B, max_cells)
        all_alphas.append(np.array(alphas))
        all_retrieval_scores.append(np.array(meta["retrieval_scores"]))
        all_strength.append(np.array(meta["memory_strength"]))
        all_rule_confidence.append(np.array(meta["rule_confidence"]))

        B = batch["test_input"].shape[0]
        G = batch["test_input"].shape[1]
        for i in range(B):
            oh, ow = batch["output_size"][i].tolist()
            ih, iw = batch["input_size"][i].tolist()

            pred_grid = preds[i].reshape(G, G)[:oh, :ow] if oh > 0 and ow > 0 else np.zeros((1, 1), dtype=int)
            tgt_grid = batch["test_output"][i][:oh, :ow] if oh > 0 and ow > 0 else np.zeros((1, 1), dtype=int)
            inp_grid = batch["test_input"][i][:ih, :iw]

            all_preds.append(pred_grid)
            all_targets.append(tgt_grid)
            all_inputs.append(inp_grid)

    alphas_arr = np.concatenate(all_alphas)
    meta_arrays = {
        "retrieval_scores": np.concatenate(all_retrieval_scores),
        "strength": all_strength[0] if all_strength else np.array([]),
        "rule_confidence": np.concatenate(all_rule_confidence),
    }
    return all_preds, all_targets, all_inputs, alphas_arr, meta_arrays


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


def plot_grid(ax: plt.Axes, grid: np.ndarray, title: str) -> None:
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


def print_memory_summary(
    model_state: dict | None,
    meta_arrays: dict[str, np.ndarray],
) -> None:
    """Print a detailed summary of the rule memory state and retrieval behaviour."""
    print("\n── Rule Memory Summary ─────────────────────────────────────────")

    retrieval = meta_arrays["retrieval_scores"]  # (N, S)
    strength = meta_arrays["strength"]            # (S,)
    confidence = meta_arrays["rule_confidence"]   # (N,) or (N, 1)

    mean_retrieval = retrieval.mean(axis=0)       # (S,)
    num_slots = len(mean_retrieval)
    top_k = min(10, num_slots)

    print(f"  Slots:            {num_slots}")
    print(f"  Rule confidence:  mean={confidence.mean():.4f}  "
          f"min={confidence.min():.4f}  max={confidence.max():.4f}")

    if len(strength) > 0:
        print(f"  Slot strength:    mean={strength.mean():.4f}  "
              f"min={strength.min():.4f}  max={strength.max():.4f}  "
              f"active(>0.1)={int((strength > 0.1).sum())}/{num_slots}")

    if model_state is not None and "memory" in model_state:
        mem = model_state["memory"]
        freq = np.array(mem.get("frequency", []))
        utility = np.array(mem.get("utility", []))
        steps = np.array(mem.get("steps_since_activation", []))

        if len(freq) > 0:
            print(f"  Slot frequency:   mean={freq.mean():.4f}  "
                  f"min={freq.min():.4f}  max={freq.max():.4f}")
        if len(utility) > 0:
            print(f"  Slot utility:     mean={utility.mean():.4f}  "
                  f"min={utility.min():.4f}  max={utility.max():.4f}  "
                  f"active(>0.01)={int((utility > 0.01).sum())}/{num_slots}")
        if len(steps) > 0:
            print(f"  Steps since act:  mean={steps.mean():.0f}  "
                  f"min={steps.min():.0f}  max={steps.max():.0f}")

    # Top-k most retrieved slots.
    top_idx = np.argsort(mean_retrieval)[::-1][:top_k]
    print(f"\n  Top {top_k} retrieved slots (by mean score across eval set):")
    for rank, idx in enumerate(top_idx):
        s = f"    #{rank+1:2d}  slot {idx:3d}  retrieval={mean_retrieval[idx]:.4f}"
        if len(strength) > 0:
            s += f"  strength={strength[idx]:.4f}"
        if model_state is not None and "memory" in model_state:
            utility = np.array(model_state["memory"].get("utility", []))
            if len(utility) > idx:
                s += f"  utility={utility[idx]:.4f}"
        print(s)

    print()


def plot_rule_utility(model_state: dict, out_dir: str) -> None:
    """Bar chart showing how much each memory slot is used."""
    utility = np.array(model_state["memory"]["utility"])

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
    parser = argparse.ArgumentParser(description="Evaluate Fusion Model on ARC-AGI-2 (JAX)")
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--out_dir", type=str, default="outputs")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_grid_size", type=int, default=30)
    parser.add_argument("--max_demos", type=int, default=5)
    parser.add_argument("--embed_dim", type=int, default=256)
    args = parser.parse_args()

    print(f"JAX devices: {jax.devices()}")

    # ── Build dataset ────────────────────────────────────────────────────
    eval_dir = os.path.join(args.data_root, "evaluation")
    val_ds = ARCDataset(
        eval_dir,
        max_grid_size=args.max_grid_size,
        max_demos=args.max_demos,
        max_samples=args.max_samples,
    )

    # ── Load trained model ───────────────────────────────────────────────
    model = FusionModel(
        embed_dim=args.embed_dim,
        num_colours=NUM_COLOURS,
        max_grid_size=args.max_grid_size,
    )

    ckpt_path = os.path.join(args.out_dir, "fusion_best.pkl")
    with open(ckpt_path, "rb") as f:
        ckpt = pickle.load(f)
    params = ckpt["params"]
    model_state = ckpt.get("model_state")
    print(f"Loaded weights from {ckpt_path}")

    # ── Predictions ──────────────────────────────────────────────────────
    preds, targets, inputs, alphas, meta_arrays = gather_predictions(
        model, params, model_state, val_ds, args.batch_size,
    )

    # ── 1. Accuracy summary ──────────────────────────────────────────────
    print_accuracy_summary(preds, targets)

    # ── 2. Memory summary ────────────────────────────────────────────────
    print_memory_summary(model_state, meta_arrays)

    # ── 3. Router summary ────────────────────────────────────────────────
    plot_router_summary(alphas, args.out_dir)

    # ── 4. Sample grid visualisations ────────────────────────────────────
    plot_sample_grids(inputs, preds, targets, args.out_dir)

    # ── 5. Training curves ───────────────────────────────────────────────
    plot_training_curves(args.out_dir)

    # ── 6. Rule utility ──────────────────────────────────────────────────
    if model_state is not None:
        plot_rule_utility(model_state, args.out_dir)

    print(f"\nAll plots saved to {args.out_dir}/")


if __name__ == "__main__":
    main()
