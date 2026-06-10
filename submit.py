"""Generate a Kaggle submission.json for ARC-style challenges.

Loads a challenges JSON file, runs the trained Fusion Model on each test
pair, and writes a ``submission.json`` with two prediction attempts per
test output.

The model predicts the **output grid size** with its size head.  When the
predicted output is larger than the test-input canvas, the sample is
re-run on an enlarged canvas so that every output cell has a token
position (two-pass inference); otherwise the canvas logits are simply
cropped to the predicted size.

Usage::

    python submit.py --challenges arc-agi_evaluation_challenges.json
    python submit.py --challenges challenges.json --checkpoint outputs/fusion_best.pt
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from fusion_model import FusionModel
from tasks.arc import NUM_COLOURS, PAD_VALUE, ChallengesDataset, arc_collate_fn


def challenges_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate challenge samples: dynamic padding + task bookkeeping."""
    task_ids = [s.pop("task_id") for s in batch]
    pair_idxs = [s.pop("test_pair_idx") for s in batch]
    out: dict[str, Any] = arc_collate_fn(batch)
    out["task_id"] = task_ids
    out["test_pair_idx"] = torch.tensor(pair_idxs, dtype=torch.long)
    return out


# ── Inference ─────────────────────────────────────────────────────────────────


def _predict_canvas(
    model: FusionModel,
    demo_inputs: torch.Tensor,
    demo_outputs: torch.Tensor,
    demo_mask: torch.Tensor,
    test_input: torch.Tensor,
    min_h: int,
    min_w: int,
) -> torch.Tensor:
    """Re-run one sample on a canvas of at least ``(min_h, min_w)``.

    Pads all grids (with :data:`PAD_VALUE`) so the output canvas can hold
    a predicted output that is larger than the test input.

    :return: Per-cell logits ``(1, min_h * min_w, num_colours)`` —
        actually ``(1, H2*W2, C)`` for the enlarged canvas.
    """
    H, W = test_input.shape[-2], test_input.shape[-1]
    dh, dw = max(0, min_h - H), max(0, min_w - W)
    if dh or dw:
        pad = (0, dw, 0, dh)  # pad last dim (W) then second-to-last (H)
        demo_inputs = F.pad(demo_inputs, pad, value=PAD_VALUE)
        demo_outputs = F.pad(demo_outputs, pad, value=PAD_VALUE)
        test_input = F.pad(test_input, pad, value=PAD_VALUE)
    logits, _, _ = model(demo_inputs, demo_outputs, demo_mask, test_input)
    return logits


def generate_submissions(
    model: FusionModel,
    loader: DataLoader,  # type: ignore[type-arg]
    device: torch.device,
    temperature: float = 1.5,
) -> list[tuple[str, int, list[list[int]], list[list[int]]]]:
    """Run inference and produce two attempts per test pair.

    :param model: Trained FusionModel.
    :param loader: DataLoader over a :class:`ChallengesDataset`.
    :param device: Torch device.
    :param temperature: Sampling temperature for attempt_2.
    :return: List of ``(task_id, test_pair_idx, attempt_1, attempt_2)``.
    """
    results: list[tuple[str, int, list[list[int]], list[list[int]]]] = []
    model.eval()

    with torch.no_grad():
        for batch in loader:
            demo_inputs = batch["demo_inputs"].to(device)
            demo_outputs = batch["demo_outputs"].to(device)
            demo_mask = batch["demo_mask"].to(device)
            test_input = batch["test_input"].to(device)
            task_ids = batch["task_id"]  # list of strings
            pair_idxs = batch["test_pair_idx"]  # tensor

            logits, _, meta = model(demo_inputs, demo_outputs, demo_mask, test_input)
            size_pred = meta["size_logits"].argmax(dim=-1) + 1  # (B, 2)

            B = test_input.size(0)
            H, W = test_input.size(1), test_input.size(2)

            for i in range(B):
                ph, pw = int(size_pred[i, 0]), int(size_pred[i, 1])

                if ph <= H and pw <= W:
                    grid_logits = logits[i].view(H, W, NUM_COLOURS)[:ph, :pw]
                else:
                    # Predicted output exceeds the canvas: re-run this
                    # sample on an enlarged canvas (second pass).
                    H2, W2 = max(H, ph), max(W, pw)
                    sample_logits = _predict_canvas(
                        model,
                        demo_inputs[i : i + 1],
                        demo_outputs[i : i + 1],
                        demo_mask[i : i + 1],
                        test_input[i : i + 1],
                        H2,
                        W2,
                    )
                    grid_logits = sample_logits[0].view(H2, W2, NUM_COLOURS)[:ph, :pw]

                # attempt_1: greedy argmax
                pred1 = grid_logits.argmax(dim=-1).cpu().tolist()

                # attempt_2: temperature-scaled sampling
                flat_logits = grid_logits.reshape(-1, NUM_COLOURS)
                scaled = flat_logits / temperature
                probs = torch.softmax(scaled, dim=-1)
                sampled = torch.multinomial(probs, num_samples=1).squeeze(-1)
                pred2 = sampled.view(ph, pw).cpu().tolist()

                results.append((
                    task_ids[i],
                    int(pair_idxs[i]),
                    pred1,
                    pred2,
                ))

    return results


# ── Build submission dict ─────────────────────────────────────────────────────


def build_submission(
    results: list[tuple[str, int, list[list[int]], list[list[int]]]],
) -> dict[str, list[dict[str, list[list[int]]]]]:
    """Group results by task_id and format for Kaggle submission."""
    grouped: dict[str, list[tuple[int, list[list[int]], list[list[int]]]]] = (
        defaultdict(list)
    )
    for task_id, pair_idx, a1, a2 in results:
        grouped[task_id].append((pair_idx, a1, a2))

    submission: dict[str, list[dict[str, list[list[int]]]]] = {}
    for task_id, pairs in sorted(grouped.items()):
        pairs.sort(key=lambda x: x[0])
        submission[task_id] = [
            {"attempt_1": a1, "attempt_2": a2} for _, a1, a2 in pairs
        ]
    return submission


# ── CLI ───────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate submission.json for ARC Kaggle competitions",
    )
    parser.add_argument(
        "--challenges", type=str, required=True,
        help="Path to Kaggle challenges JSON file",
    )
    parser.add_argument(
        "--checkpoint", type=str, default="outputs/fusion_best.pt",
        help="Path to trained model weights",
    )
    parser.add_argument("--output", type=str, default="submission.json")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=1.5)
    parser.add_argument("--max_grid_size", type=int, default=30)
    parser.add_argument("--max_demos", type=int, default=5)
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--num_rule_slots", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    # ── Dataset ─────────────────────────────────────────────────────────
    dataset = ChallengesDataset(
        args.challenges,
        max_grid_size=args.max_grid_size,
        max_demos=args.max_demos,
        max_samples=args.max_samples,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=challenges_collate,
        pin_memory=True,
    )
    print(f"Loaded {len(dataset)} test pairs from {args.challenges}")

    # ── Model ───────────────────────────────────────────────────────────
    model = FusionModel(
        embed_dim=args.embed_dim,
        num_colours=NUM_COLOURS,
        max_grid_size=args.max_grid_size,
        num_rule_slots=args.num_rule_slots,
        max_demos=max(args.max_demos, 1),
    ).to(device)
    model.load_state_dict(
        torch.load(args.checkpoint, weights_only=True, map_location=device),
    )
    print(f"Loaded weights from {args.checkpoint}")

    # ── Generate ────────────────────────────────────────────────────────
    results = generate_submissions(model, loader, device, args.temperature)
    submission = build_submission(results)

    # ── Validate completeness ───────────────────────────────────────────
    with open(args.challenges, encoding="utf-8") as fh:
        challenge_ids = set(json.load(fh).keys())
    submission_ids = set(submission.keys())
    missing = challenge_ids - submission_ids
    if missing:
        print(f"WARNING: {len(missing)} task(s) missing from submission: "
              f"{sorted(missing)[:5]}...")
    else:
        print("All task IDs present in submission.")

    # ── Write ───────────────────────────────────────────────────────────
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(submission, fh)
    print(f"Wrote {len(submission)} tasks to {args.output}")


if __name__ == "__main__":
    main()
