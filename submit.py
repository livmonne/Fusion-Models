"""Generate a Kaggle submission.json for ARC-AGI-2.

Loads a challenges JSON file, runs the trained Fusion Model on each test
pair, and writes a ``submission.json`` with two prediction attempts per
test output.

Usage::

    python submit.py --challenges arc-agi_evaluation_challenges.json
    python submit.py --challenges challenges.json --checkpoint outputs/fusion_best.pt
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict

import torch
from torch.utils.data import DataLoader

from fusion_model import FusionModel
from tasks.arc import NUM_COLOURS, ChallengesDataset


# ── Inference ─────────────────────────────────────────────────────────────────


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
            input_size = batch["input_size"]
            task_ids = batch["task_id"]  # list of strings
            pair_idxs = batch["test_pair_idx"]  # tensor

            logits, _, _ = model(demo_inputs, demo_outputs, demo_mask, test_input)
            B = test_input.size(0)
            G = test_input.size(1)

            for i in range(B):
                h, w = input_size[i].tolist()
                if h <= 0 or w <= 0:
                    h, w = 1, 1

                grid_logits = logits[i].view(G, G, NUM_COLOURS)[:h, :w]

                # attempt_1: greedy argmax
                pred1 = grid_logits.argmax(dim=-1).cpu().tolist()

                # attempt_2: temperature-scaled sampling
                flat_logits = grid_logits.reshape(-1, NUM_COLOURS)
                scaled = flat_logits / temperature
                probs = torch.softmax(scaled, dim=-1)
                sampled = torch.multinomial(probs, num_samples=1).squeeze(-1)
                pred2 = sampled.view(h, w).cpu().tolist()

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
        description="Generate submission.json for ARC-AGI-2 Kaggle competition",
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
        pin_memory=True,
    )
    print(f"Loaded {len(dataset)} test pairs from {args.challenges}")

    # ── Model ───────────────────────────────────────────────────────────
    model = FusionModel(
        embed_dim=args.embed_dim,
        num_colours=NUM_COLOURS,
        max_grid_size=args.max_grid_size,
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
