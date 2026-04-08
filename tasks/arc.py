"""ARC-AGI-2 dataset loader for grid-to-grid transformation tasks.

The ARC-AGI-2 dataset (Chollet, 2024) consists of abstract reasoning tasks
where each task provides demonstration input/output grid pairs and one or
more test inputs.  The goal is to produce the correct output grid for each
test input by inferring the transformation rule from the demonstrations.

Grids are rectangular matrices of integers in ``[0, 9]`` with dimensions
up to 30×30.  There are 10 possible cell values (visualised as colours).

This module provides:

* Constants for grid encoding (``NUM_COLOURS``, ``MAX_GRID_SIZE``).
* PyTorch ``Dataset`` classes that yield padded/flattened grid tensors
  ready for batched training.  :class:`ARCDataset` loads from a directory
  of JSON files; :class:`ParquetARCDataset` loads from one or more
  ``.parquet`` files (e.g. the Giotto augmented dataset).
* Utility functions for padding, flattening, and reconstructing grids.

Dataset structure expected on disk (JSON)::

    data/
        training/       # 1000 task JSON files
        evaluation/     # 120 task JSON files

Each JSON file contains ``{"train": [...], "test": [...]}`` where each
entry has ``"input"`` and ``"output"`` grids (list of lists of ints).
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import Any

import torch
from torch.utils.data import Dataset

# ── Constants ────────────────────────────────────────────────────────────────
NUM_COLOURS: int = 10  # cell values 0-9
PAD_VALUE: int = -1  # padding sentinel (not a valid colour)
MAX_GRID_SIZE: int = 30  # maximum grid dimension in ARC-AGI-2

# ── Grid utilities ───────────────────────────────────────────────────────────


def pad_grid(grid: list[list[int]], max_h: int, max_w: int) -> torch.Tensor:
    """Pad a variable-size grid to ``(max_h, max_w)`` with :data:`PAD_VALUE`.

    :param grid: 2-D list of integers in ``[0, 9]``.
    :param max_h: Target height.
    :param max_w: Target width.
    :return: ``int64`` tensor of shape ``(max_h, max_w)``.
    """
    h = len(grid)
    w = len(grid[0]) if h > 0 else 0
    padded = torch.full((max_h, max_w), PAD_VALUE, dtype=torch.long)
    if h > 0 and w > 0:
        padded[:h, :w] = torch.tensor(grid, dtype=torch.long)
    return padded


def grid_to_tensor(grid: list[list[int]]) -> torch.Tensor:
    """Convert a raw grid (list of lists) to a tensor without padding.

    :param grid: 2-D list of integers.
    :return: ``int64`` tensor of shape ``(H, W)``.
    """
    return torch.tensor(grid, dtype=torch.long)


def unpad_grid(padded: torch.Tensor, h: int, w: int) -> list[list[int]]:
    """Extract the top-left ``(h, w)`` region from a padded grid tensor.

    :param padded: Tensor of shape ``(max_h, max_w)``.
    :param h: True height.
    :param w: True width.
    :return: 2-D list of integers.
    """
    return padded[:h, :w].tolist()


# ── Base dataset ─────────────────────────────────────────────────────────────


class _BaseARCDataset(Dataset):  # type: ignore[type-arg]
    """Shared ``__getitem__`` logic for all ARC dataset variants.

    Subclasses must populate ``self.samples`` (a list of dicts with keys
    ``task_id``, ``demos``, ``test_input``, ``test_output``) and set
    ``self.max_grid_size`` and ``self.max_demos`` before the first call
    to ``__getitem__``.
    """

    samples: list[dict[str, Any]]
    max_grid_size: int
    max_demos: int

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Return a single sample as a dict of minimally-padded tensors.

        Grids are padded to the sample's own max dimensions (not the global
        max).  A custom collate function (:func:`arc_collate_fn`) re-pads
        to the batch maximum at collation time.

        Keys returned:

        * ``demo_inputs``  — ``(max_demos, H, W)`` padded demo input grids
        * ``demo_outputs`` — ``(max_demos, H, W)`` padded demo output grids
        * ``demo_mask``    — ``(max_demos,)`` boolean; True for real demos
        * ``test_input``   — ``(H, W)`` padded test input grid
        * ``test_output``  — ``(H, W)`` padded test output grid (or all PAD)
        * ``input_size``   — ``(2,)`` int tensor ``[H, W]`` of test input
        * ``output_size``  — ``(2,)`` int tensor ``[H, W]`` of test output
        * ``grid_dims``    — ``(2,)`` int tensor ``[H, W]`` of padded dims
        """
        sample = self.samples[idx]

        # Compute per-sample max grid dims across all grids.
        all_grids: list[list[list[int]]] = []
        for demo in sample["demos"][: self.max_demos]:
            all_grids.append(demo["input"])
            all_grids.append(demo["output"])
        all_grids.append(sample["test_input"])
        if sample["test_output"] is not None:
            all_grids.append(sample["test_output"])

        sample_h = max(len(g) for g in all_grids)
        sample_w = max(len(g[0]) if len(g) > 0 else 0 for g in all_grids)
        # Clamp to global max for safety.
        sample_h = min(sample_h, self.max_grid_size)
        sample_w = min(sample_w, self.max_grid_size)

        # Pad demonstration pairs.
        demo_inputs = torch.full(
            (self.max_demos, sample_h, sample_w), PAD_VALUE, dtype=torch.long,
        )
        demo_outputs = torch.full(
            (self.max_demos, sample_h, sample_w), PAD_VALUE, dtype=torch.long,
        )
        demo_mask = torch.zeros(self.max_demos, dtype=torch.bool)

        for i, demo in enumerate(sample["demos"][: self.max_demos]):
            demo_inputs[i] = pad_grid(demo["input"], sample_h, sample_w)
            demo_outputs[i] = pad_grid(demo["output"], sample_h, sample_w)
            demo_mask[i] = True

        # Pad test grids.
        test_input = pad_grid(sample["test_input"], sample_h, sample_w)
        ti_h, ti_w = len(sample["test_input"]), len(sample["test_input"][0])

        if sample["test_output"] is not None:
            test_output = pad_grid(sample["test_output"], sample_h, sample_w)
            to_h = len(sample["test_output"])
            to_w = len(sample["test_output"][0]) if to_h > 0 else 0
        else:
            test_output = torch.full(
                (sample_h, sample_w), PAD_VALUE, dtype=torch.long,
            )
            to_h, to_w = 0, 0

        return {
            "demo_inputs": demo_inputs,
            "demo_outputs": demo_outputs,
            "demo_mask": demo_mask,
            "test_input": test_input,
            "test_output": test_output,
            "input_size": torch.tensor([ti_h, ti_w], dtype=torch.long),
            "output_size": torch.tensor([to_h, to_w], dtype=torch.long),
            "grid_dims": torch.tensor([sample_h, sample_w], dtype=torch.long),
        }


# ── JSON dataset ─────────────────────────────────────────────────────────────


class ARCDataset(_BaseARCDataset):
    """PyTorch dataset for ARC-AGI-2 grid transformation tasks (JSON files).

    Each sample represents a single *test pair* from a task, bundled with
    all of that task's demonstration pairs as context.  The model receives
    the demo inputs/outputs and the test input, and must predict the test
    output.

    All grids are padded to ``(max_grid_size, max_grid_size)`` so they can
    be batched.  A ``PAD_VALUE`` sentinel (−1) marks cells outside the
    original grid boundaries.

    :param data_dir: Path to a directory of ARC task JSON files (e.g.
        ``data/training/``).
    :param max_grid_size: Pad all grids to this square size.
    :param max_demos: Maximum number of demonstration pairs to include.
        Tasks with more demos are truncated; tasks with fewer are
        zero-padded.
    :param max_samples: If set, only load this many samples (for debugging).
    """

    def __init__(
        self,
        data_dir: str,
        max_grid_size: int = MAX_GRID_SIZE,
        max_demos: int = 5,
        max_samples: int | None = None,
    ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.max_grid_size = max_grid_size
        self.max_demos = max_demos

        self.samples: list[dict[str, Any]] = []
        self._load_tasks(max_samples)

    def _load_tasks(self, max_samples: int | None) -> None:
        """Scan the data directory and build the sample list."""
        task_files = sorted(
            f for f in os.listdir(self.data_dir) if f.endswith(".json")
        )
        for fname in task_files:
            path = os.path.join(self.data_dir, fname)
            with open(path, encoding="utf-8") as fh:
                task: dict[str, Any] = json.load(fh)

            demos = task["train"]
            for test_pair in task["test"]:
                self.samples.append({
                    "task_id": fname.replace(".json", ""),
                    "demos": demos,
                    "test_input": test_pair["input"],
                    "test_output": test_pair.get("output"),
                })
                if max_samples is not None and len(self.samples) >= max_samples:
                    return


# ── Parquet dataset ──────────────────────────────────────────────────────────


class ChallengesDataset(_BaseARCDataset):
    """Dataset for a single Kaggle challenges JSON file.

    The Kaggle ARC-AGI-2 evaluation challenges are distributed as one JSON
    file mapping task IDs to ``{"train": [...], "test": [{"input": ...}]}``.
    Ground-truth outputs are not available, so ``test_output`` is always
    ``None``.

    Each test pair is expanded into a separate sample (matching
    :class:`ARCDataset` behaviour).  The ``task_id`` (string) and
    ``test_pair_idx`` (int) are included in the dict returned by
    ``__getitem__`` so that results can be grouped back by task.

    :param challenges_path: Path to the challenges JSON file.
    :param max_grid_size: Pad all grids to this square size.
    :param max_demos: Maximum number of demonstration pairs to include.
    :param max_samples: If set, only load this many samples (for debugging).
    """

    def __init__(
        self,
        challenges_path: str,
        max_grid_size: int = MAX_GRID_SIZE,
        max_demos: int = 5,
        max_samples: int | None = None,
    ) -> None:
        super().__init__()
        self.max_grid_size = max_grid_size
        self.max_demos = max_demos

        self.samples: list[dict[str, Any]] = []
        self._load_challenges(challenges_path, max_samples)

    def _load_challenges(
        self, challenges_path: str, max_samples: int | None
    ) -> None:
        """Parse the challenges JSON and build the sample list."""
        with open(challenges_path, encoding="utf-8") as fh:
            challenges: dict[str, Any] = json.load(fh)

        for task_id in sorted(challenges):
            task = challenges[task_id]
            demos = task["train"]
            for tp_idx, test_pair in enumerate(task["test"]):
                self.samples.append({
                    "task_id": task_id,
                    "test_pair_idx": tp_idx,
                    "demos": demos,
                    "test_input": test_pair["input"],
                    "test_output": test_pair.get("output"),
                })
                if max_samples is not None and len(self.samples) >= max_samples:
                    return

    def __getitem__(self, idx: int) -> dict[str, Any]:
        result: dict[str, Any] = super().__getitem__(idx)
        result["task_id"] = self.samples[idx]["task_id"]
        result["test_pair_idx"] = self.samples[idx]["test_pair_idx"]
        return result


class ParquetARCDataset(_BaseARCDataset):
    """PyTorch dataset that lazily loads ARC tasks from ``.parquet`` files.

    Each parquet file must have columns ``id`` (string) and ``task``
    (JSON string with the standard ARC format:
    ``{"train": [...], "test": [...]}``.  All test pairs within each task
    are expanded into separate samples, matching :class:`ARCDataset`
    behaviour.

    **Lazy loading**: Only a lightweight index is built at init time
    (mapping each sample to a table row + test-pair offset).  The actual
    JSON parsing and grid construction happen on-the-fly in
    ``__getitem__``.  This keeps RAM usage proportional to the compressed
    parquet size (~1.3 GB for the full Giotto dataset) rather than the
    fully materialised Python objects (~95 GB).

    :param parquet_paths: List of paths to ``.parquet`` files to load.
    :param max_grid_size: Pad all grids to this square size.
    :param max_demos: Maximum number of demonstration pairs to include.
    :param max_samples: If set, only index this many samples (for
        debugging).
    """

    def __init__(
        self,
        parquet_paths: list[str],
        max_grid_size: int = MAX_GRID_SIZE,
        max_demos: int = 5,
        max_samples: int | None = None,
    ) -> None:
        super().__init__()
        self.max_grid_size = max_grid_size
        self.max_demos = max_demos

        # Populated by _build_index; not used by __getitem__ (lazy).
        self.samples: list[dict[str, Any]] = []

        self._tables: list[Any] = []  # pyarrow Tables kept in memory
        self._index: list[tuple[int, int, int]] = []  # (table_idx, row_idx, test_pair_idx)
        self._build_index(parquet_paths, max_samples)

    def _build_index(
        self, parquet_paths: list[str], max_samples: int | None
    ) -> None:
        """Scan parquet files and build a sample index without materialising grids."""
        import pyarrow.parquet as pq

        for path in parquet_paths:
            table = pq.read_table(path, columns=["id", "task"])
            t_idx = len(self._tables)
            self._tables.append(table)

            tasks_col = table.column("task")
            for row_idx in range(len(table)):
                # Quick parse to count test pairs only.
                task: dict[str, Any] = json.loads(tasks_col[row_idx].as_py())
                n_test = len(task.get("test", []))
                for tp_idx in range(n_test):
                    self._index.append((t_idx, row_idx, tp_idx))
                    if max_samples is not None and len(self._index) >= max_samples:
                        print(
                            f"Indexed {len(self._index)} samples from "
                            f"{len(self._tables)} parquet file(s) (capped)"
                        )
                        return

        print(
            f"Indexed {len(self._index)} samples from "
            f"{len(self._tables)} parquet file(s)"
        )

    # ── Overrides ─────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._index)

    @lru_cache(maxsize=4096)
    def _parse_task(self, t_idx: int, row_idx: int) -> dict[str, Any]:
        """Parse and cache a task's JSON. Avoids re-parsing the same task."""
        table = self._tables[t_idx]
        task_json: str = table.column("task")[row_idx].as_py()
        return json.loads(task_json)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Parse one task on-the-fly and return padded tensors."""
        t_idx, row_idx, tp_idx = self._index[idx]
        table = self._tables[t_idx]

        task: dict[str, Any] = self._parse_task(t_idx, row_idx)

        sample: dict[str, Any] = {
            "task_id": table.column("id")[row_idx].as_py(),
            "demos": task["train"],
            "test_input": task["test"][tp_idx]["input"],
            "test_output": task["test"][tp_idx].get("output"),
        }

        # Temporarily stash sample for base class __getitem__.
        self.samples = [sample]
        result = super().__getitem__(0)
        self.samples = []
        return result


# ── Collate function ────────────────────────────────────────────────────────


def arc_collate_fn(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Collate ARC samples with dynamic per-batch padding.

    Instead of padding every grid to the global maximum (30×30), this pads
    to the maximum grid dimensions within the current batch.  This can
    reduce sequence length by ~6× and attention memory by ~36× for typical
    batches.
    """
    batch_h = max(int(s["grid_dims"][0]) for s in batch)
    batch_w = max(int(s["grid_dims"][1]) for s in batch)
    B = len(batch)
    max_demos = batch[0]["demo_inputs"].size(0)

    # Pre-allocate batch tensors filled with PAD_VALUE.
    demo_inputs = torch.full((B, max_demos, batch_h, batch_w), PAD_VALUE, dtype=torch.long)
    demo_outputs = torch.full((B, max_demos, batch_h, batch_w), PAD_VALUE, dtype=torch.long)
    test_input = torch.full((B, batch_h, batch_w), PAD_VALUE, dtype=torch.long)
    test_output = torch.full((B, batch_h, batch_w), PAD_VALUE, dtype=torch.long)

    demo_mask_list = []
    input_size_list = []
    output_size_list = []

    for i, s in enumerate(batch):
        h, w = int(s["grid_dims"][0]), int(s["grid_dims"][1])
        demo_inputs[i, :, :h, :w] = s["demo_inputs"]
        demo_outputs[i, :, :h, :w] = s["demo_outputs"]
        test_input[i, :h, :w] = s["test_input"]
        test_output[i, :h, :w] = s["test_output"]
        demo_mask_list.append(s["demo_mask"])
        input_size_list.append(s["input_size"])
        output_size_list.append(s["output_size"])

    return {
        "demo_inputs": demo_inputs,
        "demo_outputs": demo_outputs,
        "demo_mask": torch.stack(demo_mask_list),
        "test_input": test_input,
        "test_output": test_output,
        "input_size": torch.stack(input_size_list),
        "output_size": torch.stack(output_size_list),
    }
