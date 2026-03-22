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
    for r in range(h):
        for c in range(len(grid[r])):
            padded[r, c] = grid[r][c]
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
        """Return a single sample as a dict of padded tensors.

        Keys returned:

        * ``demo_inputs``  — ``(max_demos, G, G)`` padded demo input grids
        * ``demo_outputs`` — ``(max_demos, G, G)`` padded demo output grids
        * ``demo_mask``    — ``(max_demos,)`` boolean; True for real demos
        * ``test_input``   — ``(G, G)`` padded test input grid
        * ``test_output``  — ``(G, G)`` padded test output grid (or all PAD)
        * ``input_size``   — ``(2,)`` int tensor ``[H, W]`` of test input
        * ``output_size``  — ``(2,)`` int tensor ``[H, W]`` of test output
        """
        sample = self.samples[idx]
        G = self.max_grid_size

        # Pad demonstration pairs.
        demo_inputs = torch.full((self.max_demos, G, G), PAD_VALUE, dtype=torch.long)
        demo_outputs = torch.full((self.max_demos, G, G), PAD_VALUE, dtype=torch.long)
        demo_mask = torch.zeros(self.max_demos, dtype=torch.bool)

        for i, demo in enumerate(sample["demos"][: self.max_demos]):
            demo_inputs[i] = pad_grid(demo["input"], G, G)
            demo_outputs[i] = pad_grid(demo["output"], G, G)
            demo_mask[i] = True

        # Pad test grids.
        test_input = pad_grid(sample["test_input"], G, G)
        ti_h, ti_w = len(sample["test_input"]), len(sample["test_input"][0])

        if sample["test_output"] is not None:
            test_output = pad_grid(sample["test_output"], G, G)
            to_h = len(sample["test_output"])
            to_w = len(sample["test_output"][0]) if to_h > 0 else 0
        else:
            test_output = torch.full((G, G), PAD_VALUE, dtype=torch.long)
            to_h, to_w = 0, 0

        return {
            "demo_inputs": demo_inputs,
            "demo_outputs": demo_outputs,
            "demo_mask": demo_mask,
            "test_input": test_input,
            "test_output": test_output,
            "input_size": torch.tensor([ti_h, ti_w], dtype=torch.long),
            "output_size": torch.tensor([to_h, to_w], dtype=torch.long),
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

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Parse one task on-the-fly and return padded tensors."""
        t_idx, row_idx, tp_idx = self._index[idx]
        table = self._tables[t_idx]

        task_json: str = table.column("task")[row_idx].as_py()
        task: dict[str, Any] = json.loads(task_json)

        sample: dict[str, Any] = {
            "task_id": table.column("id")[row_idx].as_py(),
            "demos": task["train"],
            "test_input": task["test"][tp_idx]["input"],
            "test_output": task["test"][tp_idx].get("output"),
        }

        # Re-use the shared padding logic from the base class.
        G = self.max_grid_size
        demo_inputs = torch.full((self.max_demos, G, G), PAD_VALUE, dtype=torch.long)
        demo_outputs = torch.full((self.max_demos, G, G), PAD_VALUE, dtype=torch.long)
        demo_mask = torch.zeros(self.max_demos, dtype=torch.bool)

        for i, demo in enumerate(sample["demos"][: self.max_demos]):
            demo_inputs[i] = pad_grid(demo["input"], G, G)
            demo_outputs[i] = pad_grid(demo["output"], G, G)
            demo_mask[i] = True

        test_input = pad_grid(sample["test_input"], G, G)
        ti_h, ti_w = len(sample["test_input"]), len(sample["test_input"][0])

        if sample["test_output"] is not None:
            test_output = pad_grid(sample["test_output"], G, G)
            to_h = len(sample["test_output"])
            to_w = len(sample["test_output"][0]) if to_h > 0 else 0
        else:
            test_output = torch.full((G, G), PAD_VALUE, dtype=torch.long)
            to_h, to_w = 0, 0

        return {
            "demo_inputs": demo_inputs,
            "demo_outputs": demo_outputs,
            "demo_mask": demo_mask,
            "test_input": test_input,
            "test_output": test_output,
            "input_size": torch.tensor([ti_h, ti_w], dtype=torch.long),
            "output_size": torch.tensor([to_h, to_w], dtype=torch.long),
        }
