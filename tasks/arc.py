"""ARC-AGI-2 dataset loader for grid-to-grid transformation tasks.

The ARC-AGI-2 dataset (Chollet, 2024) consists of abstract reasoning tasks
where each task provides demonstration input/output grid pairs and one or
more test inputs.  The goal is to produce the correct output grid for each
test input by inferring the transformation rule from the demonstrations.

Grids are rectangular matrices of integers in ``[0, 9]`` with dimensions
up to 30×30.  There are 10 possible cell values (visualised as colours).

This module provides:

* Constants for grid encoding (``NUM_COLOURS``, ``MAX_GRID_SIZE``).
* Dataset classes that yield padded/flattened grid arrays ready for
  batched training.  :class:`ARCDataset` loads from a directory of JSON
  files; :class:`ParquetARCDataset` loads from one or more ``.parquet``
  files (e.g. the Giotto augmented dataset).
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

import numpy as np

# ── Constants ────────────────────────────────────────────────────────────────
NUM_COLOURS: int = 10  # cell values 0-9
PAD_VALUE: int = -1  # padding sentinel (not a valid colour)
MAX_GRID_SIZE: int = 30  # maximum grid dimension in ARC-AGI-2

# ── Grid utilities ───────────────────────────────────────────────────────────


def pad_grid(grid: list[list[int]], max_h: int, max_w: int) -> np.ndarray:
    """Pad a variable-size grid to ``(max_h, max_w)`` with :data:`PAD_VALUE`.

    :param grid: 2-D list of integers in ``[0, 9]``.
    :param max_h: Target height.
    :param max_w: Target width.
    :return: ``int32`` array of shape ``(max_h, max_w)``.
    """
    h = len(grid)
    w = len(grid[0]) if h > 0 else 0
    padded = np.full((max_h, max_w), PAD_VALUE, dtype=np.int32)
    for r in range(h):
        for c in range(len(grid[r])):
            padded[r, c] = grid[r][c]
    return padded


def grid_to_array(grid: list[list[int]]) -> np.ndarray:
    """Convert a raw grid (list of lists) to an array without padding.

    :param grid: 2-D list of integers.
    :return: ``int32`` array of shape ``(H, W)``.
    """
    return np.array(grid, dtype=np.int32)


def unpad_grid(padded: np.ndarray, h: int, w: int) -> list[list[int]]:
    """Extract the top-left ``(h, w)`` region from a padded grid array.

    :param padded: Array of shape ``(max_h, max_w)``.
    :param h: True height.
    :param w: True width.
    :return: 2-D list of integers.
    """
    return padded[:h, :w].tolist()


# ── Base dataset ─────────────────────────────────────────────────────────────


class _BaseARCDataset:
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

    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
        """Return a single sample as a dict of padded arrays.

        Keys returned:

        * ``demo_inputs``  — ``(max_demos, G, G)`` padded demo input grids
        * ``demo_outputs`` — ``(max_demos, G, G)`` padded demo output grids
        * ``demo_mask``    — ``(max_demos,)`` boolean; True for real demos
        * ``test_input``   — ``(G, G)`` padded test input grid
        * ``test_output``  — ``(G, G)`` padded test output grid (or all PAD)
        * ``input_size``   — ``(2,)`` int array ``[H, W]`` of test input
        * ``output_size``  — ``(2,)`` int array ``[H, W]`` of test output
        """
        sample = self.samples[idx]
        G = self.max_grid_size

        # Pad demonstration pairs.
        demo_inputs = np.full((self.max_demos, G, G), PAD_VALUE, dtype=np.int32)
        demo_outputs = np.full((self.max_demos, G, G), PAD_VALUE, dtype=np.int32)
        demo_mask = np.zeros(self.max_demos, dtype=np.bool_)

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
            test_output = np.full((G, G), PAD_VALUE, dtype=np.int32)
            to_h, to_w = 0, 0

        return {
            "demo_inputs": demo_inputs,
            "demo_outputs": demo_outputs,
            "demo_mask": demo_mask,
            "test_input": test_input,
            "test_output": test_output,
            "input_size": np.array([ti_h, ti_w], dtype=np.int32),
            "output_size": np.array([to_h, to_w], dtype=np.int32),
        }


# ── JSON dataset ─────────────────────────────────────────────────────────────


class ARCDataset(_BaseARCDataset):
    """Dataset for ARC-AGI-2 grid transformation tasks (JSON files).

    Each sample represents a single *test pair* from a task, bundled with
    all of that task's demonstration pairs as context.  The model receives
    the demo inputs/outputs and the test input, and must predict the test
    output.

    All grids are padded to ``(max_grid_size, max_grid_size)`` so they can
    be batched.  A ``PAD_VALUE`` sentinel (−1) marks cells outside the
    original grid boundaries.

    :param data_dir: Path to a directory of ARC task JSON files.
    :param max_grid_size: Pad all grids to this square size.
    :param max_demos: Maximum number of demonstration pairs to include.
    :param max_samples: If set, only load this many samples (for debugging).
    """

    def __init__(
        self,
        data_dir: str,
        max_grid_size: int = MAX_GRID_SIZE,
        max_demos: int = 5,
        max_samples: int | None = None,
    ) -> None:
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
    """Dataset that lazily loads ARC tasks from ``.parquet`` files.

    Each parquet file must have columns ``id`` (string) and ``task``
    (JSON string with the standard ARC format:
    ``{"train": [...], "test": [...]}``.

    **Lazy loading**: Only a lightweight index is built at init time.

    :param parquet_paths: List of paths to ``.parquet`` files to load.
    :param max_grid_size: Pad all grids to this square size.
    :param max_demos: Maximum number of demonstration pairs to include.
    :param max_samples: If set, only index this many samples.
    """

    def __init__(
        self,
        parquet_paths: list[str],
        max_grid_size: int = MAX_GRID_SIZE,
        max_demos: int = 5,
        max_samples: int | None = None,
    ) -> None:
        self.max_grid_size = max_grid_size
        self.max_demos = max_demos

        self.samples: list[dict[str, Any]] = []

        self._tables: list[Any] = []
        self._index: list[tuple[int, int, int]] = []
        self._build_index(parquet_paths, max_samples)

    def _build_index(
        self, parquet_paths: list[str], max_samples: int | None
    ) -> None:
        """Scan parquet files and build a sample index."""
        import pyarrow.parquet as pq

        for path in parquet_paths:
            table = pq.read_table(path, columns=["id", "task"])
            t_idx = len(self._tables)
            self._tables.append(table)

            tasks_col = table.column("task")
            for row_idx in range(len(table)):
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

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
        """Parse one task on-the-fly and return padded arrays."""
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

        G = self.max_grid_size
        demo_inputs = np.full((self.max_demos, G, G), PAD_VALUE, dtype=np.int32)
        demo_outputs = np.full((self.max_demos, G, G), PAD_VALUE, dtype=np.int32)
        demo_mask = np.zeros(self.max_demos, dtype=np.bool_)

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
            test_output = np.full((G, G), PAD_VALUE, dtype=np.int32)
            to_h, to_w = 0, 0

        return {
            "demo_inputs": demo_inputs,
            "demo_outputs": demo_outputs,
            "demo_mask": demo_mask,
            "test_input": test_input,
            "test_output": test_output,
            "input_size": np.array([ti_h, ti_w], dtype=np.int32),
            "output_size": np.array([to_h, to_w], dtype=np.int32),
        }


# ── Data loading utilities ───────────────────────────────────────────────────


def collate_batch(samples: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """Stack a list of samples into a batched dict of numpy arrays."""
    return {key: np.stack([s[key] for s in samples]) for key in samples[0]}


def data_loader(
    dataset: _BaseARCDataset,
    batch_size: int,
    shuffle: bool = False,
    rng: np.random.Generator | None = None,
) -> Any:
    """Simple generator-based data loader (no multiprocessing).

    :param dataset: An ARC dataset instance.
    :param batch_size: Number of samples per batch.
    :param shuffle: Whether to shuffle indices each epoch.
    :param rng: NumPy random generator for shuffling.
    :yields: Batched dicts of numpy arrays.
    """
    n = len(dataset)
    indices = np.arange(n)
    if shuffle:
        if rng is None:
            rng = np.random.default_rng()
        rng.shuffle(indices)

    for start in range(0, n, batch_size):
        batch_indices = indices[start : start + batch_size]
        samples = [dataset[int(i)] for i in batch_indices]
        yield collate_batch(samples)
