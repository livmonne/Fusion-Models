"""CLEVR dataset loader for Visual Question Answering.

The CLEVR dataset (Johnson et al., 2017) consists of rendered 3-D scenes
with accompanying natural-language questions.  Each question has exactly
one correct answer drawn from a fixed vocabulary of 28 tokens (colours,
shapes, materials, sizes, counts 0-10, and yes/no).

This module provides:

* A canonical **answer vocabulary** shared across training and evaluation.
* A lightweight **word-level tokeniser** for the question strings.
* A PyTorch ``Dataset`` that yields ``(image_tensor, question_ids, answer_index)``
  triples ready for data-loading.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# ── Answer vocabulary ───────────────────────────────────────────────────────
# CLEVR answers are one of 28 fixed strings.  We build a deterministic
# mapping so that the classification head always sees the same index for
# the same answer regardless of data split.
ANSWER_VOCAB: list[str] = sorted(
    [
        "0",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
        "10",
        "blue",
        "brown",
        "cube",
        "cyan",
        "cylinder",
        "gray",
        "green",
        "large",
        "metal",
        "no",
        "purple",
        "red",
        "rubber",
        "small",
        "sphere",
        "yellow",
        "yes",
    ]
)
ANSWER_TO_IDX: dict[str, int] = {a: i for i, a in enumerate(ANSWER_VOCAB)}
NUM_ANSWERS: int = len(ANSWER_VOCAB)  # 28

# ── Question tokeniser ─────────────────────────────────────────────────────
# Special token indices used by the tokeniser.
PAD_IDX: int = 0
UNK_IDX: int = 1

# Simple regex-based word splitter: lower-case and keep alphanumeric tokens.
_WORD_RE = re.compile(r"[a-z0-9]+")


def build_question_vocab(questions_json_path: str, min_freq: int = 1) -> dict[str, int]:
    """Scan all questions and build a word -> index mapping.

    :param questions_json_path: Path to a CLEVR ``*_questions.json`` file.
    :param min_freq: Minimum occurrence count for a word to be included.
    :return: Dictionary mapping each word string to its integer index.
    """
    with open(questions_json_path, encoding="utf-8") as fh:
        data: dict[str, Any] = json.load(fh)

    # Count word frequencies across every question.
    freq: dict[str, int] = {}
    for entry in data["questions"]:
        for word in _WORD_RE.findall(entry["question"].lower()):
            freq[word] = freq.get(word, 0) + 1

    # Reserve index 0 for <PAD> and index 1 for <UNK>.
    vocab: dict[str, int] = {"<PAD>": PAD_IDX, "<UNK>": UNK_IDX}
    idx = 2
    for word in sorted(freq):
        if freq[word] >= min_freq:
            vocab[word] = idx
            idx += 1
    return vocab


def encode_question(question: str, vocab: dict[str, int], max_len: int) -> torch.Tensor:
    """Convert a question string into a fixed-length tensor of word indices.

    Words not in *vocab* are mapped to ``<UNK>``.  Sequences shorter than
    *max_len* are right-padded with ``<PAD>``.

    :param question: Raw question string, e.g. ``"How many red cubes are there?"``.
    :param vocab: Word-to-index mapping produced by :func:`build_question_vocab`.
    :param max_len: Output tensor length (longer questions are truncated).
    :return: ``int64`` tensor of shape ``(max_len,)``.
    """
    words = _WORD_RE.findall(question.lower())
    ids = [vocab.get(w, UNK_IDX) for w in words[:max_len]]

    # Right-pad to max_len.
    ids += [PAD_IDX] * (max_len - len(ids))
    return torch.tensor(ids, dtype=torch.long)


# ── Dataset class ───────────────────────────────────────────────────────────

# Standard ImageNet-style normalisation used by most pre-trained CNNs.
_DEFAULT_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


class CLEVRDataset(Dataset):  # type: ignore[type-arg]
    """PyTorch dataset for the CLEVR Visual Question Answering task.

    Each sample is a triple ``(image, question_ids, answer_index)``:

    * **image** — a ``float32`` tensor of shape ``(3, 224, 224)`` after
      resizing and ImageNet normalisation.
    * **question_ids** — an ``int64`` tensor of shape ``(max_q_len,)``
      containing word indices (padded / truncated).
    * **answer_index** — a scalar ``int64`` tensor indexing into
      :data:`ANSWER_VOCAB`.

    :param clevr_root: Path to the ``CLEVR_v1.0`` directory.
    :param split: One of ``"train"`` or ``"val"``.
    :param question_vocab: Pre-built word-to-index mapping.  If ``None`` the
        vocabulary is built on the fly from the training questions file.
    :param max_q_len: Maximum question length in tokens.
    :param transform: Optional ``torchvision`` transform applied to each PIL
        image.  Defaults to resize-224 + ImageNet normalisation.
    :param max_samples: If set, only load this many samples (useful for quick
        debugging runs).
    """

    def __init__(
        self,
        clevr_root: str,
        split: str = "train",
        question_vocab: dict[str, int] | None = None,
        max_q_len: int = 46,
        transform: transforms.Compose | None = None,
        max_samples: int | None = None,
    ) -> None:
        super().__init__()
        self.clevr_root = clevr_root
        self.split = split
        self.max_q_len = max_q_len
        self.transform = transform or _DEFAULT_TRANSFORM

        # ── Load questions JSON ──────────────────────────────────────────
        q_path = os.path.join(clevr_root, "questions", f"CLEVR_{split}_questions.json")
        with open(q_path, encoding="utf-8") as fh:
            raw: dict[str, Any] = json.load(fh)

        # Only keep entries that have an answer (test split does not).
        self.entries: list[dict[str, Any]] = [e for e in raw["questions"] if "answer" in e]
        if max_samples is not None:
            self.entries = self.entries[:max_samples]

        # ── Build or reuse question vocabulary ───────────────────────────
        if question_vocab is None:
            train_q_path = os.path.join(clevr_root, "questions", "CLEVR_train_questions.json")
            self.question_vocab = build_question_vocab(train_q_path)
        else:
            self.question_vocab = question_vocab

        self.image_dir = os.path.join(clevr_root, "images", split)

    # ── Dataset protocol ────────────────────────────────────────────────

    def __len__(self) -> int:
        """Return the number of question-answer pairs in this split."""
        return len(self.entries)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return a single ``(image, question_ids, answer_index)`` sample.

        :param idx: Integer index into the dataset.
        :return: Tuple of tensors described in the class docstring.
        """
        entry = self.entries[idx]

        # ── Image ────────────────────────────────────────────────────────
        img_path = os.path.join(self.image_dir, entry["image_filename"])
        image = Image.open(img_path).convert("RGB")
        image_tensor: torch.Tensor = self.transform(image)

        # ── Question ─────────────────────────────────────────────────────
        question_ids = encode_question(entry["question"], self.question_vocab, self.max_q_len)

        # ── Answer ───────────────────────────────────────────────────────
        answer_idx = ANSWER_TO_IDX[entry["answer"]]
        answer_tensor = torch.tensor(answer_idx, dtype=torch.long)

        return image_tensor, question_ids, answer_tensor
