"""GuessComponent — self-attention based dense predictor.

This is the "pattern matching" pathway.  It handles fuzzy, hard-to-formalise
patterns that cannot be captured by crisp rules — for example, recognising
that a transformation involves a holistic spatial rearrangement rather than
a mechanical cell-by-cell procedure.

**Architecture:**

1. The spatial token sequence ``x`` ``(batch, seq, embed_dim)`` is processed
   by a single-layer **multi-head self-attention** block, letting tokens
   attend to each other and discover spatial interactions.
2. A residual connection + LayerNorm stabilises the attended output.
3. A per-token **linear head** maps each token to ``num_colours`` logits.
4. The attended tokens are **mean-pooled** into a single representation
   vector for the router.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class GuessComponent(nn.Module):
    """Self-attention predictor that handles fuzzy / non-rule patterns.

    Now operates on the **full spatial token sequence** rather than
    pseudo-tokens derived from a single pooled vector.

    :param embed_dim: Dimensionality of the per-token embedding.
    :param num_colours: Number of per-cell colour classes (10 for ARC).
    :param num_heads: Number of self-attention heads.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_colours: int = 10,
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim

        self.self_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=0.1, batch_first=True,
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.1)

        # Per-token classification head.
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, num_colours),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute per-token guess-pathway logits.

        :param x: Spatial token embeddings ``(batch, seq, embed_dim)``.
        :return: Tuple of ``(logits, pooled)`` where *logits* has shape
            ``(batch, seq, num_colours)`` and *pooled* is ``(batch, embed_dim)``
            for the router.
        """
        # Self-attention over spatial tokens.
        attended, _ = self.self_attn(x, x, x)
        tokens = self.norm(x + self.dropout(attended))  # (B, seq, E)

        # Per-token classification.
        logits_guess: torch.Tensor = self.head(tokens)  # (B, seq, num_colours)

        # Router representation: mean-pool.
        pooled = tokens.mean(dim=1)  # (B, E)

        return logits_guess, pooled
