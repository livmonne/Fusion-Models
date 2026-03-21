"""GuessComponent — self-attention based dense predictor.

This is the "pattern matching" pathway.  It handles fuzzy, hard-to-formalise
patterns that cannot be captured by crisp rules — for example, recognising
that a transformation involves a holistic spatial rearrangement rather than
a mechanical cell-by-cell procedure.

**Architecture:**

1. The shared embedding ``h`` (a single vector per sample) is reshaped into
   a sequence of small *pseudo-tokens*.  For example a 256-dim vector becomes
   16 tokens of dimension 16.
2. A single-head **self-attention** layer lets the tokens attend to each other,
   discovering interactions between different parts of the representation.
3. The attended tokens are **mean-pooled** back into a single vector.
4. A two-layer **MLP head** maps the pooled vector to classification logits.

This gives the pathway capacity similar to a lightweight transformer block
while remaining cheap to train.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GuessComponent(nn.Module):
    """Self-attention predictor that handles fuzzy / non-rule patterns.

    :param embed_dim: Dimensionality of the shared input embedding.
    :param num_classes: Number of output classes.
    :param num_tokens: How many pseudo-tokens to split the embedding into.
    :param head_hidden: Hidden width of the classification MLP head.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_classes: int = 9000,
        num_tokens: int = 16,
        head_hidden: int = 256,
    ) -> None:
        super().__init__()
        self.num_tokens = num_tokens
        # Each pseudo-token has this many dimensions.
        self.token_dim = embed_dim // num_tokens

        # Standard Q / K / V projections for self-attention.
        self.q_proj = nn.Linear(self.token_dim, self.token_dim)
        self.k_proj = nn.Linear(self.token_dim, self.token_dim)
        self.v_proj = nn.Linear(self.token_dim, self.token_dim)
        self.out_proj = nn.Linear(self.token_dim, self.token_dim)

        self.norm = nn.LayerNorm(self.token_dim)
        self.dropout = nn.Dropout(0.1)

        # Two-layer MLP classification head.
        self.head = nn.Sequential(
            nn.Linear(embed_dim, head_hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(head_hidden, num_classes),
        )

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute guess-pathway logits from the shared embedding.

        :param h: Shared embedding of shape ``(batch, embed_dim)``.
        :return: Tuple of ``(logits, pooled)`` where *logits* has shape
            ``(batch, num_classes)`` and *pooled* is the intermediate
            representation ``(batch, embed_dim)`` before the classification
            head (used by the :class:`~fusion_model.decision.DecisionRouter`).
        """
        batch = h.shape[0]

        # -- 1. Reshape the flat vector into a sequence of pseudo-tokens. --
        tokens = h.view(batch, self.num_tokens, self.token_dim)

        # -- 2. Single-head scaled-dot-product self-attention. --
        q = self.q_proj(tokens)  # (batch, T, d)
        k = self.k_proj(tokens)
        v = self.v_proj(tokens)

        scale = self.token_dim**0.5
        attn_weights = torch.bmm(q, k.transpose(1, 2)) / scale  # (batch, T, T)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        attended = torch.bmm(attn_weights, v)  # (batch, T, d)

        # -- 3. Residual connection + layer norm. --
        tokens = self.norm(tokens + self.out_proj(attended))

        # -- 4. Mean-pool tokens back into one vector, then classify. --
        pooled = tokens.reshape(batch, -1)  # (batch, embed_dim)
        logits_guess: torch.Tensor = self.head(pooled)
        return logits_guess, pooled
