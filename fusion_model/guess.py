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

import jax.numpy as jnp
import flax.linen as nn


class GuessComponent(nn.Module):
    """Self-attention predictor that handles fuzzy / non-rule patterns.

    Now operates on the **full spatial token sequence** rather than
    pseudo-tokens derived from a single pooled vector.

    :param embed_dim: Dimensionality of the per-token embedding.
    :param num_colours: Number of per-cell colour classes (10 for ARC).
    :param num_heads: Number of self-attention heads.
    """

    embed_dim: int = 256
    num_colours: int = 10
    num_heads: int = 4

    @nn.compact
    def __call__(
        self, x: jnp.ndarray, *, training: bool = False,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Compute per-token guess-pathway logits.

        :param x: Spatial token embeddings ``(batch, seq, embed_dim)``.
        :param training: Whether we are in training mode.
        :return: Tuple of ``(logits, pooled)`` where *logits* has shape
            ``(batch, seq, num_colours)`` and *pooled* is ``(batch, embed_dim)``
            for the router.
        """
        # Self-attention over spatial tokens.
        attended = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.embed_dim,
            dropout_rate=0.1,
            deterministic=not training,
            name="self_attn",
        )(x, x)
        dropped = nn.Dropout(0.1, deterministic=not training)(attended)
        tokens = nn.LayerNorm(name="norm")(x + dropped)  # (B, seq, E)

        # Per-token classification.
        hidden = nn.Dense(self.embed_dim, name="head_dense1")(tokens)
        hidden = nn.gelu(hidden)
        hidden = nn.Dropout(0.1, deterministic=not training)(hidden)
        logits_guess = nn.Dense(self.num_colours, name="head_dense2")(hidden)

        # Router representation: mean-pool.
        pooled = tokens.mean(axis=1)  # (B, E)

        return logits_guess, pooled
