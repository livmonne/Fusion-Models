"""GuessComponent — deep spatial predictor with local attention and FiLM.

This is the "pattern matching" pathway.  It handles fuzzy, hard-to-formalise
patterns that cannot be captured by crisp rules — for example, recognising
that a transformation involves a holistic spatial rearrangement rather than
a mechanical cell-by-cell procedure.

**Architecture:**

1. The spatial token sequence ``x`` ``(batch, seq, embed_dim)`` is processed
   by a stack of Transformer layers that **alternate** between *local*
   (windowed) and *global* self-attention.  Local layers restrict each
   token's receptive field to a Chebyshev-distance neighbourhood on the
   original 2-D grid, encouraging fine-grained spatial pattern detection.
   Global layers allow unrestricted attention for long-range integration.
2. Every layer contains a proper **feed-forward network** (FFN) with 4×
   expansion and GELU activation, following standard Transformer design.
3. After each layer, **FiLM conditioning** (Feature-wise Linear Modulation)
   injects global task context from the pooled embedding ``h``, allowing
   per-token representations to be modulated by task-level information
   that the other two pathways also receive.
4. A per-token **MLP head** maps the final tokens to ``num_colours`` logits.
5. The output tokens are **mean-pooled** into a single representation
   vector for the router.
"""

from __future__ import annotations

import flax.linen as nn
import jax.numpy as jnp


def _local_attention_mask(grid_size: int, window_size: int) -> jnp.ndarray:
    """Build a 2-D local attention mask for a flattened square grid.

    Tokens may attend to neighbours within Chebyshev distance
    ``window_size // 2`` on the original grid.

    :param grid_size: Side length of the (square, padded) grid.
    :param window_size: Diameter of the local attention window.
    :return: Boolean mask ``(1, 1, seq, seq)`` broadcastable over batch
        and head dimensions, where *True* = allow attention.
    """
    seq = grid_size * grid_size
    idx = jnp.arange(seq)
    rows = idx // grid_size
    cols = idx % grid_size
    dist = jnp.maximum(
        jnp.abs(rows[:, None] - rows[None, :]),
        jnp.abs(cols[:, None] - cols[None, :]),
    )
    radius = window_size // 2
    mask = dist <= radius
    return mask[None, None, :, :]  # (1, 1, seq, seq)


class _FiLMConditioner(nn.Module):
    """Feature-wise Linear Modulation from a conditioning vector."""

    embed_dim: int = 256

    @nn.compact
    def __call__(self, x: jnp.ndarray, h: jnp.ndarray) -> jnp.ndarray:
        """Apply FiLM: ``gamma * x + beta`` with gamma/beta derived from *h*.

        :param x: Token features ``(batch, seq, embed_dim)``.
        :param h: Conditioning vector ``(batch, embed_dim)``.
        :return: Modulated tokens ``(batch, seq, embed_dim)``.
        """
        params = nn.Dense(2 * self.embed_dim, name="film_proj")(h)  # (B, 2E)
        gamma = params[:, : self.embed_dim] + 1.0  # centred at identity
        beta = params[:, self.embed_dim :]
        return gamma[:, None, :] * x + beta[:, None, :]


class _GuessTransformerLayer(nn.Module):
    """Transformer layer with optional local-attention mask and FiLM."""

    embed_dim: int = 256
    num_heads: int = 4
    ffn_mult: int = 4
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        h: jnp.ndarray,
        *,
        mask: jnp.ndarray | None = None,
        training: bool = False,
    ) -> jnp.ndarray:
        # Self-attention (local when *mask* is supplied, global otherwise).
        attended = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.embed_dim,
            dropout_rate=self.dropout_rate,
            deterministic=not training,
            name="self_attn",
        )(x, x, mask=mask)
        attended = nn.Dropout(
            self.dropout_rate,
            deterministic=not training,
        )(attended)
        x = nn.LayerNorm(name="attn_norm")(x + attended)

        # Feed-forward network.
        ffn = nn.Dense(self.embed_dim * self.ffn_mult, name="ffn_dense1")(x)
        ffn = nn.gelu(ffn)
        ffn = nn.Dropout(
            self.dropout_rate,
            deterministic=not training,
        )(ffn)
        ffn = nn.Dense(self.embed_dim, name="ffn_dense2")(ffn)
        ffn = nn.Dropout(
            self.dropout_rate,
            deterministic=not training,
        )(ffn)
        x = nn.LayerNorm(name="ffn_norm")(x + ffn)

        # FiLM conditioning from pooled task embedding.
        x = _FiLMConditioner(embed_dim=self.embed_dim, name="film")(x, h)
        return x


class GuessComponent(nn.Module):
    """Deep spatial predictor with alternating local/global attention
    and FiLM conditioning from the pooled task embedding.

    :param embed_dim: Dimensionality of the per-token embedding.
    :param num_colours: Number of per-cell colour classes (10 for ARC).
    :param num_heads: Number of self-attention heads.
    :param num_layers: Number of stacked Transformer layers.  Layers
        alternate local → global → local → …
    :param window_size: Diameter of the local attention window (in grid
        cells, Chebyshev distance).  Must be odd for a symmetric window.
    :param ffn_mult: FFN hidden-dim multiplier relative to ``embed_dim``.
    :param dropout_rate: Dropout probability used throughout.
    """

    embed_dim: int = 256
    num_colours: int = 10
    num_heads: int = 4
    num_layers: int = 3
    window_size: int = 7
    ffn_mult: int = 4
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        h: jnp.ndarray,
        grid_size: int,
        *,
        training: bool = False,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Compute per-token guess-pathway logits.

        :param x: Spatial token embeddings ``(batch, seq, embed_dim)``.
        :param h: Pooled task embedding ``(batch, embed_dim)`` for FiLM.
        :param grid_size: Side length of the padded square grid.
        :param training: Whether we are in training mode.
        :return: Tuple of ``(logits, pooled)`` where *logits* has shape
            ``(batch, seq, num_colours)`` and *pooled* is ``(batch, embed_dim)``
            for the router.
        """
        local_mask = _local_attention_mask(grid_size, self.window_size)

        for i in range(self.num_layers):
            # Even layers → local attention; odd layers → global.
            mask = local_mask if i % 2 == 0 else None
            x = _GuessTransformerLayer(
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                ffn_mult=self.ffn_mult,
                dropout_rate=self.dropout_rate,
                name=f"layer_{i}",
            )(x, h, mask=mask, training=training)

        # Per-token classification head.
        hidden = nn.Dense(self.embed_dim, name="head_dense1")(x)
        hidden = nn.gelu(hidden)
        hidden = nn.Dropout(
            self.dropout_rate,
            deterministic=not training,
        )(hidden)
        logits_guess = nn.Dense(self.num_colours, name="head_dense2")(hidden)

        # Router representation: mean-pool.
        pooled = x.mean(axis=1)  # (B, E)

        return logits_guess, pooled
