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

import torch
import torch.nn as nn


_local_mask_cache: dict[tuple[int, int, int], torch.Tensor] = {}


def _local_attention_mask(
    grid_h: int, grid_w: int, window_size: int,
) -> torch.Tensor:
    """Build a 2-D local attention mask for a flattened (possibly rectangular) grid.

    Tokens may attend to neighbours within Chebyshev distance
    ``window_size // 2`` on the original grid.  Results are cached by
    ``(grid_h, grid_w, window_size)`` to avoid recomputation.

    :param grid_h: Height of the grid.
    :param grid_w: Width of the grid.
    :param window_size: Diameter of the local attention window.
    :return: Boolean mask ``(seq, seq)`` where *True* means the position
        is **blocked** from attending (PyTorch ``attn_mask`` convention).
    """
    key = (grid_h, grid_w, window_size)
    if key in _local_mask_cache:
        return _local_mask_cache[key]

    seq = grid_h * grid_w
    idx = torch.arange(seq)
    rows = idx // grid_w
    cols = idx % grid_w
    dist = torch.maximum(
        (rows[:, None] - rows[None, :]).abs(),
        (cols[:, None] - cols[None, :]).abs(),
    )
    radius = window_size // 2
    mask = dist > radius  # True = blocked
    _local_mask_cache[key] = mask
    return mask


class _FiLMConditioner(nn.Module):
    """Feature-wise Linear Modulation from a conditioning vector."""

    def __init__(self, embed_dim: int = 256) -> None:
        super().__init__()
        self.film_proj = nn.Linear(embed_dim, 2 * embed_dim)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """Apply FiLM: ``gamma * x + beta`` with gamma/beta derived from *h*.

        :param x: Token features ``(batch, seq, embed_dim)``.
        :param h: Conditioning vector ``(batch, embed_dim)``.
        :return: Modulated tokens ``(batch, seq, embed_dim)``.
        """
        params = self.film_proj(h)  # (B, 2E)
        gamma, beta = params.chunk(2, dim=-1)  # each (B, E)
        gamma = gamma + 1.0  # centred at identity
        return gamma.unsqueeze(1) * x + beta.unsqueeze(1)


class _GuessTransformerLayer(nn.Module):
    """Transformer layer with optional local-attention mask and FiLM."""

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 4,
        ffn_mult: int = 4,
        dropout_rate: float = 0.1,
    ) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout_rate, batch_first=True,
        )
        self.attn_dropout = nn.Dropout(dropout_rate)
        self.attn_norm = nn.LayerNorm(embed_dim)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * ffn_mult),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(embed_dim * ffn_mult, embed_dim),
            nn.Dropout(dropout_rate),
        )
        self.ffn_norm = nn.LayerNorm(embed_dim)

        self.film = _FiLMConditioner(embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        h: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        attended, _ = self.self_attn(x, x, x, attn_mask=mask)
        x = self.attn_norm(x + self.attn_dropout(attended))

        x = self.ffn_norm(x + self.ffn(x))

        x = self.film(x, h)
        return x


class GuessComponent(nn.Module):
    """Deep spatial predictor with alternating local/global attention
    and FiLM conditioning from the pooled task embedding.

    :param embed_dim: Dimensionality of the per-token embedding.
    :param num_colours: Number of per-cell colour classes (10 for ARC).
    :param num_heads: Number of self-attention heads.
    :param num_layers: Number of stacked Transformer layers.  Layers
        alternate local -> global -> local -> ...
    :param window_size: Diameter of the local attention window (in grid cells,
        Chebyshev distance).  Must be odd for a symmetric window.
    :param ffn_mult: FFN hidden-dim multiplier relative to ``embed_dim``.
    :param dropout_rate: Dropout probability used throughout.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_colours: int = 10,
        num_heads: int = 4,
        num_layers: int = 3,
        window_size: int = 7,
        ffn_mult: int = 4,
        dropout_rate: float = 0.1,
    ) -> None:
        super().__init__()
        self.window_size = window_size

        self.layers = nn.ModuleList([
            _GuessTransformerLayer(
                embed_dim=embed_dim,
                num_heads=num_heads,
                ffn_mult=ffn_mult,
                dropout_rate=dropout_rate,
            )
            for _ in range(num_layers)
        ])

        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(embed_dim, num_colours),
        )

    def forward(
        self,
        x: torch.Tensor,
        h: torch.Tensor,
        grid_h: int,
        grid_w: int,
        pad_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute per-token guess-pathway logits.

        :param x: Spatial token embeddings ``(batch, seq, embed_dim)``.
        :param h: Pooled task embedding ``(batch, embed_dim)`` for FiLM.
        :param grid_h: Height of the (padded) grid.
        :param grid_w: Width of the (padded) grid.
        :param pad_mask: Boolean mask ``(batch, seq)`` where *True* means
            the position is a valid (non-pad) cell.  Padded positions are
            blocked from attending or being attended to.
        :return: Tuple of ``(logits, pooled)`` where *logits* has shape
            ``(batch, seq, num_colours)`` and *pooled* is ``(batch, embed_dim)``
            for the router.
        """
        local_mask = _local_attention_mask(grid_h, grid_w, self.window_size).to(x.device)

        # Combine local window mask with padding mask so padded positions
        # are fully blocked.  nn.MultiheadAttention expects attn_mask as
        # float with -inf for blocked positions, shaped (B*H, seq, seq).
        num_heads = self.layers[0].self_attn.num_heads
        if pad_mask is not None:
            # pad_block: (B, 1, seq) float mask, 0.0 for valid, -inf for pad.
            pad_block = torch.zeros_like(pad_mask, dtype=x.dtype).unsqueeze(1)
            pad_block[~pad_mask.unsqueeze(1).expand_as(pad_block)] = float("-inf")
            # Expand (B, 1, seq) → (B, seq, seq) via broadcast in OR below.
            pad_block_3d = pad_block.expand(-1, pad_mask.size(1), -1)  # (B, seq, seq)

        for i, layer in enumerate(self.layers):
            if i % 2 == 0:
                # Convert bool local_mask to float: True (blocked) → -inf.
                float_local = torch.zeros_like(local_mask, dtype=x.dtype)
                float_local[local_mask] = float("-inf")
                if pad_mask is not None:
                    # Combine: element-wise min keeps the more-blocked value.
                    combined = float_local.unsqueeze(0) + pad_block_3d  # broadcast (B, seq, seq)
                    combined = combined.clamp(min=float("-inf"))
                    # Expand to (B*H, seq, seq) for MHA.
                    mask = combined.unsqueeze(1).expand(-1, num_heads, -1, -1).reshape(-1, combined.size(1), combined.size(2))
                else:
                    mask = float_local
            else:
                if pad_mask is not None:
                    mask = pad_block_3d.unsqueeze(1).expand(-1, num_heads, -1, -1).reshape(-1, pad_block_3d.size(1), pad_block_3d.size(2))
                else:
                    mask = None
            x = layer(x, h, mask=mask)

        logits_guess: torch.Tensor = self.head(x)  # (B, seq, num_colours)
        pooled = x.mean(dim=1)  # (B, E)

        return logits_guess, pooled
