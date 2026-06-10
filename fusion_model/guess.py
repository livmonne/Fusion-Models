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
2. Padding cells are **masked out of attention** (combined with the local
   window mask).  Every token keeps its self-connection so no attention row
   is ever fully masked, which would produce NaNs.
3. Every layer contains a proper **feed-forward network** (FFN) with 4×
   expansion and GELU activation, following standard Transformer design.
4. After each layer, **FiLM conditioning** (Feature-wise Linear Modulation)
   injects global task context from the pooled task embedding ``t``,
   so the guess pathway knows *what kind of task* it's working on.
5. A per-token **MLP head** maps the final tokens to ``num_colours`` logits.
6. The output tokens are **masked mean-pooled** into a single representation
   vector for the router.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .common import masked_mean


def _local_attention_mask(
    grid_h: int, grid_w: int, window_size: int,
) -> torch.Tensor:
    """Build a 2-D local attention mask for a flattened (possibly rectangular) grid.

    Tokens may attend to neighbours within Chebyshev distance
    ``window_size // 2`` on the original grid.

    :param grid_h: Height of the grid.
    :param grid_w: Width of the grid.
    :param window_size: Diameter of the local attention window.
    :return: Boolean mask ``(seq, seq)`` where *True* means the position
        is **blocked** from attending (PyTorch ``attn_mask`` convention).
    """
    seq = grid_h * grid_w
    idx = torch.arange(seq)
    rows = idx // grid_w
    cols = idx % grid_w
    dist = torch.maximum(
        (rows[:, None] - rows[None, :]).abs(),
        (cols[:, None] - cols[None, :]).abs(),
    )
    radius = window_size // 2
    return dist > radius  # True = blocked


def _combine_attn_masks(
    local_blocked: torch.Tensor | None,
    key_invalid: torch.Tensor | None,
    num_heads: int,
) -> torch.Tensor | None:
    """Merge a local-window mask with a per-sample key-padding mask.

    The diagonal is always unblocked so every query token (including
    padding tokens) can attend to itself — a fully-masked attention row
    produces NaNs in softmax.

    :param local_blocked: ``(seq, seq)`` boolean mask (True = blocked) or
        ``None`` for global attention.
    :param key_invalid: ``(batch, seq)`` boolean mask (True = padding key)
        or ``None`` when there is no padding.
    :param num_heads: Attention head count; the result is expanded to
        ``(batch * num_heads, seq, seq)`` as required by
        :class:`torch.nn.MultiheadAttention`.
    :return: Combined boolean mask or ``None`` if nothing is masked.
    """
    if local_blocked is None and key_invalid is None:
        return None

    if key_invalid is None:
        # Static (seq, seq) mask is accepted directly by MultiheadAttention.
        return local_blocked

    B, L = key_invalid.shape
    blocked = key_invalid.unsqueeze(1).expand(B, L, L).clone()  # block PAD keys
    if local_blocked is not None:
        blocked = blocked | local_blocked.unsqueeze(0)
    diag = torch.arange(L, device=key_invalid.device)
    blocked[:, diag, diag] = False
    return blocked.repeat_interleave(num_heads, dim=0)  # (B*H, L, L)


class _FiLMConditioner(nn.Module):
    """Feature-wise Linear Modulation from a conditioning vector."""

    def __init__(self, embed_dim: int = 256) -> None:
        super().__init__()
        self.film_proj = nn.Linear(embed_dim, 2 * embed_dim)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Apply FiLM: ``gamma * x + beta`` with gamma/beta derived from *t*.

        :param x: Token features ``(batch, seq, embed_dim)``.
        :param t: Conditioning vector ``(batch, embed_dim)``.
        :return: Modulated tokens ``(batch, seq, embed_dim)``.
        """
        params: torch.Tensor = self.film_proj(t)  # (B, 2E)
        gamma, beta = params.chunk(2, dim=-1)  # each (B, E)
        gamma = gamma + 1.0  # centred at identity
        return gamma.unsqueeze(1) * x + beta.unsqueeze(1)


class _GuessTransformerLayer(nn.Module):
    """Transformer layer with optional attention mask and FiLM."""

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
        t: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # need_weights=False keeps the fused SDPA path (no O(L²·H) maps).
        attended, _ = self.self_attn(x, x, x, attn_mask=mask, need_weights=False)
        x = self.attn_norm(x + self.attn_dropout(attended))

        x = self.ffn_norm(x + self.ffn(x))

        x = self.film(x, t)
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
        self.num_heads = num_heads

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
        t: torch.Tensor,
        grid_h: int,
        grid_w: int,
        pad_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute per-token guess-pathway logits.

        :param x: Spatial token embeddings ``(batch, seq, embed_dim)``.
        :param t: Pooled task embedding ``(batch, embed_dim)`` for FiLM.
        :param grid_h: Height of the (padded) grid.
        :param grid_w: Width of the (padded) grid.
        :param pad_mask: Boolean validity mask ``(batch, seq)``; True for
            real cells.  Padding cells are excluded from attention keys
            and from the pooled router representation.
        :return: Tuple of ``(logits, pooled)`` where *logits* has shape
            ``(batch, seq, num_colours)`` and *pooled* is ``(batch, embed_dim)``
            for the router.
        """
        local_blocked = _local_attention_mask(
            grid_h, grid_w, self.window_size,
        ).to(x.device)
        key_invalid = None if pad_mask is None else ~pad_mask

        local_mask = _combine_attn_masks(local_blocked, key_invalid, self.num_heads)
        global_mask = _combine_attn_masks(None, key_invalid, self.num_heads)

        for i, layer in enumerate(self.layers):
            mask = local_mask if i % 2 == 0 else global_mask
            x = layer(x, t, mask=mask)

        logits_guess: torch.Tensor = self.head(x)  # (B, seq, num_colours)
        pooled = masked_mean(x, pad_mask)  # (B, E)

        return logits_guess, pooled
