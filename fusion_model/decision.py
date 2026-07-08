"""DecisionRouter — multi-head cross-attention router over the four pathways.

The router is the *arbiter* of the Fusion Model.  Rather than concatenating
summary signals into a flat vector and passing them through an MLP, the
router uses **multi-head cross-attention**: the shared embedding ``h``
serves as the *query*, and each expert pathway provides a *key/value*
token summarising what it can offer for the current input.

Each attention head operates in its own learned subspace, allowing the
router to evaluate the pathways along multiple independent criteria
simultaneously.  A **residual connection** adds the original shared
embedding ``h`` back to the attention context, followed by **LayerNorm**,
and a **two-layer MLP** maps the result to one routing logit per pathway.

**Grounded routing via demo fit.**  When the model runs leave-one-out
verification (predicting a held-out demonstration's output with each
pathway), the measured per-pathway fit — the centred negative
cross-entropy on the held-out demo — is added to the routing logits
through a learned scale.  This gives the router an *objective* signal
("pathway 2 actually reproduced the held-out demo") on top of the learned
one, instead of having to infer pathway quality purely from representation
vectors:

    ``alpha = softmax((MLP(LayerNorm(context + h)) + fit_scale * fit) / temperature)``

**Pathway masking.**  The proposal pathway only activates once the rule
generator's history buffer has enough entries.  While inactive, its
routing logit is masked to ``-inf`` so its mixture weight is exactly zero.

A learnable temperature controls the sharpness of the routing
distribution; the entropy regulariser in the loss interacts naturally with
it.  The raw per-head attention weights are returned for interpretability.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

#: Index of each pathway in the routing weight vector ``alpha``.
PATHWAY_NAMES: tuple[str, ...] = ("mem", "rule", "prop", "guess")
NUM_PATHWAYS: int = len(PATHWAY_NAMES)
PROP_INDEX: int = PATHWAY_NAMES.index("prop")
GUESS_INDEX: int = PATHWAY_NAMES.index("guess")


class DecisionRouter(nn.Module):
    """Multi-head cross-attention router producing softmax mixture weights
    over the four expert pathways.

    The shared embedding ``h`` is used as the attention query, while each
    pathway's intermediate representation serves as both key and value.
    ``nn.MultiheadAttention`` computes scaled-dot-product attention across
    ``num_heads`` independent subspaces, producing a context vector that
    captures *what* information the router extracted from the pathways —
    not just *which* pathway was most similar.

    :param embed_dim: Dimensionality of the shared input embedding and of
        each pathway's intermediate representation.
    :param num_heads: Number of attention heads.  Each head evaluates the
        pathway tokens in its own subspace.  Values of 2–4 work well given
        only four key/value tokens.
    """

    def __init__(self, embed_dim: int = 256, num_heads: int = 4) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.mha = nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=True,
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.alpha_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, NUM_PATHWAYS),
        )
        self.temperature = nn.Parameter(torch.tensor(1.0))
        # Learned scale on the measured demo-fit signal.  Initialised at
        # 1.0 so the objective signal matters from the start; the model
        # can amplify or attenuate it during training.
        self.fit_scale = nn.Parameter(torch.tensor(1.0))

    def forward(
        self,
        h: torch.Tensor,
        pathway_reprs: list[torch.Tensor],
        fit: torch.Tensor | None = None,
        prop_active: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute routing weights via multi-head cross-attention.

        :param h: Shared embedding ``(batch, embed_dim)`` — used as the
            attention query and as the residual.
        :param pathway_reprs: List of ``NUM_PATHWAYS`` pathway
            representations, each ``(batch, embed_dim)``, in
            :data:`PATHWAY_NAMES` order (mem, rule, prop, guess).
        :param fit: Optional measured demo-fit signal
            ``(batch, NUM_PATHWAYS)`` — centred negative CE of each
            pathway on a held-out demonstration.  Added to the routing
            logits through a learned scale.
        :param prop_active: Whether the proposal pathway produced logits
            this step.  When False its routing weight is forced to zero.
        :return: Tuple of ``(alpha, attn_weights)`` where *alpha* has shape
            ``(batch, NUM_PATHWAYS)`` and *attn_weights* has shape
            ``(batch, num_heads, NUM_PATHWAYS)`` containing the raw
            per-head attention distributions over the pathway tokens.
        """
        assert len(pathway_reprs) == NUM_PATHWAYS, (
            f"expected {NUM_PATHWAYS} pathway representations, "
            f"got {len(pathway_reprs)}"
        )
        pathway_tokens = torch.stack(pathway_reprs, dim=1)  # (B, P, E)

        context, attn_weights = self.mha(
            query=h.unsqueeze(1),
            key=pathway_tokens,
            value=pathway_tokens,
            average_attn_weights=False,
        )  # context: (B, 1, E), attn_weights: (B, num_heads, 1, P)

        fused = self.norm(context.squeeze(1) + h)  # (B, E)

        logits = self.alpha_mlp(fused)  # (B, P)
        if fit is not None:
            logits = logits + self.fit_scale * fit

        temp = self.temperature.clamp(min=0.01)
        scaled = logits / temp
        if not prop_active:
            # Mask the inactive proposal pathway out of the mixture.
            # masked_fill *after* the temperature division: an -inf fed
            # through the division would make the temperature gradient
            # NaN (-inf · 0) in the backward pass.
            mask = torch.zeros_like(scaled, dtype=torch.bool)
            mask[:, PROP_INDEX] = True
            scaled = scaled.masked_fill(mask, float("-inf"))
        alpha: torch.Tensor = F.softmax(scaled, dim=-1)  # (B, P)

        head_weights = attn_weights.squeeze(2)  # (B, num_heads, P)

        return alpha, head_weights
