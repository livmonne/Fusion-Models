"""DecisionRouter — multi-head cross-attention router that mixes the three pathways.

The router is the *arbiter* of the Fusion Model.  Rather than concatenating
summary signals into a flat vector and passing them through an MLP, the
router uses **multi-head cross-attention**: the shared embedding ``h``
serves as the *query*, and each expert pathway provides a *key/value*
token summarising what it can offer for the current input.

Each attention head operates in its own learned subspace, allowing the
router to evaluate the pathways along multiple independent criteria
simultaneously — e.g. one head might focus on semantic relevance while
another tracks confidence signals.  The multi-head attention produces a
context vector that is a rich, value-weighted blend of the pathway
representations, which is then projected to three routing logits:

    ``alpha = softmax(W_alpha · context / temperature)``

so that the final prediction is a soft mixture:

    ``p(y|x) = alpha_mem * p_mem + alpha_rule * p_rule + alpha_guess * p_guess``

Because the routing decision passes through both the attention mechanism
*and* the value/projection layers, the router can learn relationships
richer than simple dot-product similarity — each pathway's key controls
*when* to attract attention, while its value controls *what information*
to communicate to the routing decision.

A learnable temperature parameter controls the sharpness of the routing
distribution: low temperature → peaky (hard routing), high temperature →
uniform (soft routing).  The entropy regulariser in the loss still applies
and interacts naturally with this temperature.

The raw per-head attention weights are returned alongside the routing
coefficients for interpretability and debugging.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DecisionRouter(nn.Module):
    """Multi-head cross-attention router that produces softmax mixture weights
    over three expert pathways.

    The shared embedding ``h`` is used as the attention query, while each
    pathway's intermediate representation serves as both key and value.
    ``nn.MultiheadAttention`` computes scaled-dot-product attention across
    ``num_heads`` independent subspaces, producing a context vector that
    captures *what* information the router extracted from the pathways —
    not just *which* pathway was most similar.  A final linear projection
    maps this context to three routing logits.

    :param embed_dim: Dimensionality of the shared input embedding and of
        each pathway's intermediate representation.
    :param num_heads: Number of attention heads.  Each head evaluates the
        three pathway tokens in its own subspace, enabling the router to
        weigh multiple criteria (relevance, confidence, complementarity)
        in parallel.  Values of 2–4 work well given only 3 key/value
        tokens.
    """

    def __init__(self, embed_dim: int = 256, num_heads: int = 4) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.mha = nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=True,
        )
        self.alpha_proj = nn.Linear(embed_dim, 3)
        self.temperature = nn.Parameter(torch.tensor(1.0))

    def forward(
        self,
        h: torch.Tensor,
        mem_repr: torch.Tensor,
        rule_repr: torch.Tensor,
        guess_repr: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute routing weights via multi-head cross-attention.

        :param h: Shared embedding ``(batch, embed_dim)`` — used as the
            attention query.
        :param mem_repr: Memory pathway intermediate representation
            ``(batch, embed_dim)`` — the blended correction vector from
            :class:`~fusion_model.memory.RuleMemory`.
        :param rule_repr: Rule pathway intermediate representation
            ``(batch, embed_dim)`` — the ephemeral correction vector from
            :class:`~fusion_model.rule_engine.RuleGenerator`.
        :param guess_repr: Guess pathway intermediate representation
            ``(batch, embed_dim)`` — the pooled self-attention output from
            :class:`~fusion_model.guess.GuessComponent`.
        :return: Tuple of ``(alpha, attn_weights)`` where *alpha* has shape
            ``(batch, 3)`` — ``[alpha_mem, alpha_rule, alpha_guess]`` — and
            *attn_weights* has shape ``(batch, num_heads, 3)`` containing
            the raw per-head attention distributions over the three
            pathway tokens (useful for interpretability/debugging).
        """
        pathway_tokens = torch.stack(
            [mem_repr, rule_repr, guess_repr],
            dim=1,
        )  # (batch, 3, embed_dim)

        context, attn_weights = self.mha(
            query=h.unsqueeze(1),
            key=pathway_tokens,
            value=pathway_tokens,
            average_attn_weights=False,
        )  # context: (batch, 1, embed_dim), attn_weights: (batch, num_heads, 1, 3)

        temp = self.temperature.clamp(min=0.01)
        alpha: torch.Tensor = F.softmax(
            self.alpha_proj(context.squeeze(1)) / temp,
            dim=-1,
        )  # (batch, 3)

        head_weights = attn_weights.squeeze(2)  # (batch, num_heads, 3)

        return alpha, head_weights
