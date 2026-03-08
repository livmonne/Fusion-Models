"""DecisionRouter — cross-attention router that mixes the three pathways.

The router is the *arbiter* of the Fusion Model.  Rather than concatenating
summary signals into a flat vector and passing them through an MLP, the
router uses **cross-attention**: the shared embedding ``h`` serves as the
*query*, and each expert pathway provides a *key/value* token summarising
what it can offer for the current input.

The attention weights over the three pathway tokens directly yield the
mixture coefficients:

    ``alpha = softmax(q · K^T / sqrt(d))``

so that the final prediction is a soft mixture:

    ``p(y|x) = alpha_mem * p_mem + alpha_rule * p_rule + alpha_guess * p_guess``

Because attention is inherently input-dependent, the router can learn
*dynamic* relationships between the fused context and each expert's
intermediate representation — something a static MLP over concatenated
features cannot do.

A learnable temperature parameter controls the sharpness of the routing
distribution: low temperature → peaky (hard routing), high temperature →
uniform (soft routing).  The entropy regulariser in the loss still applies
and interacts naturally with this temperature.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DecisionRouter(nn.Module):
    """Cross-attention router that produces softmax mixture weights over three
    expert pathways.

    The shared embedding ``h`` is projected into a query vector, while each
    pathway's intermediate representation is projected into key/value space.
    A single-head scaled-dot-product attention over the three pathway tokens
    yields the routing weights directly.

    :param embed_dim: Dimensionality of the shared input embedding and of
        each pathway's intermediate representation.
    :param num_heads: Number of attention heads.  Defaults to 1 so that the
        attention weights map cleanly onto the three routing coefficients.
    """

    def __init__(self, embed_dim: int = 256, num_heads: int = 1) -> None:
        super().__init__()
        self.embed_dim = embed_dim

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)

        self.temperature = nn.Parameter(torch.tensor(1.0))

    def forward(
        self,
        h: torch.Tensor,
        mem_repr: torch.Tensor,
        rule_repr: torch.Tensor,
        guess_repr: torch.Tensor,
    ) -> torch.Tensor:
        """Compute routing weights via cross-attention.

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
        :return: Softmax weights ``(batch, 3)`` —
            ``[alpha_mem, alpha_rule, alpha_guess]``.
        """
        query = self.q_proj(h)  # (batch, embed_dim)

        pathway_tokens = torch.stack(
            [mem_repr, rule_repr, guess_repr], dim=1,
        )  # (batch, 3, embed_dim)
        keys = self.k_proj(pathway_tokens)  # (batch, 3, embed_dim)

        scale = self.embed_dim ** 0.5
        attn_logits = torch.bmm(
            query.unsqueeze(1), keys.transpose(1, 2),
        ).squeeze(1)  # (batch, 3)

        temp = self.temperature.clamp(min=0.01)
        alpha: torch.Tensor = F.softmax(attn_logits / (scale * temp), dim=-1)
        return alpha
