"""DecisionRouter — learns to mix the three pathways (memory, rule, guess).

The router is the *arbiter* of the Fusion Model.  Given the shared embedding
plus summary signals from the memory and rule-generator pathways, it outputs
a three-element probability vector:

    ``alpha = [alpha_mem, alpha_rule, alpha_guess]``

These weights are applied to the logits of each pathway so that the final
prediction is a soft mixture:

    ``p(y|x) = alpha_mem * p_mem + alpha_rule * p_rule + alpha_guess * p_guess``

Early in training the entropy regulariser in the loss keeps the weights
close to uniform so that every pathway gets gradient signal.  As the model
converges the router learns to specialise — ideally pushing structured /
rule-amenable questions towards the rule pathways and fuzzy questions
towards the guess pathway.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DecisionRouter(nn.Module):
    """Two-layer MLP that produces softmax mixture weights over three pathways.

    :param embed_dim: Dimensionality of the shared input embedding.
    :param hidden_dim: Width of the router's hidden layer.
    """

    def __init__(self, embed_dim: int = 256, hidden_dim: int = 128) -> None:
        super().__init__()
        # The router sees three things concatenated:
        #   1. The shared embedding  h            (embed_dim)
        #   2. The top retrieved memory key       (embed_dim)
        #   3. The rule-generator confidence      (1)
        input_dim = embed_dim + embed_dim + 1
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3),  # one logit per pathway
        )

    def forward(
        self,
        h: torch.Tensor,
        retrieval_top_key: torch.Tensor,
        rule_confidence: torch.Tensor,
    ) -> torch.Tensor:
        """Compute routing weights.

        :param h: Shared embedding ``(batch, embed_dim)``.
        :param retrieval_top_key: Key of the best-matching memory rule
            ``(batch, embed_dim)``.
        :param rule_confidence: Generator confidence ``(batch, 1)``.
        :return: Softmax weights ``(batch, 3)`` —
            ``[alpha_mem, alpha_rule, alpha_guess]``.
        """
        x = torch.cat([h, retrieval_top_key, rule_confidence], dim=-1)
        alpha: torch.Tensor = F.softmax(self.mlp(x), dim=-1)
        return alpha
