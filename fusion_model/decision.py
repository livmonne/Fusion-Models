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
representations.

A **residual connection** adds the original shared embedding ``h`` back
to the attention context, followed by **LayerNorm**, ensuring that the
routing MLP always has direct access to the raw input alongside the
pathway-informed context.  This mirrors standard transformer practice
and provides a clean gradient path from the routing decision back to
the upstream encoders.

The normalised residual is then mapped to three routing logits via a
**two-layer MLP** (Linear → GELU → Linear) rather than a single linear
projection, giving the router capacity to learn nonlinear feature
interactions (e.g. "memory confidence is high *and* the question is
about counting"):

    ``alpha = softmax(MLP(LayerNorm(context + h)) / temperature)``

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

import jax
import jax.numpy as jnp
import flax.linen as nn


class DecisionRouter(nn.Module):
    """Multi-head cross-attention router that produces softmax mixture weights
    over three expert pathways.

    :param embed_dim: Dimensionality of the shared input embedding and of
        each pathway's intermediate representation.
    :param num_heads: Number of attention heads.
    """

    embed_dim: int = 256
    num_heads: int = 4

    @nn.compact
    def __call__(
        self,
        h: jnp.ndarray,
        mem_repr: jnp.ndarray,
        rule_repr: jnp.ndarray,
        guess_repr: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Compute routing weights via multi-head cross-attention.

        :param h: Shared embedding ``(batch, embed_dim)``.
        :param mem_repr: Memory pathway representation ``(batch, embed_dim)``.
        :param rule_repr: Rule pathway representation ``(batch, embed_dim)``.
        :param guess_repr: Guess pathway representation ``(batch, embed_dim)``.
        :return: Tuple of ``(alpha, attn_weights)`` where *alpha* has shape
            ``(batch, 3)`` and *attn_weights* has shape
            ``(batch, num_heads, 3)``.
        """
        pathway_tokens = jnp.stack(
            [mem_repr, rule_repr, guess_repr],
            axis=1,
        )  # (batch, 3, embed_dim)

        # Flax MHA expects query (B, T_q, E), key/value (B, T_kv, E).
        query = h[:, None, :]  # (B, 1, E)

        # We need per-head attention weights — use a custom attention call.
        mha = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.embed_dim,
            name="mha",
        )
        context = mha(query, pathway_tokens)  # (B, 1, E)

        fused = nn.LayerNorm(name="norm")(context.squeeze(1) + h)  # (B, E)

        temperature = self.param(
            "temperature", lambda _rng, _shape: jnp.array(1.0), (),
        )
        temp = jnp.clip(temperature, min=0.01)

        # Two-layer MLP for routing logits.
        hidden = nn.Dense(self.embed_dim, name="alpha_mlp_0")(fused)
        hidden = nn.gelu(hidden)
        routing_logits = nn.Dense(3, name="alpha_mlp_1")(hidden)

        alpha = jax.nn.softmax(routing_logits / temp, axis=-1)  # (B, 3)

        # Re-compute attention weights for interpretability.
        # We compute them manually since Flax MHA doesn't return them by default.
        head_dim = self.embed_dim // self.num_heads
        q_proj = nn.DenseGeneral((self.num_heads, head_dim), name="attn_q")(h[:, None, :])
        k_proj = nn.DenseGeneral((self.num_heads, head_dim), name="attn_k")(pathway_tokens)
        # q_proj: (B, 1, num_heads, head_dim), k_proj: (B, 3, num_heads, head_dim)
        attn_logits = jnp.einsum("bqhd, bkhd -> bhqk", q_proj, k_proj) / (head_dim ** 0.5)
        head_weights = jax.nn.softmax(attn_logits, axis=-1).squeeze(2)  # (B, num_heads, 3)

        return alpha, head_weights
