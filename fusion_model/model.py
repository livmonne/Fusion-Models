"""FusionModel — orchestrator that wires all components together.

The Fusion Model is an architecture for abstract reasoning on grid
transformation tasks (ARC-AGI-2).  It accepts a set of **demonstration
input/output grid pairs** and a **test input grid**, and produces a
predicted output grid by inferring the transformation rule from the demos.

**Forward pass overview:**

1. **Grid cell embedding** — each cell value (0–9) is mapped to a learned
   embedding vector.  2-D sinusoidal positional encodings are added so the
   model knows where each cell sits in the grid.
2. **Demo pair encoding** — for each demonstration pair the input and
   output cell embeddings are concatenated along the sequence dimension
   and processed by a shared Transformer encoder (self-attention blocks).
   The resulting token sequences are concatenated across all demos into a
   single *demo context* sequence.
3. **Multi-head cross-attention** — the test input cell embeddings
   cross-attend to the demo context via multiple attention heads.
4. **Spatial tokens + pooled embedding** — the cross-attended test tokens
   ``x`` of shape ``(batch, seq, embed_dim)`` carry per-cell spatial
   information.  A mean-pooled vector ``h`` summarises the task globally.
5. **Three expert pathways** each receive the full spatial sequence ``x``
   (and pooled ``h`` where needed) and return **per-cell colour logits**
   ``(batch, seq, num_colours)`` plus a pooled representation for the
   router:
   - :class:`~fusion_model.memory.RuleMemory`
   - :class:`~fusion_model.rule_engine.RuleGenerator`
   - :class:`~fusion_model.guess.GuessComponent`
6. **DecisionRouter** — produces softmax mixture weights ``alpha`` over
   the three pathways.
7. **Output** — the blended per-cell logits ``(batch, seq, num_colours)``.
8. **Rule commitment** and **history update** proceed as before.
"""

from __future__ import annotations

import math
from typing import Any

import jax
import jax.numpy as jnp
import flax.linen as nn

from .decision import DecisionRouter
from .guess import GuessComponent
from .memory import RuleMemory
from .rule_engine import RuleGenerator


def _sinusoidal_pos_encoding_2d(
    max_h: int, max_w: int, embed_dim: int
) -> jnp.ndarray:
    """Generate 2-D sinusoidal positional encodings.

    Returns an array of shape ``(max_h * max_w, embed_dim)`` where the
    first half of channels encode the row position and the second half
    encode the column position.
    """
    half = embed_dim // 2
    pos_h = jnp.arange(max_h, dtype=jnp.float32)[:, None]  # (H, 1)
    pos_w = jnp.arange(max_w, dtype=jnp.float32)[:, None]  # (W, 1)
    div = jnp.exp(jnp.arange(0, half, 2, dtype=jnp.float32) * -(math.log(10000.0) / half))

    pe_h = jnp.zeros((max_h, half))
    pe_h = pe_h.at[:, 0::2].set(jnp.sin(pos_h * div[: half // 2 + (half % 2)]))
    pe_h = pe_h.at[:, 1::2].set(jnp.cos(pos_h * div[: half // 2]))

    pe_w = jnp.zeros((max_w, half))
    pe_w = pe_w.at[:, 0::2].set(jnp.sin(pos_w * div[: half // 2 + (half % 2)]))
    pe_w = pe_w.at[:, 1::2].set(jnp.cos(pos_w * div[: half // 2]))

    # Broadcast: (H, 1, half) + (1, W, half) → (H, W, half)
    pe = jnp.concatenate(
        [jnp.broadcast_to(pe_h[:, None, :], (max_h, max_w, half)),
         jnp.broadcast_to(pe_w[None, :, :], (max_h, max_w, half))],
        axis=-1,
    )  # (H, W, embed_dim)
    return pe.reshape(max_h * max_w, embed_dim)


class TransformerEncoderLayer(nn.Module):
    """Single Transformer encoder layer with self-attention + FFN."""
    embed_dim: int = 256
    num_heads: int = 8
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x: jnp.ndarray, *, training: bool = False) -> jnp.ndarray:
        # Self-attention.
        attended = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.embed_dim,
            dropout_rate=self.dropout_rate,
            deterministic=not training,
        )(x, x)
        x = nn.LayerNorm()(x + nn.Dropout(self.dropout_rate, deterministic=not training)(attended))

        # Feed-forward.
        ffn = nn.Dense(self.embed_dim * 4)(x)
        ffn = nn.gelu(ffn)
        ffn = nn.Dropout(self.dropout_rate, deterministic=not training)(ffn)
        ffn = nn.Dense(self.embed_dim)(ffn)
        ffn = nn.Dropout(self.dropout_rate, deterministic=not training)(ffn)
        x = nn.LayerNorm()(x + ffn)
        return x


class CrossAttentionLayer(nn.Module):
    """Cross-attention layer: query attends to key/value context."""
    embed_dim: int = 256
    num_heads: int = 8
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(
        self, x: jnp.ndarray, context: jnp.ndarray, *, training: bool = False,
    ) -> jnp.ndarray:
        attended = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.embed_dim,
            dropout_rate=self.dropout_rate,
            deterministic=not training,
        )(x, context)
        x = nn.LayerNorm()(x + attended)

        # FFN.
        ffn = nn.Dense(self.embed_dim * 4)(x)
        ffn = nn.gelu(ffn)
        ffn = nn.Dropout(self.dropout_rate, deterministic=not training)(ffn)
        ffn = nn.Dense(self.embed_dim)(ffn)
        ffn = nn.Dropout(self.dropout_rate, deterministic=not training)(ffn)
        x = nn.LayerNorm()(x + ffn)
        return x


class FusionModel(nn.Module):
    """End-to-end Fusion Model for ARC-AGI-2 grid transformation tasks.

    :param embed_dim: Internal embedding dimensionality shared by all
        components.
    :param num_colours: Number of distinct cell values (10 for ARC).
    :param max_grid_size: Maximum grid dimension (30 for ARC).
    :param num_encoder_layers: Number of Transformer encoder layers for
        processing demo pairs.
    :param num_cross_attn_layers: Number of cross-attention layers for
        transferring demo context to the test input.
    :param num_attn_heads: Number of attention heads in all multi-head
        attention layers.
    :param num_rule_slots: Number of rule slots in the memory bank.
    :param rule_rank: Low-rank dimension used by memory and generator.
    :param history_size: Capacity of the RuleGenerator's circular history
        buffer.
    """

    embed_dim: int = 256
    num_colours: int = 10
    max_grid_size: int = 30
    num_encoder_layers: int = 4
    num_cross_attn_layers: int = 4
    num_attn_heads: int = 8
    num_rule_slots: int = 128
    rule_rank: int = 16
    history_size: int = 512

    def setup(self) -> None:
        self.max_output_cells = self.max_grid_size * self.max_grid_size

        # ── Cell embedding ────────────────────────────────────────────────
        # +1 for the PAD sentinel (-1 mapped to index num_colours).
        self.cell_embed = nn.Embed(self.num_colours + 1, self.embed_dim)

        # Learnable type embeddings to distinguish demo-input, demo-output,
        # and test-input tokens within the same sequence.
        self.type_embed = nn.Embed(3, self.embed_dim)  # 0=demo_in, 1=demo_out, 2=test_in

        # 2-D sinusoidal positional encoding (computed once, stored as constant).
        self.pos_encoding = _sinusoidal_pos_encoding_2d(
            self.max_grid_size, self.max_grid_size, self.embed_dim,
        )

        # ── Demo pair encoder (shared Transformer) ────────────────────────
        self.encoder_layers = [
            TransformerEncoderLayer(
                embed_dim=self.embed_dim,
                num_heads=self.num_attn_heads,
                name=f"encoder_layer_{i}",
            )
            for i in range(self.num_encoder_layers)
        ]

        # ── Multi-head cross-attention (test ← demo context) ─────────────
        self.cross_attn_layers = [
            CrossAttentionLayer(
                embed_dim=self.embed_dim,
                num_heads=self.num_attn_heads,
                name=f"cross_attn_layer_{i}",
            )
            for i in range(self.num_cross_attn_layers)
        ]

        # ── Pooling projection ────────────────────────────────────────────
        self.pool_proj_dense = nn.Dense(self.embed_dim)

        # ── Expert pathways ──────────────────────────────────────────────
        self.memory = RuleMemory(
            embed_dim=self.embed_dim,
            num_colours=self.num_colours,
            num_slots=self.num_rule_slots,
            rank=self.rule_rank,
        )
        self.rule_gen = RuleGenerator(
            embed_dim=self.embed_dim,
            num_colours=self.num_colours,
            rank=self.rule_rank,
            history_size=self.history_size,
        )
        self.guess = GuessComponent(
            embed_dim=self.embed_dim,
            num_colours=self.num_colours,
        )

        # ── Decision router ──────────────────────────────────────────────
        self.router = DecisionRouter(embed_dim=self.embed_dim)

    # ── Helper: embed a batch of grids ────────────────────────────────────

    def _embed_grid(
        self, grid: jnp.ndarray, type_id: int,
    ) -> jnp.ndarray:
        """Embed a padded grid into a sequence of token vectors.

        :param grid: ``(batch, H, W)`` int array.
        :param type_id: Type embedding index (0=demo_in, 1=demo_out, 2=test_in).
        :return: ``(batch, H*W, embed_dim)`` token embeddings.
        """
        B, H, W = grid.shape
        safe = jnp.where(grid < 0, self.num_colours, grid)
        tokens = self.cell_embed(safe.reshape(B, -1))  # (B, H*W, embed_dim)
        tokens = tokens + self.pos_encoding[: H * W][None, :, :]
        type_emb = self.type_embed(jnp.array(type_id))  # (embed_dim,)
        tokens = tokens + type_emb[None, None, :]
        return tokens

    # ── Forward pass ──────────────────────────────────────────────────────

    def __call__(
        self,
        demo_inputs: jnp.ndarray,
        demo_outputs: jnp.ndarray,
        demo_mask: jnp.ndarray,
        test_input: jnp.ndarray,
        *,
        training: bool = False,
    ) -> tuple[jnp.ndarray, jnp.ndarray, dict[str, Any]]:
        """Run the full Fusion Model forward pass.

        :param demo_inputs: ``(batch, max_demos, G, G)`` padded demo input grids.
        :param demo_outputs: ``(batch, max_demos, G, G)`` padded demo output grids.
        :param demo_mask: ``(batch, max_demos)`` boolean mask.
        :param test_input: ``(batch, G, G)`` padded test input grid.
        :param training: Whether we are in training mode.
        :return: Tuple of ``(logits, alphas, metadata)``.
        """
        B, D, G, _ = demo_inputs.shape

        # ── 1. Encode each demo pair ─────────────────────────────────────
        # We process all demos and then mask, to keep shapes static for JIT.
        def encode_one_demo(d: int) -> jnp.ndarray:
            inp_emb = self._embed_grid(demo_inputs[:, d], type_id=0)
            out_emb = self._embed_grid(demo_outputs[:, d], type_id=1)
            pair_emb = jnp.concatenate([inp_emb, out_emb], axis=1)
            pair_encoded = pair_emb
            for layer in self.encoder_layers:
                pair_encoded = layer(pair_encoded, training=training)
            # Mask out demos that don't exist.
            mask_d = demo_mask[:, d].astype(jnp.float32)
            return pair_encoded * mask_d[:, None, None]

        # Static unroll over demos (D is fixed at init time).
        demo_tokens_list = [encode_one_demo(d) for d in range(D)]
        demo_context = jnp.concatenate(demo_tokens_list, axis=1)  # (B, D*2*G*G, E)

        # ── 2. Embed test input ──────────────────────────────────────────
        test_emb = self._embed_grid(test_input, type_id=2)  # (B, seq, E)

        # ── 3. Multi-head cross-attention: test ← demo context ───────────
        x = test_emb
        for cross_layer in self.cross_attn_layers:
            x = cross_layer(x, demo_context, training=training)

        # ── 4. Pool into shared embedding h ──────────────────────────────
        pad_mask = (test_input.reshape(B, -1) >= 0).astype(jnp.float32)  # (B, seq)
        pad_mask_sum = jnp.clip(pad_mask.sum(axis=-1, keepdims=True), min=1.0)
        h = (x * pad_mask[:, :, None]).sum(axis=1) / pad_mask_sum  # (B, E)
        h = nn.gelu(self.pool_proj_dense(h))

        # ── 5. Expert pathways (spatial) ─────────────────────────────────
        logits_mem, mem_repr, retrieval_info = self.memory(x, h, training=training)
        logits_rule, confidence, rule_repr, proposal = self.rule_gen(x, h, training=training)
        logits_guess, guess_repr = self.guess(x, training=training)

        # ── 6. Route and blend ───────────────────────────────────────────
        alpha, router_attn = self.router(h, mem_repr, rule_repr, guess_repr)

        # alpha: (B, 3) → expand for per-cell blending.
        logits = (
            alpha[:, 0:1, None] * logits_mem
            + alpha[:, 1:2, None] * logits_rule
            + alpha[:, 2:3, None] * logits_guess
        )  # (B, seq, num_colours)

        metadata: dict[str, Any] = {
            "logits_mem": logits_mem,
            "logits_rule": logits_rule,
            "logits_guess": logits_guess,
            "retrieval_scores": retrieval_info["scores"],
            "memory_strength": retrieval_info["strength"],
            "rule_confidence": confidence,
            "router_attn": router_attn,
            "proposal": proposal,
        }
        return logits, alpha, metadata
