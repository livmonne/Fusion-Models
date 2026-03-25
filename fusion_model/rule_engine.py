"""RuleGenerator — ephemeral corrections *and* persistent rule proposals.

**Concept:**  The RuleGenerator serves two roles:

1. **Ephemeral correction** — a small MLP takes the pooled embedding ``h``
   and outputs a one-shot low-rank correction applied **per spatial token**:
   ``A_new @ (B_new @ x_token)`` plus a confidence scalar.

2. **Rule proposal** — the generator maintains a circular history buffer of
   recent ``(h, decision, outcome)`` triples.  The proposal pipeline has
   three stages of cross-attention that operate *entirely on history* — the
   current input is deliberately excluded so that proposed rules reflect
   what actually worked rather than what the model is currently looking at.

This two-tier design lets the model invent hypotheses on the fly *and*
distill recurring patterns into the long-term rule bank.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import flax.linen as nn


class RuleGenerator(nn.Module):
    """Generate ephemeral low-rank corrections and propose persistent rules.

    Now operates on **per-token spatial embeddings**: the MLP produces A, B
    matrices from the pooled ``h``, but the correction is applied to every
    token in the spatial sequence ``x``.

    :param embed_dim: Dimensionality of the shared input embedding.
    :param num_colours: Number of per-cell colour classes (10 for ARC).
    :param rank: Inner rank of the generated low-rank matrices.
    :param hidden_dim: Width of the internal MLP.
    :param history_size: Capacity of the circular buffer.
    :param min_history: Minimum entries before rule proposal is attempted.
    :param decision_vocab_size: Vocabulary size for decision embeddings
        stored in history.
    :param decision_embed_dim: Embedding dimension for stored decision
        indices before projection to ``embed_dim``.
    :param outcome_proj_dim: Hidden dimension for projecting scalar outcome
        signals to ``embed_dim``.
    """

    embed_dim: int = 256
    num_colours: int = 10
    rank: int = 8
    hidden_dim: int = 256
    history_size: int = 512
    min_history: int = 64
    decision_vocab_size: int = 9000
    decision_embed_dim: int = 32
    outcome_proj_dim: int = 64

    def setup(self) -> None:
        # ── Ephemeral correction MLP ────────────────────────
        out_dim = (self.rank * self.embed_dim) + (self.embed_dim * self.rank) + 1
        self.mlp_dense1 = nn.Dense(self.hidden_dim)
        self.mlp_dropout = nn.Dropout(0.1)
        self.mlp_dense2 = nn.Dense(out_dim)
        # Per-token classification head.
        self.head = nn.Dense(self.num_colours)

        # ── Rule proposer ────────────────────────────────────────────────
        self.decision_embed = nn.Embed(self.decision_vocab_size, self.decision_embed_dim)

        self.input_dec_cross_attn = nn.MultiHeadDotProductAttention(
            num_heads=4, qkv_features=self.embed_dim,
        )
        self.dec_proj = nn.Dense(self.embed_dim)

        self.outcome_proj = nn.Sequential([
            nn.Dense(self.outcome_proj_dim),
            nn.gelu,
            nn.Dense(self.embed_dim),
        ])
        self.outcome_cross_attn = nn.MultiHeadDotProductAttention(
            num_heads=4, qkv_features=self.embed_dim,
        )

        self.synthesis_query = self.param(
            "synthesis_query",
            lambda rng, shape: jax.random.normal(rng, shape) * 0.02,
            (1, 1, self.embed_dim),
        )
        self.synthesis_cross_attn = nn.MultiHeadDotProductAttention(
            num_heads=4, qkv_features=self.embed_dim,
        )

        proposer_out = self.embed_dim + (self.embed_dim * self.rank) + (self.rank * self.embed_dim)
        self.rule_proj = nn.Dense(proposer_out)

        self.commit_threshold_logit = self.param(
            "commit_threshold_logit", lambda _rng, _shape: jnp.array(0.0), (),
        )
        self.commit_temperature = self.param(
            "commit_temperature", lambda _rng, _shape: jnp.array(1.0), (),
        )

        # ── Mutable state (history buffer) ───────────────────────────────
        self._history_h = self.variable(
            "state", "history_h",
            lambda: jnp.zeros((self.history_size, self.embed_dim)),
        )
        self._history_decisions = self.variable(
            "state", "history_decisions",
            lambda: jnp.full((self.history_size,), -1, dtype=jnp.int32),
        )
        self._history_outcomes = self.variable(
            "state", "history_outcomes",
            lambda: jnp.zeros((self.history_size,)),
        )
        self._history_ptr = self.variable(
            "state", "history_ptr",
            lambda: jnp.array(0, dtype=jnp.int32),
        )
        self._history_count = self.variable(
            "state", "history_count",
            lambda: jnp.array(0, dtype=jnp.int32),
        )

    # ── History management ───────────────────────────────────────────────

    def update_history(
        self,
        h: jnp.ndarray,
        decisions: jnp.ndarray,
        outcomes: jnp.ndarray,
    ) -> None:
        """Append a batch of ``(h, decision, outcome)`` triples to the buffer."""
        h_det = jax.lax.stop_gradient(h)
        dec_det = jax.lax.stop_gradient(decisions)
        out_det = jax.lax.stop_gradient(outcomes)

        history_h = self._history_h.value
        history_decisions = self._history_decisions.value
        history_outcomes = self._history_outcomes.value
        ptr = self._history_ptr.value

        batch = h_det.shape[0]

        # Write each sample using scan with wrapping indices.
        def write_one(carry, i):
            hh, hd, ho = carry
            idx = (ptr + i) % self.history_size
            hh = hh.at[idx].set(h_det[i])
            hd = hd.at[idx].set(dec_det[i])
            ho = ho.at[idx].set(out_det[i])
            return (hh, hd, ho), None

        (history_h, history_decisions, history_outcomes), _ = jax.lax.scan(
            write_one,
            (history_h, history_decisions, history_outcomes),
            jnp.arange(batch),
        )

        self._history_h.value = history_h
        self._history_decisions.value = history_decisions
        self._history_outcomes.value = history_outcomes
        self._history_ptr.value = (ptr + batch) % self.history_size
        self._history_count.value = jnp.minimum(
            self._history_count.value + batch, self.history_size,
        )

    # ── Rule proposal ────────────────────────────────────────────────────

    def propose_rule(self, batch_size: int) -> dict[str, jnp.ndarray] | None:
        """Propose a persistent rule from historical context only.

        Returns ``None`` when the history buffer has fewer than
        ``min_history`` entries.
        """
        count = self._history_count.value
        count_val = count.item() if hasattr(count, 'item') else int(count)
        if count_val < self.min_history:
            return None

        history_h = self._history_h.value
        history_decisions = self._history_decisions.value
        history_outcomes = self._history_outcomes.value

        valid_h = history_h[:count_val]
        valid_dec = history_decisions[:count_val]
        valid_out = history_outcomes[:count_val]

        hist_h = jnp.broadcast_to(
            valid_h[None, :, :], (batch_size, count_val, self.embed_dim),
        )

        # Stage 1: historical inputs cross-attend over historical decisions.
        dec_emb = self.dec_proj(self.decision_embed(valid_dec))
        kv_dec = jnp.broadcast_to(
            dec_emb[None, :, :], (batch_size, count_val, self.embed_dim),
        )
        attended_input_dec = self.input_dec_cross_attn(hist_h, kv_dec)

        # Stage 2: input→decision result cross-attends over outcome embeddings.
        outcome_emb = self.outcome_proj(valid_out[:, None])
        kv_out = jnp.broadcast_to(
            outcome_emb[None, :, :], (batch_size, count_val, self.embed_dim),
        )
        attended_outcome = self.outcome_cross_attn(attended_input_dec, kv_out)

        attended_input_dec_pooled = attended_input_dec.mean(axis=1)
        attended_outcome_pooled = attended_outcome.mean(axis=1)

        # Stage 3: learned synthesis query attends over the two pooled results.
        context_tokens = jnp.stack(
            [attended_input_dec_pooled, attended_outcome_pooled], axis=1,
        )
        syn_query = jnp.broadcast_to(self.synthesis_query, (batch_size, 1, self.embed_dim))
        synthesised = self.synthesis_cross_attn(syn_query, context_tokens)
        synthesised = synthesised.squeeze(1)

        raw = self.rule_proj(synthesised)

        e, r = self.embed_dim, self.rank
        key = raw[:, :e]
        a_flat = raw[:, e : e + e * r]
        b_flat = raw[:, e + e * r : e + 2 * e * r]

        A = a_flat.reshape(-1, e, r)
        B = b_flat.reshape(-1, r, e)

        mean_hist = valid_h.mean(axis=0, keepdims=True)
        mean_hist = jnp.broadcast_to(mean_hist, (batch_size, e))
        # Cosine similarity.
        similarity = (
            (key * mean_hist).sum(axis=-1)
            / (jnp.linalg.norm(key, axis=-1) * jnp.linalg.norm(mean_hist, axis=-1) + 1e-8)
        )
        threshold = jax.nn.sigmoid(self.commit_threshold_logit)
        temperature = jnp.clip(self.commit_temperature, min=0.01)
        commit_weight = jax.nn.sigmoid(
            (similarity - threshold) * temperature,
        )[:, None]

        return {
            "key": key,
            "A": A,
            "B": B,
            "commit_weight": commit_weight,
        }

    # ── Forward pass ─────────────────────────────────────────────────────

    def __call__(
        self, x: jnp.ndarray, h: jnp.ndarray, *, training: bool = False,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, dict[str, jnp.ndarray] | None]:
        """Produce per-token ephemeral rule logits, confidence, repr, and a rule proposal.

        :param x: Spatial token embeddings ``(batch, seq, embed_dim)``.
        :param h: Pooled embedding ``(batch, embed_dim)`` for the MLP and
            history/proposal machinery.
        :param training: Whether we are in training mode.
        :return: Tuple ``(logits_rule, confidence, rule_repr, proposal)``.
        """
        batch = h.shape[0]
        hidden = nn.gelu(self.mlp_dense1(h))
        hidden = self.mlp_dropout(hidden, deterministic=not training)
        params = self.mlp_dense2(hidden)

        b_size = self.rank * self.embed_dim
        a_size = self.embed_dim * self.rank
        b_flat = params[:, :b_size]
        a_flat = params[:, b_size : b_size + a_size]
        confidence = jax.nn.sigmoid(params[:, -1:])

        b_mat = b_flat.reshape(batch, self.rank, self.embed_dim)       # (B, r, E)
        a_mat = a_flat.reshape(batch, self.embed_dim, self.rank)       # (B, E, r)

        # Per-token correction: A @ (B @ x_token) for every token.
        compressed = jnp.matmul(b_mat, jnp.transpose(x, (0, 2, 1)))  # (B, r, seq)
        correction = jnp.transpose(
            jnp.matmul(a_mat, compressed), (0, 2, 1),
        )  # (B, seq, E)

        logits_rule = self.head(correction)  # (B, seq, num_colours)

        # Router representation: mean-pool the per-token corrections.
        rule_repr = correction.mean(axis=1)  # (B, E)

        proposal = self.propose_rule(batch)

        return logits_rule, confidence, rule_repr, proposal
