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
   - :class:`~fusion_model.guess.GuessComponent` (FiLM-conditioned local/global attention)
6. **DecisionRouter** — produces softmax mixture weights ``alpha`` over
   the three pathways.
7. **Output** — the blended per-cell logits ``(batch, seq, num_colours)``.
8. **Rule commitment** and **history update** proceed as before.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn

from .decision import DecisionRouter
from .guess import GuessComponent
from .memory import RuleMemory
from .rule_engine import RuleGenerator


def _sinusoidal_pos_encoding_2d(
    max_h: int, max_w: int, embed_dim: int
) -> torch.Tensor:
    """Generate 2-D sinusoidal positional encodings.

    Returns a tensor of shape ``(max_h * max_w, embed_dim)`` where the
    first half of channels encode the row position and the second half
    encode the column position.
    """
    half = embed_dim // 2
    pos_h = torch.arange(max_h, dtype=torch.float).unsqueeze(1)  # (H, 1)
    pos_w = torch.arange(max_w, dtype=torch.float).unsqueeze(1)  # (W, 1)
    div = torch.exp(torch.arange(0, half, 2, dtype=torch.float) * -(math.log(10000.0) / half))

    pe_h = torch.zeros(max_h, half)
    pe_h[:, 0::2] = torch.sin(pos_h * div[: half // 2 + (half % 2)])
    pe_h[:, 1::2] = torch.cos(pos_h * div[: half // 2])

    pe_w = torch.zeros(max_w, half)
    pe_w[:, 0::2] = torch.sin(pos_w * div[: half // 2 + (half % 2)])
    pe_w[:, 1::2] = torch.cos(pos_w * div[: half // 2])

    # Broadcast: (H, 1, half) + (1, W, half) → (H, W, half)
    pe = torch.cat(
        [pe_h.unsqueeze(1).expand(-1, max_w, -1),
         pe_w.unsqueeze(0).expand(max_h, -1, -1)],
        dim=-1,
    )  # (H, W, embed_dim)
    return pe


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

    def __init__(
        self,
        embed_dim: int = 256,
        num_colours: int = 10,
        max_grid_size: int = 30,
        num_encoder_layers: int = 4,
        num_cross_attn_layers: int = 4,
        num_attn_heads: int = 8,
        num_rule_slots: int = 128,
        rule_rank: int = 16,
        history_size: int = 512,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_colours = num_colours
        self.max_grid_size = max_grid_size
        self.max_output_cells = max_grid_size * max_grid_size  # 900 for 30×30

        # ── Cell embedding ────────────────────────────────────────────────
        # +1 for the PAD sentinel (-1 mapped to index num_colours).
        self.cell_embed = nn.Embedding(
            num_colours + 1, embed_dim, padding_idx=num_colours,
        )

        # Learnable type embeddings to distinguish demo-input, demo-output,
        # and test-input tokens within the same sequence.
        self.type_embed = nn.Embedding(3, embed_dim)  # 0=demo_in, 1=demo_out, 2=test_in

        # 2-D sinusoidal positional encoding (registered as buffer).
        self.register_buffer(
            "pos_encoding",
            _sinusoidal_pos_encoding_2d(max_grid_size, max_grid_size, embed_dim),
        )

        # ── Demo pair encoder (shared Transformer) ────────────────────────
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_attn_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )
        self.demo_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers,
        )

        # ── Multi-head cross-attention (test ← demo context) ─────────────
        self.cross_attn_layers = nn.ModuleList()
        self.cross_norms = nn.ModuleList()
        self.cross_ffns = nn.ModuleList()
        self.cross_ffn_norms = nn.ModuleList()
        for _ in range(num_cross_attn_layers):
            self.cross_attn_layers.append(
                nn.MultiheadAttention(
                    embed_dim, num_attn_heads, dropout=0.1, batch_first=True,
                )
            )
            self.cross_norms.append(nn.LayerNorm(embed_dim))
            self.cross_ffns.append(nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 4),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(embed_dim * 4, embed_dim),
                nn.Dropout(0.1),
            ))
            self.cross_ffn_norms.append(nn.LayerNorm(embed_dim))

        # ── Pooling projection ────────────────────────────────────────────
        self.pool_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
        )

        # ── Expert pathways ──────────────────────────────────────────────
        self.memory = RuleMemory(
            embed_dim=embed_dim,
            num_colours=num_colours,
            num_slots=num_rule_slots,
            rank=rule_rank,
        )
        self.rule_gen = RuleGenerator(
            embed_dim=embed_dim,
            num_colours=num_colours,
            rank=rule_rank,
            history_size=history_size,
        )
        self.guess = GuessComponent(
            embed_dim=embed_dim,
            num_colours=num_colours,
        )

        # ── Decision router ──────────────────────────────────────────────
        self.router = DecisionRouter(embed_dim=embed_dim)

    # ── Deferred history update ─────────────────────────────────────────

    @torch.no_grad()
    def update_rule_history(self, outcomes: torch.Tensor) -> None:
        """Push the pending ``(h, decision, outcome)`` triple into the
        RuleGenerator's history buffer.

        Must be called by the training loop *after* computing the per-sample
        loss.  Does nothing if no pending history exists (e.g. during eval).

        :param outcomes: Per-sample outcome signal ``(batch,)``.
        """
        pending = getattr(self, "_pending_history", None)
        if pending is None:
            return
        self.rule_gen.update_history(
            pending["h"], pending["decisions"], outcomes,
        )
        self._pending_history = None

    # ── Helper: embed a batch of grids ────────────────────────────────────

    def _embed_grid(
        self, grid: torch.Tensor, type_id: int
    ) -> torch.Tensor:
        """Embed a padded grid into a sequence of token vectors.

        :param grid: ``(batch, H, W)`` int tensor.
        :param type_id: Type embedding index (0=demo_in, 1=demo_out, 2=test_in).
        :return: ``(batch, H*W, embed_dim)`` token embeddings.
        """
        B, H, W = grid.shape
        safe = grid.clone()
        safe[safe < 0] = self.num_colours
        tokens = self.cell_embed(safe.view(B, -1))  # (B, H*W, embed_dim)
        tokens = tokens + self.pos_encoding[:H, :W].reshape(H * W, -1).unsqueeze(0)
        tokens = tokens + self.type_embed(
            torch.full((1,), type_id, device=grid.device, dtype=torch.long)
        )
        return tokens

    # ── Forward pass ──────────────────────────────────────────────────────

    def forward(
        self,
        demo_inputs: torch.Tensor,
        demo_outputs: torch.Tensor,
        demo_mask: torch.Tensor,
        test_input: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Run the full Fusion Model forward pass.

        :param demo_inputs: ``(batch, max_demos, G, G)`` padded demo input grids.
        :param demo_outputs: ``(batch, max_demos, G, G)`` padded demo output grids.
        :param demo_mask: ``(batch, max_demos)`` boolean mask.
        :param test_input: ``(batch, G, G)`` padded test input grid.
        :return: Tuple of ``(logits, alphas, metadata)`` where *logits*
            has shape ``(batch, max_output_cells, num_colours)``.
        """
        B, D, H, W = demo_inputs.shape

        # ── 1. Encode each demo pair ─────────────────────────────────────
        demo_tokens_list: list[torch.Tensor] = []

        for d in range(D):
            mask_d = demo_mask[:, d]
            if not mask_d.any():
                continue

            inp_emb = self._embed_grid(demo_inputs[:, d], type_id=0)
            out_emb = self._embed_grid(demo_outputs[:, d], type_id=1)
            pair_emb = torch.cat([inp_emb, out_emb], dim=1)

            pair_encoded = self.demo_encoder(pair_emb)

            pair_encoded = pair_encoded * mask_d.float().view(B, 1, 1)
            demo_tokens_list.append(pair_encoded)

        if demo_tokens_list:
            demo_context = torch.cat(demo_tokens_list, dim=1)
        else:
            demo_context = torch.zeros(
                B, 1, self.embed_dim, device=test_input.device,
            )

        # ── 2. Embed test input ──────────────────────────────────────────
        test_emb = self._embed_grid(test_input, type_id=2)  # (B, seq, E)

        # ── 3. Multi-head cross-attention: test ← demo context ───────────
        x = test_emb
        for cross_attn, norm, ffn, ffn_norm in zip(
            self.cross_attn_layers,
            self.cross_norms,
            self.cross_ffns,
            self.cross_ffn_norms,
            strict=True,
        ):
            attended, _ = cross_attn(query=x, key=demo_context, value=demo_context)
            x = norm(x + attended)
            x = ffn_norm(x + ffn(x))

        # ── 4. Pool into shared embedding h ──────────────────────────────
        pad_mask = (test_input.view(B, -1) >= 0).float()  # (B, seq)
        pad_mask_sum = pad_mask.sum(dim=-1, keepdim=True).clamp(min=1.0)
        h = (x * pad_mask.unsqueeze(-1)).sum(dim=1) / pad_mask_sum  # (B, E)
        h = self.pool_proj(h)

        # ── 5. Expert pathways (spatial) ─────────────────────────────────
        # Each expert receives x (B, seq, E) and produces (B, seq, num_colours).
        logits_mem, mem_repr, retrieval_info = self.memory(x, h)
        logits_rule, confidence, rule_repr, proposal = self.rule_gen(x, h)
        logits_guess, guess_repr = self.guess(x, h, grid_h=H, grid_w=W)

        # ── 6. Route and blend ───────────────────────────────────────────
        alpha, router_attn = self.router(h, mem_repr, rule_repr, guess_repr)

        # alpha: (B, 3) → expand for per-cell blending.
        logits = (
            alpha[:, 0:1].unsqueeze(-1) * logits_mem
            + alpha[:, 1:2].unsqueeze(-1) * logits_rule
            + alpha[:, 2:3].unsqueeze(-1) * logits_guess
        )  # (B, seq, num_colours)

        # ── 7. Rule commitment (soft blend into weakest slot) ────────────
        committed = False
        commit_weight_used = 0.0
        if self.training and proposal is not None:
            weights = proposal["commit_weight"]  # (batch, 1)
            best_idx = int(weights.argmax(dim=0).item())
            w = weights[best_idx, 0].item()

            if w > 1e-3:
                slot = self.memory.get_weakest_slot()
                self.memory.commit_rule(
                    slot,
                    proposal["key"][best_idx],
                    proposal["A"][best_idx],
                    proposal["B"][best_idx],
                    commit_weight=w,
                )
                committed = True
                commit_weight_used = w

        # ── 8. Stash info for deferred history update ─────────────────────
        if self.training:
            pred_cells = logits.detach().argmax(dim=-1)  # (B, seq)
            pred_hash = pred_cells.sum(dim=-1) % self.rule_gen.decision_vocab_size
            self._pending_history = {
                "h": h.detach(),
                "decisions": pred_hash,
            }

        metadata: dict[str, Any] = {
            "logits_mem": logits_mem,
            "logits_rule": logits_rule,
            "logits_guess": logits_guess,
            "retrieval_scores": retrieval_info["scores"],
            "memory_strength": retrieval_info["strength"],
            "rule_confidence": confidence,
            "router_attn": router_attn,
            "proposal": proposal,
            "committed": committed,
            "commit_weight": commit_weight_used,
        }
        return logits, alpha, metadata
