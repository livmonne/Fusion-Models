"""FusionModel — orchestrator that wires all components together.

The Fusion Model is an architecture for abstract reasoning on grid
transformation tasks.  It accepts a set of **demonstration input/output
grid pairs** and a **test input grid**, and produces a predicted output
grid by inferring the transformation rule from the demos.

**Forward pass overview:**

1. **Grid cell embedding** — each cell value (0–9) is mapped to a learned
   embedding vector.  2-D sinusoidal positional encodings are added so the
   model knows where each cell sits, plus a type embedding (demo-in /
   demo-out / test-in) and a demo-index embedding so tokens from different
   demonstrations are distinguishable.
2. **Demo pair encoding** — for each demonstration pair the input and
   output cell embeddings are concatenated along the sequence dimension
   and processed by a shared Transformer encoder.  Padding cells are
   excluded from attention via key-padding masks.
3. **Task embedding ``t``** — per-demo masked mean-pools of the encoded
   pairs are averaged over valid demos and projected.  ``t`` describes the
   *transformation* (it is derived purely from the demos) and conditions
   rule retrieval, rule generation, and FiLM in the guess pathway.
4. **Multi-head cross-attention** — the test input cell embeddings
   cross-attend to the demo context (padding and invalid demos masked).
5. **Instance embedding ``h``** — masked mean-pool of the cross-attended
   test tokens; used by the router and the size head.
6. **Four expert pathways** each receive the spatial sequence ``x`` and
   the task embedding ``t`` and return per-cell colour logits plus a
   pooled representation for the router:
   - :class:`~fusion_model.memory.RuleMemory` (pathway "mem"),
   - the ephemeral generator of
     :class:`~fusion_model.rule_engine.RuleGenerator` (pathway "rule"),
   - the history proposer of the same module (pathway "prop") — the
     proposed rule is *tried* on the current input so the task loss can
     grade and train the proposal machinery,
   - :class:`~fusion_model.guess.GuessComponent` (pathway "guess").
7. **Leave-one-out verification** (optional, on by default) — one
   demonstration is held out, its output is predicted from the remaining
   demos by every pathway, and the measured per-pathway fit is (a) fed to
   the router as a grounded routing signal and (b) returned as an extra
   training signal.  This is the mechanism that makes "rule" mean
   something: a rule is good iff it reproduces held-out demonstrations.
8. **Output size head** — predicts the output grid's height and width
   from ``[t ‖ h]`` (ARC outputs frequently differ in size from inputs).
9. **DecisionRouter** — produces softmax mixture weights ``alpha`` over
   the four pathways (grounded by the verification fit when available).
10. **Output** — the blended per-cell logits ``(batch, seq, num_colours)``.

**Deferred history & commits:** during training the forward pass stashes
the task embedding, a decision summary (predicted colour histogram +
routing weights), and the latest rule proposal.  The training loop calls
:meth:`apply_outcomes` *after* the backward pass with the measured
per-sample losses; this updates the proposer's history buffer and commits
the proposed rule into the memory bank only when it demonstrably beat the
running average of recent proposals.  Keeping the in-place parameter
update out of the forward/backward window also keeps gradients exact.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import masked_mean, per_sample_ce
from .decision import NUM_PATHWAYS, PROP_INDEX, DecisionRouter
from .guess import GuessComponent
from .memory import RuleMemory
from .rule_engine import RuleGenerator


def _sinusoidal_pos_encoding_2d(
    max_h: int, max_w: int, embed_dim: int
) -> torch.Tensor:
    """Generate 2-D sinusoidal positional encodings.

    Returns a tensor of shape ``(max_h, max_w, embed_dim)`` where the
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
    """End-to-end Fusion Model for few-shot grid transformation tasks.

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
    :param max_demos: Upper bound on the number of demonstrations per
        task (used for the demo-index embedding table).
    :param verify_demos: Whether to run leave-one-out demo verification.
        During training one random demo per sample is verified; during
        evaluation the fit is averaged over all valid demos.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_colours: int = 10,
        max_grid_size: int = 30,
        num_encoder_layers: int = 4,
        num_cross_attn_layers: int = 4,
        num_attn_heads: int = 8,
        num_rule_slots: int = 64,
        rule_rank: int = 16,
        history_size: int = 512,
        max_demos: int = 10,
        verify_demos: bool = True,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_colours = num_colours
        self.max_grid_size = max_grid_size
        self.max_demos = max_demos
        self.verify_demos = verify_demos

        # ── Cell embedding ────────────────────────────────────────────────
        # +1 for the PAD sentinel (-1 mapped to index num_colours).
        self.cell_embed = nn.Embedding(
            num_colours + 1, embed_dim, padding_idx=num_colours,
        )

        # Learnable type embeddings to distinguish demo-input, demo-output,
        # and test-input tokens within the same sequence.
        self.type_embed = nn.Embedding(3, embed_dim)  # 0=demo_in, 1=demo_out, 2=test_in

        # Demo-index embedding so tokens from different demonstrations are
        # distinguishable inside the concatenated demo context.
        self.demo_idx_embed = nn.Embedding(max_demos, embed_dim)

        # 2-D sinusoidal positional encoding (registered as buffer).
        self.pos_encoding: torch.Tensor
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
        # enable_nested_tensor=False: the nested-tensor fast path only
        # activates at inference and its mask pre-check is unimplemented
        # on MPS; disabling it keeps behaviour identical everywhere.
        self.demo_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers,
            enable_nested_tensor=False,
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

        # ── Pooling projections ──────────────────────────────────────────
        # h: instance embedding from the cross-attended test tokens.
        self.pool_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
        )
        # t: task embedding from the encoded demo pairs.
        self.task_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
        )

        # ── Output size head ─────────────────────────────────────────────
        # Predicts output grid height and width (class k ↔ size k+1) from
        # the task and instance embeddings.
        self.size_head = nn.Sequential(
            nn.Linear(2 * embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 2 * max_grid_size),
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
            num_pathways=NUM_PATHWAYS,
        )
        self.guess = GuessComponent(
            embed_dim=embed_dim,
            num_colours=num_colours,
        )

        # ── Decision router ──────────────────────────────────────────────
        self.router = DecisionRouter(embed_dim=embed_dim)

        # Pending state consumed by apply_outcomes (training only).
        self._pending_history: dict[str, torch.Tensor] | None = None
        self._pending_proposal: dict[str, torch.Tensor] | None = None

    # ── Deferred history update & rule commitment ────────────────────────

    @torch.no_grad()
    def apply_outcomes(
        self,
        outcomes: torch.Tensor,
        prop_outcomes: torch.Tensor | None = None,
        allow_commit: bool = True,
    ) -> dict[str, Any]:
        """Feed measured outcomes back into the model after the backward pass.

        Must be called by the training loop *after* ``loss.backward()``:

        1. Pushes the pending ``(task, decision, outcome)`` triple into the
           RuleGenerator's history buffer.
        2. Decides whether to commit the pending rule proposal into the
           memory bank.  The gate is *outcome-based*: the proposal
           pathway's measured per-sample loss must beat the running
           average of recent proposal losses (see
           :meth:`RuleGenerator.should_commit`).  Pass
           ``allow_commit=False`` while gradients are still accumulating
           so the in-place slot update never races a pending
           ``optimizer.step()``.

        :param outcomes: Per-sample task loss ``(batch,)`` of the blended
            prediction (history outcome signal).
        :param prop_outcomes: Per-sample task loss ``(batch,)`` of the
            proposal pathway, used for the commit gate.  ``None`` when the
            proposal pathway was inactive.
        :param allow_commit: Whether a commit may be performed this call.
        :return: Dict with ``committed`` (bool) and ``commit_weight``
            (float) for logging.
        """
        info: dict[str, Any] = {"committed": False, "commit_weight": 0.0}

        pending = self._pending_history
        if pending is not None:
            self.rule_gen.update_history(
                pending["task"], pending["decisions"], outcomes,
            )
            self._pending_history = None

        proposal = self._pending_proposal
        if proposal is not None and prop_outcomes is not None:
            best = self.rule_gen.should_commit(prop_outcomes)
            if allow_commit and best is not None:
                w = float(proposal["commit_weight"][best, 0].item())
                if w > 1e-3:
                    slot = self.memory.get_weakest_slot()
                    self.memory.commit_rule(
                        slot,
                        proposal["key"][best],
                        proposal["A"][best],
                        proposal["B"][best],
                        commit_weight=w,
                    )
                    info = {"committed": True, "commit_weight": w}
        if allow_commit:
            self._pending_proposal = None
        return info

    # ── Helpers ───────────────────────────────────────────────────────────

    def _embed_grid(
        self, grid: torch.Tensor, type_id: int, demo_idx: int | None = None
    ) -> torch.Tensor:
        """Embed a padded grid into a sequence of token vectors.

        :param grid: ``(batch, H, W)`` int tensor.
        :param type_id: Type embedding index (0=demo_in, 1=demo_out, 2=test_in).
        :param demo_idx: Demonstration index for demo grids (adds the
            demo-index embedding); ``None`` for the test input.
        :return: ``(batch, H*W, embed_dim)`` token embeddings.
        """
        B, H, W = grid.shape
        safe = grid.clone()
        safe[safe < 0] = self.num_colours
        tokens: torch.Tensor = self.cell_embed(safe.view(B, -1))  # (B, H*W, embed_dim)
        tokens = tokens + self.pos_encoding[:H, :W].reshape(H * W, -1).unsqueeze(0)
        tokens = tokens + self.type_embed(
            torch.full((1,), type_id, device=grid.device, dtype=torch.long)
        )
        if demo_idx is not None:
            idx = min(demo_idx, self.max_demos - 1)
            tokens = tokens + self.demo_idx_embed(
                torch.full((1,), idx, device=grid.device, dtype=torch.long)
            )
        return tokens

    def _cross_attend(
        self,
        queries: torch.Tensor,
        demo_context: torch.Tensor,
        context_invalid: torch.Tensor,
    ) -> torch.Tensor:
        """Run the cross-attention stack: queries ← demo context.

        :param queries: ``(batch, seq, embed_dim)`` query tokens.
        :param demo_context: ``(batch, ctx, embed_dim)`` demo tokens.
        :param context_invalid: ``(batch, ctx)`` boolean; True for context
            positions to ignore (padding cells / absent demos).
        :return: Cross-attended tokens ``(batch, seq, embed_dim)``.
        """
        # Safety: a sample with no valid context tokens would make every
        # attention row -inf (NaN).  Should not happen (every task has at
        # least one demo), but guard against malformed batches.
        no_ctx = context_invalid.all(dim=1)
        if bool(no_ctx.any()):
            context_invalid = context_invalid.clone()
            context_invalid[no_ctx] = False

        x = queries
        for cross_attn, norm, ffn, ffn_norm in zip(
            self.cross_attn_layers,
            self.cross_norms,
            self.cross_ffns,
            self.cross_ffn_norms,
            strict=True,
        ):
            # need_weights=False enables the fused SDPA path — we never
            # use the per-layer attention maps, and materialising them
            # costs O(seq × ctx) memory per head per layer.
            attended, _ = cross_attn(
                query=x,
                key=demo_context,
                value=demo_context,
                key_padding_mask=context_invalid,
                need_weights=False,
            )
            x = norm(x + attended)
            x = ffn_norm(x + ffn(x))
        return x

    def _pool_task(
        self, demo_pooled: torch.Tensor, demo_valid: torch.Tensor
    ) -> torch.Tensor:
        """Average per-demo pooled embeddings over valid demos and project.

        :param demo_pooled: ``(batch, n_demos, embed_dim)`` per-demo pools.
        :param demo_valid: ``(batch, n_demos)`` boolean validity mask.
        :return: Task embedding ``t`` of shape ``(batch, embed_dim)``.
        """
        w = demo_valid.float()
        pooled = (demo_pooled * w.unsqueeze(-1)).sum(dim=1)
        pooled = pooled / w.sum(dim=1, keepdim=True).clamp(min=1.0)
        task: torch.Tensor = self.task_proj(pooled)
        return task

    def _run_pathways(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        pad_mask: torch.Tensor,
        grid_h: int,
        grid_w: int,
        proposal: dict[str, torch.Tensor] | None,
        update_state: bool,
    ) -> tuple[list[torch.Tensor | None], torch.Tensor | None]:
        """Run mem/rule/prop/guess on a token sequence; return their logits.

        Used by the verification pass (the main pass needs the pathway
        representations and retrieval info as well, so it calls the
        components directly).

        :return: ``(logits_list, ce_placeholder)`` where *logits_list* is
            in pathway order and the proposal entry is ``None`` when no
            proposal exists.
        """
        logits_mem, _, _ = self.memory(x, t, pad_mask, update_state=update_state)
        logits_rule, _ = self.rule_gen.ephemeral_logits(x, t, pad_mask)
        if proposal is not None:
            logits_prop, _ = self.rule_gen.proposal_logits(x, proposal, pad_mask)
        else:
            logits_prop = None
        logits_guess, _ = self.guess(x, t, grid_h, grid_w, pad_mask)
        return [logits_mem, logits_rule, logits_prop, logits_guess], None

    # ── Leave-one-out demo verification ──────────────────────────────────

    def _verify(
        self,
        demo_inputs: torch.Tensor,
        demo_outputs: torch.Tensor,
        demo_mask: torch.Tensor,
        demo_context: torch.Tensor,
        context_invalid: torch.Tensor,
        demo_pooled: torch.Tensor,
        d_star: torch.Tensor,
        eligible: torch.Tensor,
        proposal: dict[str, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict held-out demo ``d_star`` with every pathway and measure CE.

        For each sample, demo ``d_star[b]`` is removed from the
        cross-attention context and the task pooling, its *input* grid is
        embedded like a test input, and all four pathways predict its
        output.  The per-pathway cross-entropy on the held-out output is
        the **measured fit** — the objective signal for both routing and
        the verification loss term.

        Samples that are not ``eligible`` (fewer than two valid demos)
        keep their full context — excluding their only demo would empty
        the context — and are masked out of the returned validity instead.

        :param d_star: ``(batch,)`` held-out demo index per sample.
        :param eligible: ``(batch,)`` True where leave-one-out is possible.
        :return: ``(ces, valid)`` — per-pathway CE ``(batch, NUM_PATHWAYS)``
            (zeros for the proposal column when inactive) and a ``(batch,)``
            validity mask.
        """
        B, D_eff, H, W = demo_inputs.shape
        device = demo_inputs.device
        ar = torch.arange(B, device=device)

        ver_input = demo_inputs[ar, d_star]                  # (B, H, W)
        ver_target = demo_outputs[ar, d_star].reshape(B, -1)  # (B, H*W)

        # Exclude the held-out demo's tokens from the context.
        block_ids = torch.arange(D_eff, device=device).repeat_interleave(2 * H * W)
        exclude = (block_ids.unsqueeze(0) == d_star.unsqueeze(1)) & eligible.unsqueeze(1)
        ctx_invalid = context_invalid | exclude

        # Task embedding from the remaining demos only.
        keep = demo_mask.clone()
        keep[ar[eligible], d_star[eligible]] = False
        t_v = self._pool_task(demo_pooled, keep)

        x_v = self._embed_grid(ver_input, type_id=2)
        x_v = self._cross_attend(x_v, demo_context, ctx_invalid)
        pad_v = ver_input.reshape(B, -1) >= 0

        logits_list, _ = self._run_pathways(
            x_v, t_v, pad_v, H, W, proposal, update_state=False,
        )

        ces: list[torch.Tensor] = []
        valid = eligible
        for logits_p in logits_list:
            if logits_p is None:
                ces.append(torch.zeros(B, device=device))
                continue
            ce_p, valid_p = per_sample_ce(logits_p, ver_target)
            ces.append(ce_p)
            valid = valid & valid_p
        return torch.stack(ces, dim=1), valid

    @staticmethod
    def _fit_from_ces(
        ces: torch.Tensor, valid: torch.Tensor, prop_active: bool
    ) -> torch.Tensor:
        """Convert per-pathway CEs into a centred fit signal for the router.

        Fit is the negative CE centred over the *active* pathways, so a
        pathway that reproduced the held-out demo better than its peers
        gets a positive routing bias.  Invalid samples get all-zero fit.
        """
        neg = -ces
        if prop_active:
            mean = neg.mean(dim=1, keepdim=True)
        else:
            active = [i for i in range(NUM_PATHWAYS) if i != PROP_INDEX]
            mean = neg[:, active].mean(dim=1, keepdim=True)
        fit = neg - mean
        if not prop_active:
            fit = fit.clone()
            fit[:, PROP_INDEX] = 0.0
        return fit * valid.unsqueeze(1).float()

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
        :return: Tuple of ``(logits, alpha, metadata)`` where *logits*
            has shape ``(batch, H*W, num_colours)`` and *alpha* has shape
            ``(batch, NUM_PATHWAYS)`` in (mem, rule, prop, guess) order.
        """
        B = demo_inputs.shape[0]
        H, W = demo_inputs.shape[2], demo_inputs.shape[3]
        device = test_input.device

        # Trim unused demo columns (datasets pad to a global max_demos).
        n_valid = demo_mask.sum(dim=1)  # (B,)
        D_eff = max(1, int(n_valid.max().item()))
        demo_inputs = demo_inputs[:, :D_eff]
        demo_outputs = demo_outputs[:, :D_eff]
        demo_mask = demo_mask[:, :D_eff]

        # ── 1. Encode each demo pair ─────────────────────────────────────
        demo_blocks: list[torch.Tensor] = []
        demo_pooled: list[torch.Tensor] = []
        ctx_invalid_blocks: list[torch.Tensor] = []

        for d in range(D_eff):
            inp_emb = self._embed_grid(demo_inputs[:, d], type_id=0, demo_idx=d)
            out_emb = self._embed_grid(demo_outputs[:, d], type_id=1, demo_idx=d)
            pair_emb = torch.cat([inp_emb, out_emb], dim=1)  # (B, 2HW, E)

            cell_valid = torch.cat(
                [
                    demo_inputs[:, d].reshape(B, -1) >= 0,
                    demo_outputs[:, d].reshape(B, -1) >= 0,
                ],
                dim=1,
            )  # (B, 2HW)

            # Encoder key-padding mask.  For samples where this demo slot
            # is entirely absent every key would be masked (NaN), so those
            # rows run unmasked and are excluded downstream instead.
            enc_invalid = ~cell_valid
            enc_invalid[~demo_mask[:, d]] = False
            pair_encoded = self.demo_encoder(
                pair_emb, src_key_padding_mask=enc_invalid,
            )

            demo_blocks.append(pair_encoded)
            demo_pooled.append(masked_mean(pair_encoded, cell_valid))
            # Context tokens are invalid if the cell is padding or the
            # whole demo slot is absent for that sample.
            ctx_invalid_blocks.append(
                (~cell_valid) | (~demo_mask[:, d]).unsqueeze(1)
            )

        demo_context = torch.cat(demo_blocks, dim=1)          # (B, D*2HW, E)
        context_invalid = torch.cat(ctx_invalid_blocks, dim=1)  # (B, D*2HW)
        demo_pooled_stack = torch.stack(demo_pooled, dim=1)    # (B, D, E)

        # ── 2. Task embedding t (from demos only) ────────────────────────
        t = self._pool_task(demo_pooled_stack, demo_mask)  # (B, E)

        # ── 3. Embed test input & cross-attend to demo context ───────────
        test_emb = self._embed_grid(test_input, type_id=2)  # (B, seq, E)
        x = self._cross_attend(test_emb, demo_context, context_invalid)

        # ── 4. Instance embedding h ──────────────────────────────────────
        pad_mask = test_input.reshape(B, -1) >= 0  # (B, seq)
        h = self.pool_proj(masked_mean(x, pad_mask))  # (B, E)

        # ── 5. Expert pathways ───────────────────────────────────────────
        logits_mem, mem_repr, retrieval_info = self.memory(x, t, pad_mask)
        gen_out = self.rule_gen(x, t, pad_mask)
        logits_rule = gen_out["logits_rule"]
        logits_prop = gen_out["logits_prop"]
        prop_active: bool = gen_out["prop_active"]
        proposal = gen_out["proposal"]
        logits_guess, guess_repr = self.guess(x, t, H, W, pad_mask)

        # ── 6. Leave-one-out verification ────────────────────────────────
        fit: torch.Tensor | None = None
        verify_ce: torch.Tensor | None = None
        verify_valid: torch.Tensor | None = None
        if self.verify_demos and D_eff >= 1:
            eligible_base = n_valid >= 2
            if self.training:
                # One random held-out demo per sample.
                d_star = (
                    torch.rand(B, device=device) * n_valid.clamp(min=1).float()
                ).long().clamp(max=D_eff - 1)
                verify_ce, verify_valid = self._verify(
                    demo_inputs, demo_outputs, demo_mask,
                    demo_context, context_invalid, demo_pooled_stack,
                    d_star, eligible_base, proposal,
                )
                fit = self._fit_from_ces(verify_ce, verify_valid, prop_active)
            else:
                # Average the fit over every valid held-out demo.
                fit_sum = torch.zeros(B, NUM_PATHWAYS, device=device)
                count = torch.zeros(B, device=device)
                for d in range(D_eff):
                    eligible_d = demo_mask[:, d] & eligible_base
                    if not bool(eligible_d.any()):
                        continue
                    d_star = torch.full((B,), d, device=device, dtype=torch.long)
                    ces_d, valid_d = self._verify(
                        demo_inputs, demo_outputs, demo_mask,
                        demo_context, context_invalid, demo_pooled_stack,
                        d_star, eligible_d, proposal,
                    )
                    fit_sum += self._fit_from_ces(ces_d, valid_d, prop_active)
                    count += valid_d.float()
                fit = fit_sum / count.clamp(min=1.0).unsqueeze(1)

        # ── 7. Output size prediction ────────────────────────────────────
        size_logits = self.size_head(torch.cat([t, h], dim=-1))
        size_logits = size_logits.view(B, 2, self.max_grid_size)

        # ── 8. Route and blend ───────────────────────────────────────────
        alpha, router_attn = self.router(
            h,
            [mem_repr, gen_out["rule_repr"], gen_out["prop_repr"], guess_repr],
            fit=fit,
            prop_active=prop_active,
        )

        # The inactive proposal pathway has alpha exactly 0, so a zero
        # placeholder keeps the blend shape-correct without contributing.
        logits_prop_safe = (
            logits_prop if logits_prop is not None else torch.zeros_like(logits_mem)
        )
        logits = (
            alpha[:, 0:1].unsqueeze(-1) * logits_mem
            + alpha[:, 1:2].unsqueeze(-1) * logits_rule
            + alpha[:, 2:3].unsqueeze(-1) * logits_prop_safe
            + alpha[:, 3:4].unsqueeze(-1) * logits_guess
        )  # (B, seq, num_colours)

        # ── 9. Stash pending state for apply_outcomes ────────────────────
        if self.training:
            preds = logits.detach().argmax(dim=-1)  # (B, seq)
            one_hot = F.one_hot(preds, self.num_colours).float()
            one_hot = one_hot * pad_mask.unsqueeze(-1).float()
            colour_hist = one_hot.sum(dim=1) / pad_mask.float().sum(
                dim=1, keepdim=True
            ).clamp(min=1.0)
            self._pending_history = {
                "task": t.detach(),
                "decisions": torch.cat([colour_hist, alpha.detach()], dim=-1),
            }
            if proposal is not None:
                self._pending_proposal = {
                    k: v.detach().clone() for k, v in proposal.items()
                }

        metadata: dict[str, Any] = {
            "logits_mem": logits_mem,
            "logits_rule": logits_rule,
            "logits_prop": logits_prop,
            "logits_guess": logits_guess,
            "prop_active": prop_active,
            "retrieval_scores": retrieval_info["scores"],
            "memory_strength": retrieval_info["strength"],
            "retrieval_head_attn": retrieval_info["head_attn"],
            "router_attn": router_attn,
            "proposal": proposal,
            "size_logits": size_logits,
            "fit": fit,
            "verify_ce": verify_ce,
            "verify_valid": verify_valid,
        }
        return logits, alpha, metadata
