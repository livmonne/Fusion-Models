"""Unit tests for the individual Fusion Model components.

Each test creates a component with small dimensions, feeds it a random
batch, and verifies that the output shapes and value ranges are correct.
The suite also contains regression tests for the architecture's key
training-dynamics guarantees:

* the history proposer receives task gradient (it is a routed pathway);
* rule commits are outcome-gated and deferred to ``apply_outcomes``;
* every pathway decodes from a residual base path (``x + correction``);
* padding never produces NaNs (attention masks keep self-connections).

These tests run on CPU and do not require any data files (except for
parquet tests which create a temporary file on the fly).
"""

from __future__ import annotations

import json
import os
import pickle
import tempfile

import torch

from fusion_model.decision import (
    NUM_PATHWAYS,
    PROP_INDEX,
    DecisionRouter,
)
from fusion_model.guess import GuessComponent
from fusion_model.loss import FusionLoss
from fusion_model.memory import MultiHeadMemoryCrossAttention, RuleMemory
from fusion_model.model import FusionModel
from fusion_model.rule_engine import RuleGenerator
from tasks.arc import ParquetARCDataset, arc_collate_fn

# Shared test dimensions — kept small so tests run in milliseconds.
BATCH = 4
EMBED = 64
NUM_COLOURS = 10
MAX_GRID = 4  # small grid for fast tests
MAX_CELLS = MAX_GRID * MAX_GRID
N_SLOTS = 8
RANK = 4


def _filled_generator(min_hist: int = 8) -> RuleGenerator:
    """Build a RuleGenerator whose history buffer already has entries."""
    gen = RuleGenerator(
        embed_dim=EMBED, num_colours=NUM_COLOURS, rank=RANK,
        history_size=64, min_history=min_hist,
    )
    for _ in range(min_hist // BATCH + 1):
        t_fill = torch.randn(BATCH, EMBED)
        decisions = torch.rand(BATCH, gen.decision_feature_dim)
        outcomes = torch.rand(BATCH)
        gen.update_history(t_fill, decisions, outcomes)
    return gen


class TestRuleMemory:
    """Tests for :class:`fusion_model.memory.RuleMemory`."""

    def test_output_shapes(self) -> None:
        """Logits, repr, and retrieval info must have the expected shapes."""
        num_heads = 4
        mem = RuleMemory(
            embed_dim=EMBED, num_colours=NUM_COLOURS, num_slots=N_SLOTS,
            rank=RANK, num_retrieval_heads=num_heads,
        )
        x = torch.randn(BATCH, MAX_CELLS, EMBED)
        t = torch.randn(BATCH, EMBED)
        logits, mem_repr, info = mem(x, t)

        assert logits.shape == (BATCH, MAX_CELLS, NUM_COLOURS)
        assert mem_repr.shape == (BATCH, EMBED)
        assert info["scores"].shape == (BATCH, N_SLOTS)
        assert info["strength"].shape == (N_SLOTS,)
        assert info["head_attn"].shape == (BATCH, num_heads, N_SLOTS)

    def test_base_path_decoding(self) -> None:
        """Logits must not collapse to zero when corrections are zero.

        ``B`` is zero-initialised, so all corrections start as no-ops; the
        residual base path (``head(norm(x + 0))``) must still decode real
        information from ``x``.  (Regression test: an earlier revision
        decoded the correction alone, forcing everything through the
        rank-r bottleneck and producing exactly-zero logits at init.)
        """
        mem = RuleMemory(embed_dim=EMBED, num_colours=NUM_COLOURS, num_slots=N_SLOTS, rank=RANK)
        x = torch.randn(BATCH, MAX_CELLS, EMBED)
        t = torch.randn(BATCH, EMBED)
        logits, _, _ = mem(x, t)
        assert not torch.allclose(logits, torch.zeros_like(logits))

    def test_scores_sum_to_one(self) -> None:
        """Strength-gated retrieval scores must be valid probabilities."""
        mem = RuleMemory(embed_dim=EMBED, num_colours=NUM_COLOURS, num_slots=N_SLOTS, rank=RANK)
        x = torch.randn(BATCH, MAX_CELLS, EMBED)
        t = torch.randn(BATCH, EMBED)
        _, _, info = mem(x, t)
        sums = info["scores"].sum(dim=-1)
        assert torch.allclose(sums, torch.ones(BATCH), atol=1e-5)

    def test_strength_in_unit_interval(self) -> None:
        """Memory strength values must lie in [0, 1]."""
        mem = RuleMemory(embed_dim=EMBED, num_colours=NUM_COLOURS, num_slots=N_SLOTS, rank=RANK)
        strength = mem.get_strength()
        assert (strength >= 0.0).all() and (strength <= 1.0).all()

    def test_frequency_decays_over_time(self) -> None:
        """Frequency should decay toward zero when no retrieval happens."""
        mem = RuleMemory(
            embed_dim=EMBED, num_colours=NUM_COLOURS, num_slots=N_SLOTS, rank=RANK,
            prune_every_n_steps=0,
        )
        mem.train()
        mem.frequency.fill_(0.8)
        mem.steps_since_activation.zero_()

        with torch.no_grad():
            mem.reinforce_rate_logit.fill_(-20.0)

        x = torch.randn(BATCH, MAX_CELLS, EMBED)
        t = torch.randn(BATCH, EMBED)
        for _ in range(100):
            mem(x, t)

        assert (mem.frequency < 0.8).all(), (
            "Frequency should decrease when reinforcement is suppressed"
        )

    def test_update_state_false_freezes_buffers(self) -> None:
        """The verification pass must not advance memory dynamics."""
        mem = RuleMemory(
            embed_dim=EMBED, num_colours=NUM_COLOURS, num_slots=N_SLOTS, rank=RANK,
        )
        mem.train()
        freq_before = mem.frequency.clone()
        step_before = mem.step_counter.clone()

        x = torch.randn(BATCH, MAX_CELLS, EMBED)
        t = torch.randn(BATCH, EMBED)
        mem(x, t, update_state=False)

        assert torch.equal(mem.frequency, freq_before)
        assert torch.equal(mem.step_counter, step_before)

    def test_prune_weak_slots_resets_parameters(self) -> None:
        """Pruning should recycle dead slots and reset their strength."""
        mem = RuleMemory(embed_dim=EMBED, num_colours=NUM_COLOURS, num_slots=N_SLOTS, rank=RANK)
        mem.frequency.fill_(0.01)
        mem.steps_since_activation.fill_(10000.0)

        n_pruned = mem.prune_weak_slots(threshold=0.5)
        assert n_pruned == N_SLOTS
        assert (mem.frequency == 0.5).all()
        assert (mem.steps_since_activation == 0.0).all()
        # Recycled slots restart as no-op corrections (B == 0).
        assert torch.equal(mem.B.data, torch.zeros_like(mem.B.data))

    def test_commit_rule_resets_strength(self) -> None:
        """Committing a rule should give the target slot a warm start."""
        mem = RuleMemory(embed_dim=EMBED, num_colours=NUM_COLOURS, num_slots=N_SLOTS, rank=RANK)
        mem.frequency.fill_(0.0)
        mem.steps_since_activation.fill_(999.0)

        key = torch.randn(EMBED)
        A = torch.randn(EMBED, RANK)
        B = torch.randn(RANK, EMBED)
        mem.commit_rule(0, key, A, B, commit_weight=0.7)

        assert torch.allclose(mem.frequency[0], torch.tensor(0.7), atol=1e-5)
        assert mem.steps_since_activation[0].item() == 0.0

    def test_get_weakest_slot_prefers_low_strength(self) -> None:
        """get_weakest_slot should return the slot with lowest strength."""
        mem = RuleMemory(embed_dim=EMBED, num_colours=NUM_COLOURS, num_slots=N_SLOTS, rank=RANK)
        mem.frequency.fill_(0.9)
        mem.steps_since_activation.zero_()
        mem.frequency[3] = 0.01
        mem.steps_since_activation[3] = 10000.0

        assert mem.get_weakest_slot() == 3

    def test_learnable_rates_are_parameters(self) -> None:
        """Decay rate, reinforcement rate, and recency half-life must be nn.Parameters."""
        mem = RuleMemory(embed_dim=EMBED, num_colours=NUM_COLOURS, num_slots=N_SLOTS, rank=RANK)
        param_names = {name for name, _ in mem.named_parameters()}
        assert "decay_rate_logit" in param_names
        assert "reinforce_rate_logit" in param_names
        assert "recency_halflife_log" in param_names


class TestMultiHeadMemoryCrossAttention:
    """Tests for :class:`fusion_model.memory.MultiHeadMemoryCrossAttention`."""

    def test_output_shapes(self) -> None:
        """Scores, context, and head_attn must have the expected shapes."""
        num_heads = 4
        attn = MultiHeadMemoryCrossAttention(embed_dim=EMBED, num_heads=num_heads)
        t = torch.randn(BATCH, EMBED)
        keys = torch.randn(N_SLOTS, EMBED)
        scores, context, head_attn = attn(t, keys)

        assert scores.shape == (BATCH, N_SLOTS)
        assert context.shape == (BATCH, EMBED)
        assert head_attn.shape == (BATCH, num_heads, N_SLOTS)

    def test_scores_sum_to_one(self) -> None:
        """Combined retrieval scores must form valid probability distributions."""
        attn = MultiHeadMemoryCrossAttention(embed_dim=EMBED, num_heads=4)
        t = torch.randn(BATCH, EMBED)
        keys = torch.randn(N_SLOTS, EMBED)
        scores, _, _ = attn(t, keys)
        sums = scores.sum(dim=-1)
        assert torch.allclose(sums, torch.ones(BATCH), atol=1e-5)

    def test_per_head_attn_sums_to_one(self) -> None:
        """Each head's attention distribution must sum to 1 over slots."""
        num_heads = 4
        attn = MultiHeadMemoryCrossAttention(embed_dim=EMBED, num_heads=num_heads)
        t = torch.randn(BATCH, EMBED)
        keys = torch.randn(N_SLOTS, EMBED)
        _, _, head_attn = attn(t, keys)
        sums = head_attn.sum(dim=-1)  # (B, H)
        assert torch.allclose(sums, torch.ones(BATCH, num_heads), atol=1e-5)

    def test_gradients_flow_to_projections(self) -> None:
        """Gradients must reach Q/K/V projections and the head_combine param."""
        attn = MultiHeadMemoryCrossAttention(embed_dim=EMBED, num_heads=4)
        t = torch.randn(BATCH, EMBED)
        keys = torch.randn(N_SLOTS, EMBED)
        scores, context, _ = attn(t, keys)
        loss = scores.sum() + context.sum()
        loss.backward()

        assert attn.q_proj.weight.grad is not None
        assert attn.k_proj.weight.grad is not None
        assert attn.v_proj.weight.grad is not None
        assert attn.head_combine.grad is not None


class TestRuleGenerator:
    """Tests for :class:`fusion_model.rule_engine.RuleGenerator`."""

    def test_output_shapes(self) -> None:
        """Generator logits and repr must have the expected shapes."""
        gen = RuleGenerator(embed_dim=EMBED, num_colours=NUM_COLOURS, rank=RANK)
        x = torch.randn(BATCH, MAX_CELLS, EMBED)
        t = torch.randn(BATCH, EMBED)
        out = gen(x, t)

        assert out["logits_rule"].shape == (BATCH, MAX_CELLS, NUM_COLOURS)
        assert out["rule_repr"].shape == (BATCH, EMBED)
        # Cold history buffer: proposal pathway inactive.
        assert out["prop_active"] is False
        assert out["logits_prop"] is None
        assert out["prop_repr"].shape == (BATCH, EMBED)

    def test_proposal_pathway_active_after_history_fill(self) -> None:
        """With enough history the proposal pathway must produce logits."""
        gen = _filled_generator(min_hist=8)
        x = torch.randn(BATCH, MAX_CELLS, EMBED)
        t = torch.randn(BATCH, EMBED)
        out = gen(x, t)

        assert out["prop_active"] is True
        assert out["logits_prop"].shape == (BATCH, MAX_CELLS, NUM_COLOURS)
        proposal = out["proposal"]
        assert proposal is not None
        assert proposal["key"].shape == (BATCH, EMBED)
        assert proposal["A"].shape == (BATCH, EMBED, RANK)
        assert proposal["B"].shape == (BATCH, RANK, EMBED)
        assert proposal["commit_weight"].shape == (BATCH, 1)
        assert (proposal["commit_weight"] >= 0.0).all()
        assert (proposal["commit_weight"] <= 1.0).all()

    def test_proposal_none_before_min_history(self) -> None:
        """propose_rule must return None when history is below min_history."""
        gen = RuleGenerator(embed_dim=EMBED, num_colours=NUM_COLOURS, rank=RANK, min_history=64)
        assert gen.propose_rule(BATCH) is None

    def test_proposer_receives_task_gradient(self) -> None:
        """The proposal pathway's logits must train the proposer's rule content.

        Regression test for the gradient-isolation bug: previously the
        proposed A/B matrices were only consumed by a ``no_grad`` commit,
        so the entire history-attention pipeline received no task signal.
        Now the proposal is a routed pathway, so a loss on its logits must
        reach ``rule_proj`` (including the A/B rows) and the synthesis
        attention stack.
        """
        gen = _filled_generator(min_hist=8)
        x = torch.randn(BATCH, MAX_CELLS, EMBED)
        t = torch.randn(BATCH, EMBED)
        out = gen(x, t)

        loss = out["logits_prop"].sum()
        loss.backward()

        assert gen.rule_proj.weight.grad is not None
        # The A/B-producing rows (everything after the key block) must see
        # non-zero gradient.
        ab_rows = gen.rule_proj.weight.grad[EMBED:]
        assert ab_rows.abs().sum() > 0
        assert gen.synthesis_query.grad is not None

    def test_should_commit_outcome_gate(self) -> None:
        """Commits require beating the EMA of recent proposal losses."""
        gen = RuleGenerator(embed_dim=EMBED, num_colours=NUM_COLOURS, rank=RANK)

        # First call only initialises the EMA.
        assert gen.should_commit(torch.full((BATCH,), 1.0)) is None
        # A clearly better batch must be approved.
        assert gen.should_commit(torch.full((BATCH,), 0.1)) is not None
        # A clearly worse batch must be rejected.
        assert gen.should_commit(torch.full((BATCH,), 10.0)) is None


class TestGuessComponent:
    """Tests for :class:`fusion_model.guess.GuessComponent`."""

    def test_output_shapes(self) -> None:
        """Output logits and pooled representation must have expected shapes."""
        guess = GuessComponent(embed_dim=EMBED, num_colours=NUM_COLOURS)
        x = torch.randn(BATCH, MAX_CELLS, EMBED)
        t = torch.randn(BATCH, EMBED)
        logits, pooled = guess(x, t, grid_h=MAX_GRID, grid_w=MAX_GRID)
        assert logits.shape == (BATCH, MAX_CELLS, NUM_COLOURS)
        assert pooled.shape == (BATCH, EMBED)

    def test_local_attention_alternation(self) -> None:
        """Even layers should use local attention, odd layers global."""
        guess = GuessComponent(
            embed_dim=EMBED, num_colours=NUM_COLOURS, num_layers=4, window_size=3,
        )
        x = torch.randn(BATCH, MAX_CELLS, EMBED)
        t = torch.randn(BATCH, EMBED)
        logits, pooled = guess(x, t, grid_h=MAX_GRID, grid_w=MAX_GRID)
        assert logits.shape == (BATCH, MAX_CELLS, NUM_COLOURS)
        assert pooled.shape == (BATCH, EMBED)

    def test_rectangular_grid(self) -> None:
        """GuessComponent should handle non-square grids (H != W)."""
        H, W = 3, MAX_GRID
        guess = GuessComponent(embed_dim=EMBED, num_colours=NUM_COLOURS)
        x = torch.randn(BATCH, H * W, EMBED)
        t = torch.randn(BATCH, EMBED)
        logits, pooled = guess(x, t, grid_h=H, grid_w=W)
        assert logits.shape == (BATCH, H * W, NUM_COLOURS)
        assert pooled.shape == (BATCH, EMBED)

    def test_heavy_padding_no_nan(self) -> None:
        """Padding masks must never produce NaNs (self-attention kept)."""
        guess = GuessComponent(embed_dim=EMBED, num_colours=NUM_COLOURS, window_size=3)
        x = torch.randn(BATCH, MAX_CELLS, EMBED)
        t = torch.randn(BATCH, EMBED)
        pad_mask = torch.zeros(BATCH, MAX_CELLS, dtype=torch.bool)
        pad_mask[:, :2] = True  # only two real cells; whole windows are PAD
        logits, pooled = guess(x, t, grid_h=MAX_GRID, grid_w=MAX_GRID, pad_mask=pad_mask)
        assert torch.isfinite(logits).all()
        assert torch.isfinite(pooled).all()


class TestDecisionRouter:
    """Tests for :class:`fusion_model.decision.DecisionRouter`."""

    def test_alpha_shape_and_sum(self) -> None:
        """Routing weights must be (batch, NUM_PATHWAYS) and sum to 1."""
        router = DecisionRouter(embed_dim=EMBED)
        h = torch.randn(BATCH, EMBED)
        reprs = [torch.randn(BATCH, EMBED) for _ in range(NUM_PATHWAYS)]
        alpha, attn_weights = router(h, reprs)

        assert alpha.shape == (BATCH, NUM_PATHWAYS)
        assert torch.allclose(alpha.sum(dim=-1), torch.ones(BATCH), atol=1e-5)

    def test_attn_weights_shape(self) -> None:
        """Per-head attention weights must have shape (batch, heads, NUM_PATHWAYS)."""
        num_heads = 4
        router = DecisionRouter(embed_dim=EMBED, num_heads=num_heads)
        h = torch.randn(BATCH, EMBED)
        reprs = [torch.randn(BATCH, EMBED) for _ in range(NUM_PATHWAYS)]
        _alpha, attn_weights = router(h, reprs)

        assert attn_weights.shape == (BATCH, num_heads, NUM_PATHWAYS)
        sums = attn_weights.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_inactive_proposal_masked(self) -> None:
        """When the proposal pathway is inactive its weight must be zero."""
        router = DecisionRouter(embed_dim=EMBED)
        h = torch.randn(BATCH, EMBED)
        reprs = [torch.randn(BATCH, EMBED) for _ in range(NUM_PATHWAYS)]
        alpha, _ = router(h, reprs, prop_active=False)

        assert torch.allclose(alpha[:, PROP_INDEX], torch.zeros(BATCH))
        assert torch.allclose(alpha.sum(dim=-1), torch.ones(BATCH), atol=1e-5)

    def test_fit_bias_shifts_routing(self) -> None:
        """A strong measured fit must increase that pathway's weight."""
        router = DecisionRouter(embed_dim=EMBED)
        router.eval()
        h = torch.randn(BATCH, EMBED)
        reprs = [torch.randn(BATCH, EMBED) for _ in range(NUM_PATHWAYS)]

        alpha_base, _ = router(h, reprs)
        fit = torch.zeros(BATCH, NUM_PATHWAYS)
        fit[:, 0] = 10.0  # memory pathway nailed the held-out demo
        alpha_fit, _ = router(h, reprs, fit=fit)

        assert (alpha_fit[:, 0] > alpha_base[:, 0]).all()


class TestFusionLoss:
    """Tests for :class:`fusion_model.loss.FusionLoss`."""

    def _alphas(self) -> torch.Tensor:
        return torch.softmax(torch.randn(BATCH, NUM_PATHWAYS), dim=-1)

    def test_loss_is_scalar(self) -> None:
        """Total loss must be a scalar tensor."""
        criterion = FusionLoss(pad_value=-1)
        logits = torch.randn(BATCH, MAX_CELLS, NUM_COLOURS)
        targets = torch.randint(0, NUM_COLOURS, (BATCH, MAX_CELLS))
        scores = torch.softmax(torch.randn(BATCH, N_SLOTS), dim=-1)

        loss, loss_dict = criterion(logits, targets, self._alphas(), scores)
        assert loss.shape == ()
        assert "total" in loss_dict

    def test_aux_loss_uses_per_cell_ce(self) -> None:
        """Auxiliary loss must correctly compute per-cell CE for each pathway."""
        criterion = FusionLoss(pad_value=-1)
        logits = torch.randn(BATCH, MAX_CELLS, NUM_COLOURS)
        targets = torch.randint(0, NUM_COLOURS, (BATCH, MAX_CELLS))
        targets[:, -4:] = -1  # mark some cells as padding
        scores = torch.softmax(torch.randn(BATCH, N_SLOTS), dim=-1)

        metadata = {
            "logits_mem": torch.randn(BATCH, MAX_CELLS, NUM_COLOURS),
            "logits_rule": torch.randn(BATCH, MAX_CELLS, NUM_COLOURS),
            "logits_prop": torch.randn(BATCH, MAX_CELLS, NUM_COLOURS),
            "logits_guess": torch.randn(BATCH, MAX_CELLS, NUM_COLOURS),
        }

        loss, loss_dict = criterion(logits, targets, self._alphas(), scores, metadata=metadata)
        assert loss.shape == ()
        assert loss_dict["aux"] > 0.0
        assert loss_dict["aux_prop"] > 0.0

    def test_size_loss(self) -> None:
        """Size targets must add a size CE term."""
        criterion = FusionLoss(pad_value=-1)
        logits = torch.randn(BATCH, MAX_CELLS, NUM_COLOURS)
        targets = torch.randint(0, NUM_COLOURS, (BATCH, MAX_CELLS))
        scores = torch.softmax(torch.randn(BATCH, N_SLOTS), dim=-1)

        metadata = {"size_logits": torch.randn(BATCH, 2, MAX_GRID)}
        size_targets = torch.randint(1, MAX_GRID + 1, (BATCH, 2))

        _, loss_dict = criterion(
            logits, targets, self._alphas(), scores,
            metadata=metadata, size_targets=size_targets,
        )
        assert loss_dict["size"] > 0.0

    def test_verify_loss(self) -> None:
        """Verification CEs must contribute when valid samples exist."""
        criterion = FusionLoss(pad_value=-1)
        logits = torch.randn(BATCH, MAX_CELLS, NUM_COLOURS)
        targets = torch.randint(0, NUM_COLOURS, (BATCH, MAX_CELLS))
        scores = torch.softmax(torch.randn(BATCH, N_SLOTS), dim=-1)

        metadata = {
            "verify_ce": torch.rand(BATCH, NUM_PATHWAYS) + 0.5,
            "verify_valid": torch.tensor([True, True, False, True]),
            "prop_active": False,
        }
        _, loss_dict = criterion(logits, targets, self._alphas(), scores, metadata=metadata)
        assert loss_dict["verify"] > 0.0


class TestFusionModel:
    """Integration tests for :class:`fusion_model.model.FusionModel`."""

    @staticmethod
    def _make_model(**overrides: object) -> FusionModel:
        kwargs: dict[str, object] = dict(
            embed_dim=EMBED,
            num_colours=NUM_COLOURS,
            max_grid_size=MAX_GRID,
            num_encoder_layers=1,
            num_cross_attn_layers=1,
            num_attn_heads=4,
            num_rule_slots=N_SLOTS,
            rule_rank=RANK,
            max_demos=3,
        )
        kwargs.update(overrides)
        return FusionModel(**kwargs)  # type: ignore[arg-type]

    @staticmethod
    def _make_batch(
        max_demos: int = 3, H: int = MAX_GRID, W: int = MAX_GRID
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        demo_inputs = torch.randint(0, NUM_COLOURS, (BATCH, max_demos, H, W))
        demo_outputs = torch.randint(0, NUM_COLOURS, (BATCH, max_demos, H, W))
        demo_mask = torch.ones(BATCH, max_demos, dtype=torch.bool)
        test_input = torch.randint(0, NUM_COLOURS, (BATCH, H, W))
        return demo_inputs, demo_outputs, demo_mask, test_input

    def test_forward_shapes(self) -> None:
        """Full forward pass must produce correctly shaped outputs."""
        model = self._make_model()
        logits, alpha, meta = model(*self._make_batch())

        assert logits.shape == (BATCH, MAX_CELLS, NUM_COLOURS)
        assert alpha.shape == (BATCH, NUM_PATHWAYS)
        assert meta["logits_mem"].shape == (BATCH, MAX_CELLS, NUM_COLOURS)
        assert meta["logits_rule"].shape == (BATCH, MAX_CELLS, NUM_COLOURS)
        assert meta["logits_guess"].shape == (BATCH, MAX_CELLS, NUM_COLOURS)
        assert meta["memory_strength"].shape == (N_SLOTS,)
        assert meta["size_logits"].shape == (BATCH, 2, MAX_GRID)
        assert "retrieval_head_attn" in meta

    def test_forward_with_padding(self) -> None:
        """Forward pass should handle padded grids (PAD_VALUE = -1) without NaNs."""
        model = self._make_model()
        demo_inputs, demo_outputs, _, test_input = self._make_batch()
        demo_mask = torch.tensor([[True, True, False]] * BATCH)
        demo_inputs[:, 2] = -1
        demo_outputs[:, 2] = -1
        test_input[:, 3:, :] = -1  # pad bottom rows

        logits, alpha, meta = model(demo_inputs, demo_outputs, demo_mask, test_input)

        assert logits.shape == (BATCH, MAX_CELLS, NUM_COLOURS)
        assert alpha.shape == (BATCH, NUM_PATHWAYS)
        assert torch.isfinite(logits).all()
        assert torch.isfinite(alpha).all()

    def test_verification_produces_fit(self) -> None:
        """With >= 2 demos, leave-one-out verification must produce fit signals."""
        model = self._make_model()
        model.train()
        _, _, meta = model(*self._make_batch())

        assert meta["fit"] is not None
        assert meta["fit"].shape == (BATCH, NUM_PATHWAYS)
        assert meta["verify_ce"] is not None
        assert meta["verify_ce"].shape == (BATCH, NUM_PATHWAYS)
        assert bool(meta["verify_valid"].all())
        assert torch.isfinite(meta["fit"]).all()

    def test_verification_disabled(self) -> None:
        """verify_demos=False must skip verification entirely."""
        model = self._make_model(verify_demos=False)
        model.train()
        _, _, meta = model(*self._make_batch())
        assert meta["fit"] is None
        assert meta["verify_ce"] is None

    def test_apply_outcomes_updates_history_and_commits(self) -> None:
        """History fills via apply_outcomes; commits are outcome-gated."""
        model = self._make_model()
        model.rule_gen.min_history = BATCH  # activate proposals quickly
        model.train()
        batch = self._make_batch()

        # Step 1: no proposal yet (empty history); history gets filled.
        _, _, meta = model(*batch)
        assert meta["prop_active"] is False
        model.apply_outcomes(torch.rand(BATCH), None, allow_commit=True)
        assert int(model.rule_gen.history_count.item()) == BATCH

        # Step 2: proposal active; first prop outcome initialises the EMA.
        _, _, meta = model(*batch)
        assert meta["prop_active"] is True
        info = model.apply_outcomes(
            torch.rand(BATCH), torch.full((BATCH,), 1.0), allow_commit=True,
        )
        assert info["committed"] is False

        # Step 3: clearly better proposal outcome → commit approved.
        _, _, meta = model(*batch)
        info = model.apply_outcomes(
            torch.rand(BATCH), torch.full((BATCH,), 0.01), allow_commit=True,
        )
        assert info["committed"] is True
        assert info["commit_weight"] > 0.0

    def test_commit_deferred_while_accumulating(self) -> None:
        """allow_commit=False must never write to the memory bank."""
        model = self._make_model()
        model.rule_gen.min_history = BATCH
        model.train()
        batch = self._make_batch()

        model(*batch)
        model.apply_outcomes(torch.rand(BATCH), None, allow_commit=True)
        model(*batch)
        keys_before = model.memory.keys.data.clone()
        info = model.apply_outcomes(
            torch.rand(BATCH), torch.full((BATCH,), 0.0), allow_commit=False,
        )
        assert info["committed"] is False
        assert torch.equal(model.memory.keys.data, keys_before)


class TestHebbianGradientFlow:
    """Tests verifying that decay/reinforce rate parameters receive gradients."""

    def test_rate_params_receive_gradients_with_divergent_freq(self) -> None:
        """After frequencies diverge, decay/reinforce logits must have non-zero grad."""
        mem = RuleMemory(
            embed_dim=EMBED, num_colours=NUM_COLOURS, num_slots=N_SLOTS, rank=RANK,
            prune_every_n_steps=0,
        )
        mem.train()

        # Set divergent frequencies to simulate mid-training state.
        with torch.no_grad():
            mem.frequency.copy_(torch.tensor([0.9, 0.7, 0.3, 0.1, 0.8, 0.05, 0.6, 0.4]))
            mem.steps_since_activation.copy_(
                torch.tensor([0.0, 50.0, 200.0, 500.0, 10.0, 800.0, 30.0, 150.0])
            )

        x = torch.randn(BATCH, MAX_CELLS, EMBED)
        t = torch.randn(BATCH, EMBED)
        logits, _, info = mem(x, t)

        # Use a loss that depends on both logits and strength.
        loss = logits.sum() + 0.001 * (info["strength"].mean() - 0.5) ** 2
        loss.backward()

        assert mem.decay_rate_logit.grad is not None
        assert mem.reinforce_rate_logit.grad is not None
        assert mem.recency_halflife_log.grad is not None

    def test_strength_returned_with_grad(self) -> None:
        """Returned strength must be part of the computation graph during training."""
        mem = RuleMemory(
            embed_dim=EMBED, num_colours=NUM_COLOURS, num_slots=N_SLOTS, rank=RANK,
        )
        mem.train()

        x = torch.randn(BATCH, MAX_CELLS, EMBED)
        t = torch.randn(BATCH, EMBED)
        _, _, info = mem(x, t)

        assert info["strength"].requires_grad, (
            "strength should be differentiable during training"
        )

    def test_no_frequency_update_at_eval(self) -> None:
        """Frequency buffer should remain unchanged during eval forward passes."""
        mem = RuleMemory(
            embed_dim=EMBED, num_colours=NUM_COLOURS, num_slots=N_SLOTS, rank=RANK,
        )
        mem.eval()
        freq_before = mem.frequency.clone()

        x = torch.randn(BATCH, MAX_CELLS, EMBED)
        t = torch.randn(BATCH, EMBED)
        mem(x, t)

        assert torch.equal(mem.frequency, freq_before), (
            "Frequency buffer should not change during eval"
        )

    def test_frequency_buffer_updates_during_training(self) -> None:
        """self.frequency buffer should be updated after a forward pass in train mode."""
        mem = RuleMemory(
            embed_dim=EMBED, num_colours=NUM_COLOURS, num_slots=N_SLOTS, rank=RANK,
            prune_every_n_steps=0,
        )
        mem.train()
        freq_before = mem.frequency.clone()

        x = torch.randn(BATCH, MAX_CELLS, EMBED)
        t = torch.randn(BATCH, EMBED)
        mem(x, t)

        assert not torch.equal(mem.frequency, freq_before), (
            "Frequency buffer should change after a training forward pass"
        )


# ── Parquet dataset tests ────────────────────────────────────────────────────


def _make_parquet_file(tmp_dir: str, n_tasks: int = 5) -> str:
    """Create a minimal parquet file with synthetic ARC tasks."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    ids = []
    tasks = []
    for i in range(n_tasks):
        task = {
            "train": [
                {"input": [[i, 0], [0, i]], "output": [[0, i], [i, 0]]},
            ],
            "test": [
                {"input": [[i, i], [0, 0]], "output": [[0, 0], [i, i]]},
            ],
        }
        ids.append(f"task_{i:04d}")
        tasks.append(json.dumps(task))

    table = pa.table({"id": ids, "task": tasks})
    path = os.path.join(tmp_dir, "test_data.parquet")
    pq.write_table(table, path)
    return path


class TestParquetARCDataset:
    """Tests for :class:`tasks.arc.ParquetARCDataset`."""

    def test_load_and_length(self) -> None:
        """Dataset length must match the number of test pairs across all tasks."""
        with tempfile.TemporaryDirectory() as tmp:
            path = _make_parquet_file(tmp, n_tasks=5)
            ds = ParquetARCDataset([path])
            # Each task has 1 test pair → 5 samples.
            assert len(ds) == 5

    def test_getitem_returns_expected_keys(self) -> None:
        """Each sample must contain the standard ARC tensor dict."""
        with tempfile.TemporaryDirectory() as tmp:
            path = _make_parquet_file(tmp, n_tasks=3)
            ds = ParquetARCDataset([path], max_grid_size=4)
            sample = ds[0]

            expected_keys = {
                "demo_inputs", "demo_outputs", "demo_mask",
                "test_input", "test_output", "input_size", "output_size",
                "grid_dims",
            }
            assert set(sample.keys()) == expected_keys

    def test_getitem_shapes(self) -> None:
        """Tensor shapes must match the per-sample grid dims and demo count."""
        max_demos = 3
        with tempfile.TemporaryDirectory() as tmp:
            path = _make_parquet_file(tmp, n_tasks=2)
            ds = ParquetARCDataset([path], max_grid_size=4, max_demos=max_demos)
            sample = ds[0]

            # Synthetic grids are 2×2, so sample_h=sample_w=2.
            sh, sw = sample["grid_dims"].tolist()
            assert sample["demo_inputs"].shape == (max_demos, sh, sw)
            assert sample["demo_outputs"].shape == (max_demos, sh, sw)
            assert sample["demo_mask"].shape == (max_demos,)
            assert sample["test_input"].shape == (sh, sw)
            assert sample["test_output"].shape == (sh, sw)
            assert sample["input_size"].shape == (2,)

    def test_max_samples_cap(self) -> None:
        """max_samples should limit the total number of indexed samples."""
        with tempfile.TemporaryDirectory() as tmp:
            path = _make_parquet_file(tmp, n_tasks=10)
            ds = ParquetARCDataset([path], max_samples=3)
            assert len(ds) == 3

    def test_multiple_parquet_files(self) -> None:
        """Dataset should load from multiple parquet files."""
        with tempfile.TemporaryDirectory() as tmp:
            path1 = _make_parquet_file(tmp, n_tasks=3)
            # Create a second file with a different name.
            path2 = os.path.join(tmp, "test_data2.parquet")
            import shutil
            shutil.copy(path1, path2)

            ds = ParquetARCDataset([path1, path2])
            assert len(ds) == 6  # 3 + 3

    def test_multi_test_pair_expansion(self) -> None:
        """Tasks with multiple test pairs should be expanded into separate samples."""
        import pyarrow as pa
        import pyarrow.parquet as pq

        with tempfile.TemporaryDirectory() as tmp:
            task = {
                "train": [{"input": [[1]], "output": [[2]]}],
                "test": [
                    {"input": [[3]], "output": [[4]]},
                    {"input": [[5]], "output": [[6]]},
                ],
            }
            table = pa.table({"id": ["multi"], "task": [json.dumps(task)]})
            path = os.path.join(tmp, "multi.parquet")
            pq.write_table(table, path)

            ds = ParquetARCDataset([path], max_grid_size=2)
            assert len(ds) == 2

            # Verify the two samples have different test inputs.
            s0 = ds[0]
            s1 = ds[1]
            assert not torch.equal(s0["test_input"], s1["test_input"])

    def test_worker_pickling_drops_table_cache(self) -> None:
        """Pickling (spawned DataLoader workers) must not copy open tables."""
        with tempfile.TemporaryDirectory() as tmp:
            path = _make_parquet_file(tmp, n_tasks=3)
            ds = ParquetARCDataset([path])
            ds[0]  # populate the per-process table cache
            assert ds._tables

            clone = pickle.loads(pickle.dumps(ds))
            assert clone._tables == {}
            # The clone must still be able to load samples (lazy re-open).
            sample = clone[0]
            assert sample["test_input"].shape == (2, 2)


# ── Dynamic batch padding tests ─────────────────────────────────────────────


class TestArcCollateFn:
    """Tests for :func:`tasks.arc.arc_collate_fn`."""

    def test_pads_to_batch_max(self) -> None:
        """Collated tensors should be padded to the max dims in the batch."""
        # Sample A: 2×2 grids, Sample B: 3×4 grids.
        sample_a = {
            "demo_inputs": torch.ones(2, 2, 2, dtype=torch.long),
            "demo_outputs": torch.ones(2, 2, 2, dtype=torch.long),
            "demo_mask": torch.tensor([True, False]),
            "test_input": torch.ones(2, 2, dtype=torch.long),
            "test_output": torch.ones(2, 2, dtype=torch.long),
            "input_size": torch.tensor([2, 2]),
            "output_size": torch.tensor([2, 2]),
            "grid_dims": torch.tensor([2, 2]),
        }
        sample_b = {
            "demo_inputs": torch.full((2, 3, 4), 2, dtype=torch.long),
            "demo_outputs": torch.full((2, 3, 4), 2, dtype=torch.long),
            "demo_mask": torch.tensor([True, True]),
            "test_input": torch.full((3, 4), 2, dtype=torch.long),
            "test_output": torch.full((3, 4), 2, dtype=torch.long),
            "input_size": torch.tensor([3, 4]),
            "output_size": torch.tensor([3, 4]),
            "grid_dims": torch.tensor([3, 4]),
        }

        batch = arc_collate_fn([sample_a, sample_b])

        assert batch["demo_inputs"].shape == (2, 2, 3, 4)
        assert batch["test_input"].shape == (2, 3, 4)
        assert batch["test_output"].shape == (2, 3, 4)
        # Sample A's data in top-left, rest is PAD_VALUE (-1).
        assert batch["test_input"][0, 0, 0] == 1
        assert batch["test_input"][0, 2, 0] == -1  # padded row
        assert batch["test_input"][0, 0, 3] == -1  # padded col
        # Sample B fully occupies the 3×4 region.
        assert batch["test_input"][1, 2, 3] == 2


class TestNonSquareGridForward:
    """Test that the model handles non-square batch grids correctly."""

    def test_rectangular_grid(self) -> None:
        """Forward pass with H != W should produce correctly shaped outputs."""
        model = TestFusionModel._make_model()

        H, W = 3, MAX_GRID
        demo_inputs, demo_outputs, demo_mask, test_input = TestFusionModel._make_batch(
            max_demos=2, H=H, W=W,
        )
        # Re-make the model demo budget to match.
        logits, alpha, meta = model(demo_inputs, demo_outputs, demo_mask, test_input)

        assert logits.shape == (BATCH, H * W, NUM_COLOURS)
        assert alpha.shape == (BATCH, NUM_PATHWAYS)
        assert torch.isfinite(logits).all()
