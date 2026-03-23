"""Unit tests for the individual Fusion Model components.

Each test creates a component with small dimensions, feeds it a random
batch, and verifies that the output shapes and value ranges are correct.
These tests run on CPU and do not require any data files (except for
parquet tests which create a temporary file on the fly).
"""

from __future__ import annotations

import json
import os
import tempfile

import torch

from fusion_model.decision import DecisionRouter
from fusion_model.guess import GuessComponent
from fusion_model.loss import FusionLoss
from fusion_model.memory import RuleMemory
from fusion_model.model import FusionModel
from fusion_model.rule_engine import RuleGenerator
from tasks.arc import ARCDataset, ParquetARCDataset, arc_collate_fn, pad_grid

# Shared test dimensions — kept small so tests run in milliseconds.
BATCH = 4
EMBED = 64
NUM_COLOURS = 10
MAX_GRID = 4  # small grid for fast tests
MAX_CELLS = MAX_GRID * MAX_GRID
N_SLOTS = 8
RANK = 4


class TestRuleMemory:
    """Tests for :class:`fusion_model.memory.RuleMemory`."""

    def test_output_shapes(self) -> None:
        """Logits, repr, and retrieval info must have the expected shapes."""
        mem = RuleMemory(embed_dim=EMBED, num_colours=NUM_COLOURS, num_slots=N_SLOTS, rank=RANK)
        x = torch.randn(BATCH, MAX_CELLS, EMBED)
        h = torch.randn(BATCH, EMBED)
        logits, mem_repr, info = mem(x, h)

        assert logits.shape == (BATCH, MAX_CELLS, NUM_COLOURS)
        assert mem_repr.shape == (BATCH, EMBED)
        assert info["scores"].shape == (BATCH, N_SLOTS)
        assert info["strength"].shape == (N_SLOTS,)

    def test_scores_sum_to_one(self) -> None:
        """Strength-gated retrieval scores must be valid probabilities."""
        mem = RuleMemory(embed_dim=EMBED, num_colours=NUM_COLOURS, num_slots=N_SLOTS, rank=RANK)
        x = torch.randn(BATCH, MAX_CELLS, EMBED)
        h = torch.randn(BATCH, EMBED)
        _, _, info = mem(x, h)
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
        h = torch.randn(BATCH, EMBED)
        for _ in range(100):
            mem(x, h)

        assert (mem.frequency < 0.8).all(), (
            "Frequency should decrease when reinforcement is suppressed"
        )

    def test_prune_weak_slots_resets_parameters(self) -> None:
        """Pruning should recycle dead slots and reset their strength."""
        mem = RuleMemory(embed_dim=EMBED, num_colours=NUM_COLOURS, num_slots=N_SLOTS, rank=RANK)
        mem.frequency.fill_(0.01)
        mem.steps_since_activation.fill_(10000.0)

        n_pruned = mem.prune_weak_slots(threshold=0.5)
        assert n_pruned == N_SLOTS
        assert (mem.frequency == 0.5).all()
        assert (mem.steps_since_activation == 0.0).all()

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


class TestRuleGenerator:
    """Tests for :class:`fusion_model.rule_engine.RuleGenerator`."""

    def test_output_shapes(self) -> None:
        """Logits, confidence, and repr must have the expected shapes."""
        gen = RuleGenerator(embed_dim=EMBED, num_colours=NUM_COLOURS, rank=RANK)
        x = torch.randn(BATCH, MAX_CELLS, EMBED)
        h = torch.randn(BATCH, EMBED)
        logits, confidence, rule_repr, _proposal = gen(x, h)

        assert logits.shape == (BATCH, MAX_CELLS, NUM_COLOURS)
        assert confidence.shape == (BATCH, 1)
        assert rule_repr.shape == (BATCH, EMBED)

    def test_confidence_range(self) -> None:
        """Confidence must lie in [0, 1] (sigmoid output)."""
        gen = RuleGenerator(embed_dim=EMBED, num_colours=NUM_COLOURS, rank=RANK)
        x = torch.randn(BATCH, MAX_CELLS, EMBED)
        h = torch.randn(BATCH, EMBED)
        _, confidence, _, _proposal = gen(x, h)
        assert (confidence >= 0.0).all() and (confidence <= 1.0).all()

    def test_proposal_shapes_after_history_fill(self) -> None:
        """After enough history, propose_rule must return correctly shaped tensors."""
        min_hist = 16
        gen = RuleGenerator(
            embed_dim=EMBED, num_colours=NUM_COLOURS, rank=RANK,
            history_size=64, min_history=min_hist,
        )
        gen.train()

        for _ in range(min_hist // BATCH + 1):
            h_fill = torch.randn(BATCH, EMBED)
            decisions = torch.randint(0, gen.decision_vocab_size, (BATCH,))
            outcomes = torch.rand(BATCH)
            gen.update_history(h_fill, decisions, outcomes)

        proposal = gen.propose_rule(BATCH)

        assert proposal is not None
        assert proposal["key"].shape == (BATCH, EMBED)
        assert proposal["A"].shape == (BATCH, EMBED, RANK)
        assert proposal["B"].shape == (BATCH, RANK, EMBED)
        assert proposal["commit_weight"].shape == (BATCH, 1)
        assert (proposal["commit_weight"] >= 0.0).all() and (proposal["commit_weight"] <= 1.0).all()

    def test_proposal_none_before_min_history(self) -> None:
        """propose_rule must return None when history is below min_history."""
        gen = RuleGenerator(embed_dim=EMBED, num_colours=NUM_COLOURS, rank=RANK, min_history=64)
        assert gen.propose_rule(BATCH) is None


class TestGuessComponent:
    """Tests for :class:`fusion_model.guess.GuessComponent`."""

    def test_output_shapes(self) -> None:
        """Output logits and pooled representation must have expected shapes."""
        guess = GuessComponent(embed_dim=EMBED, num_colours=NUM_COLOURS)
        x = torch.randn(BATCH, MAX_CELLS, EMBED)
        logits, pooled = guess(x)
        assert logits.shape == (BATCH, MAX_CELLS, NUM_COLOURS)
        assert pooled.shape == (BATCH, EMBED)


class TestDecisionRouter:
    """Tests for :class:`fusion_model.decision.DecisionRouter`."""

    def test_alpha_shape_and_sum(self) -> None:
        """Routing weights must be (batch, 3) and sum to 1."""
        router = DecisionRouter(embed_dim=EMBED)
        h = torch.randn(BATCH, EMBED)
        mem_repr = torch.randn(BATCH, EMBED)
        rule_repr = torch.randn(BATCH, EMBED)
        guess_repr = torch.randn(BATCH, EMBED)
        alpha, attn_weights = router(h, mem_repr, rule_repr, guess_repr)

        assert alpha.shape == (BATCH, 3)
        assert torch.allclose(alpha.sum(dim=-1), torch.ones(BATCH), atol=1e-5)

    def test_attn_weights_shape(self) -> None:
        """Per-head attention weights must have shape (batch, num_heads, 3)."""
        num_heads = 4
        router = DecisionRouter(embed_dim=EMBED, num_heads=num_heads)
        h = torch.randn(BATCH, EMBED)
        mem_repr = torch.randn(BATCH, EMBED)
        rule_repr = torch.randn(BATCH, EMBED)
        guess_repr = torch.randn(BATCH, EMBED)
        _alpha, attn_weights = router(h, mem_repr, rule_repr, guess_repr)

        assert attn_weights.shape == (BATCH, num_heads, 3)
        sums = attn_weights.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_num_heads_configurable(self) -> None:
        """Router must work with different head counts."""
        for n_heads in (1, 2, 4):
            router = DecisionRouter(embed_dim=EMBED, num_heads=n_heads)
            h = torch.randn(BATCH, EMBED)
            reprs = [torch.randn(BATCH, EMBED) for _ in range(3)]
            alpha, attn = router(h, *reprs)
            assert alpha.shape == (BATCH, 3)
            assert attn.shape == (BATCH, n_heads, 3)


class TestFusionLoss:
    """Tests for :class:`fusion_model.loss.FusionLoss`."""

    def test_loss_is_scalar(self) -> None:
        """Total loss must be a scalar tensor."""
        criterion = FusionLoss(pad_value=-1)
        logits = torch.randn(BATCH, MAX_CELLS, NUM_COLOURS)
        targets = torch.randint(0, NUM_COLOURS, (BATCH, MAX_CELLS))
        alphas = torch.softmax(torch.randn(BATCH, 3), dim=-1)
        scores = torch.softmax(torch.randn(BATCH, N_SLOTS), dim=-1)

        loss, loss_dict = criterion(logits, targets, alphas, scores)
        assert loss.shape == ()
        assert "total" in loss_dict

    def test_aux_loss_uses_per_cell_ce(self) -> None:
        """Auxiliary loss must correctly compute per-cell CE for each pathway."""
        criterion = FusionLoss(pad_value=-1)
        logits = torch.randn(BATCH, MAX_CELLS, NUM_COLOURS)
        targets = torch.randint(0, NUM_COLOURS, (BATCH, MAX_CELLS))
        # Mark some cells as padding.
        targets[:, -4:] = -1
        alphas = torch.softmax(torch.randn(BATCH, 3), dim=-1)
        scores = torch.softmax(torch.randn(BATCH, N_SLOTS), dim=-1)

        metadata = {
            "logits_mem": torch.randn(BATCH, MAX_CELLS, NUM_COLOURS),
            "logits_rule": torch.randn(BATCH, MAX_CELLS, NUM_COLOURS),
            "logits_guess": torch.randn(BATCH, MAX_CELLS, NUM_COLOURS),
        }

        loss, loss_dict = criterion(logits, targets, alphas, scores, metadata=metadata)
        assert loss.shape == ()
        assert loss_dict["aux"] > 0.0


class TestFusionModel:
    """Integration tests for :class:`fusion_model.model.FusionModel`."""

    def test_forward_shapes(self) -> None:
        """Full forward pass must produce correctly shaped outputs."""
        model = FusionModel(
            embed_dim=EMBED,
            num_colours=NUM_COLOURS,
            max_grid_size=MAX_GRID,
            num_encoder_layers=1,
            num_cross_attn_layers=1,
            num_attn_heads=4,
            num_rule_slots=N_SLOTS,
            rule_rank=RANK,
        )

        G = MAX_GRID
        max_demos = 3
        demo_inputs = torch.randint(0, NUM_COLOURS, (BATCH, max_demos, G, G))
        demo_outputs = torch.randint(0, NUM_COLOURS, (BATCH, max_demos, G, G))
        demo_mask = torch.ones(BATCH, max_demos, dtype=torch.bool)
        test_input = torch.randint(0, NUM_COLOURS, (BATCH, G, G))

        logits, alpha, meta = model(demo_inputs, demo_outputs, demo_mask, test_input)

        assert logits.shape == (BATCH, MAX_CELLS, NUM_COLOURS)
        assert alpha.shape == (BATCH, 3)
        assert meta["logits_mem"].shape == (BATCH, MAX_CELLS, NUM_COLOURS)
        assert meta["logits_rule"].shape == (BATCH, MAX_CELLS, NUM_COLOURS)
        assert meta["logits_guess"].shape == (BATCH, MAX_CELLS, NUM_COLOURS)
        assert meta["memory_strength"].shape == (N_SLOTS,)

    def test_forward_with_padding(self) -> None:
        """Forward pass should handle padded grids (PAD_VALUE = -1)."""
        model = FusionModel(
            embed_dim=EMBED,
            num_colours=NUM_COLOURS,
            max_grid_size=MAX_GRID,
            num_encoder_layers=1,
            num_cross_attn_layers=1,
            num_attn_heads=4,
            num_rule_slots=N_SLOTS,
            rule_rank=RANK,
        )

        G = MAX_GRID
        max_demos = 3
        demo_inputs = torch.randint(0, NUM_COLOURS, (BATCH, max_demos, G, G))
        demo_outputs = torch.randint(0, NUM_COLOURS, (BATCH, max_demos, G, G))
        demo_mask = torch.tensor([[True, True, False]] * BATCH)
        test_input = torch.randint(0, NUM_COLOURS, (BATCH, G, G))
        test_input[:, 3:, :] = -1  # pad bottom rows

        logits, alpha, _ = model(demo_inputs, demo_outputs, demo_mask, test_input)

        assert logits.shape == (BATCH, MAX_CELLS, NUM_COLOURS)
        assert alpha.shape == (BATCH, 3)


# ── Hebbian gradient fix tests ───────────────────────────────────────────────


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
        h = torch.randn(BATCH, EMBED)
        logits, _, info = mem(x, h)

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
        h = torch.randn(BATCH, EMBED)
        _, _, info = mem(x, h)

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
        h = torch.randn(BATCH, EMBED)
        mem(x, h)

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
        h = torch.randn(BATCH, EMBED)
        mem(x, h)

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
        model = FusionModel(
            embed_dim=EMBED,
            num_colours=NUM_COLOURS,
            max_grid_size=MAX_GRID,
            num_encoder_layers=1,
            num_cross_attn_layers=1,
            num_attn_heads=4,
            num_rule_slots=N_SLOTS,
            rule_rank=RANK,
        )

        H, W = 3, MAX_GRID
        max_demos = 2
        demo_inputs = torch.randint(0, NUM_COLOURS, (BATCH, max_demos, H, W))
        demo_outputs = torch.randint(0, NUM_COLOURS, (BATCH, max_demos, H, W))
        demo_mask = torch.ones(BATCH, max_demos, dtype=torch.bool)
        test_input = torch.randint(0, NUM_COLOURS, (BATCH, H, W))

        logits, alpha, meta = model(demo_inputs, demo_outputs, demo_mask, test_input)

        assert logits.shape == (BATCH, H * W, NUM_COLOURS)
        assert alpha.shape == (BATCH, 3)
