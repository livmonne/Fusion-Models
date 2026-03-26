"""Unit tests for the individual Fusion Model components (JAX/Flax).

Each test creates a component with small dimensions, feeds it a random
batch, and verifies that the output shapes and value ranges are correct.
These tests run on CPU and do not require any data files (except for
parquet tests which create a temporary file on the fly).
"""

from __future__ import annotations

import json
import os
import tempfile

import jax
import jax.numpy as jnp
import numpy as np

from fusion_model.decision import DecisionRouter
from fusion_model.guess import GuessComponent
from fusion_model.loss import fusion_loss
from fusion_model.memory import RuleMemory
from fusion_model.model import FusionModel
from fusion_model.rule_engine import RuleGenerator
from tasks.arc import ARCDataset, ParquetARCDataset, pad_grid

# Shared test dimensions — kept small so tests run in milliseconds.
BATCH = 4
EMBED = 64
NUM_COLOURS = 10
MAX_GRID = 4  # small grid for fast tests
MAX_CELLS = MAX_GRID * MAX_GRID
N_SLOTS = 8
RANK = 4

# Shared RNG key.
RNG = jax.random.PRNGKey(42)


def _init_module(module, *args, rng=None, training=False, **kwargs):
    """Initialise a Flax module and return (variables, output)."""
    if rng is None:
        rng = RNG
    init_rng, dropout_rng = jax.random.split(rng)
    variables = module.init(
        {"params": init_rng, "dropout": dropout_rng},
        *args,
        training=training,
        **kwargs,
    )
    return variables


def _apply_module(module, variables, *args, training=False, mutable=None, rng=None):
    """Apply a Flax module forward pass."""
    if rng is None:
        rng = RNG
    _, dropout_rng = jax.random.split(rng)
    rngs = {"dropout": dropout_rng}
    if mutable:
        return module.apply(variables, *args, training=training, rngs=rngs, mutable=mutable)
    return module.apply(variables, *args, training=training, rngs=rngs)


class TestRuleMemory:
    """Tests for :class:`fusion_model.memory.RuleMemory`."""

    def _make_mem(self, prune_every_n_steps=100):
        mem = RuleMemory(
            embed_dim=EMBED, num_colours=NUM_COLOURS,
            num_slots=N_SLOTS, rank=RANK,
            prune_every_n_steps=prune_every_n_steps,
        )
        x = jax.random.normal(RNG, (BATCH, MAX_CELLS, EMBED))
        h = jax.random.normal(jax.random.PRNGKey(1), (BATCH, EMBED))
        variables = _init_module(mem, x, h, training=True)
        return mem, variables, x, h

    def test_output_shapes(self) -> None:
        """Logits, repr, and retrieval info must have the expected shapes."""
        mem, variables, x, h = self._make_mem()
        logits, mem_repr, info = _apply_module(mem, variables, x, h)

        assert logits.shape == (BATCH, MAX_CELLS, NUM_COLOURS)
        assert mem_repr.shape == (BATCH, EMBED)
        assert info["scores"].shape == (BATCH, N_SLOTS)
        assert info["strength"].shape == (N_SLOTS,)

    def test_scores_sum_to_one(self) -> None:
        """Strength-gated retrieval scores must be valid probabilities."""
        mem, variables, x, h = self._make_mem()
        _, _, info = _apply_module(mem, variables, x, h)
        sums = info["scores"].sum(axis=-1)
        assert jnp.allclose(sums, jnp.ones(BATCH), atol=1e-5)

    def test_strength_in_unit_interval(self) -> None:
        """Memory strength values must lie in [0, 1]."""
        mem, variables, x, h = self._make_mem()
        _, _, info = _apply_module(mem, variables, x, h)
        strength = info["strength"]
        assert (strength >= 0.0).all() and (strength <= 1.0).all()

    def test_frequency_updates_during_training(self) -> None:
        """Frequency state should change after a training forward pass."""
        mem, variables, x, h = self._make_mem(prune_every_n_steps=0)

        # Run forward in training mode with mutable state.
        (logits, mem_repr, info), mutated = _apply_module(
            mem, variables, x, h, training=True, mutable=["state"],
        )

        # State should have been created/updated.
        assert "state" in mutated
        assert "frequency" in mutated["state"]

    def test_learnable_rates_are_parameters(self) -> None:
        """Decay rate, reinforcement rate, and recency half-life must be params."""
        mem, variables, x, h = self._make_mem()
        param_keys = set(variables["params"].keys())
        assert "decay_rate_logit" in param_keys
        assert "reinforce_rate_logit" in param_keys
        assert "recency_halflife_log" in param_keys


class TestRuleGenerator:
    """Tests for :class:`fusion_model.rule_engine.RuleGenerator`."""

    def _make_gen(self, **kwargs):
        gen = RuleGenerator(embed_dim=EMBED, num_colours=NUM_COLOURS, rank=RANK, **kwargs)
        x = jax.random.normal(RNG, (BATCH, MAX_CELLS, EMBED))
        h = jax.random.normal(jax.random.PRNGKey(1), (BATCH, EMBED))
        variables = _init_module(gen, x, h, training=True)
        return gen, variables, x, h

    def test_output_shapes(self) -> None:
        """Logits, confidence, and repr must have the expected shapes."""
        gen, variables, x, h = self._make_gen()
        logits, confidence, rule_repr, _proposal = _apply_module(
            gen, variables, x, h,
        )

        assert logits.shape == (BATCH, MAX_CELLS, NUM_COLOURS)
        assert confidence.shape == (BATCH, 1)
        assert rule_repr.shape == (BATCH, EMBED)

    def test_confidence_range(self) -> None:
        """Confidence must lie in [0, 1] (sigmoid output)."""
        gen, variables, x, h = self._make_gen()
        _, confidence, _, _ = _apply_module(gen, variables, x, h)
        assert (confidence >= 0.0).all() and (confidence <= 1.0).all()

    def test_proposal_disabled_before_min_history(self) -> None:
        """commit_weight must be zero when history is below min_history."""
        gen, variables, x, h = self._make_gen(min_history=64)
        _, _, _, proposal = _apply_module(gen, variables, x, h)
        assert proposal is not None
        assert float(proposal["commit_weight"].sum()) == 0.0


class TestGuessComponent:
    """Tests for :class:`fusion_model.guess.GuessComponent`."""

    def test_output_shapes(self) -> None:
        """Output logits and pooled representation must have expected shapes."""
        guess = GuessComponent(embed_dim=EMBED, num_colours=NUM_COLOURS)
        x = jax.random.normal(RNG, (BATCH, MAX_CELLS, EMBED))
        variables = _init_module(guess, x)
        logits, pooled = _apply_module(guess, variables, x)
        assert logits.shape == (BATCH, MAX_CELLS, NUM_COLOURS)
        assert pooled.shape == (BATCH, EMBED)


class TestDecisionRouter:
    """Tests for :class:`fusion_model.decision.DecisionRouter`."""

    def test_alpha_shape_and_sum(self) -> None:
        """Routing weights must be (batch, 3) and sum to 1."""
        router = DecisionRouter(embed_dim=EMBED)
        h = jax.random.normal(RNG, (BATCH, EMBED))
        mem_repr = jax.random.normal(jax.random.PRNGKey(1), (BATCH, EMBED))
        rule_repr = jax.random.normal(jax.random.PRNGKey(2), (BATCH, EMBED))
        guess_repr = jax.random.normal(jax.random.PRNGKey(3), (BATCH, EMBED))

        init_rng = jax.random.PRNGKey(0)
        variables = router.init(init_rng, h, mem_repr, rule_repr, guess_repr)
        alpha, attn_weights = router.apply(variables, h, mem_repr, rule_repr, guess_repr)

        assert alpha.shape == (BATCH, 3)
        assert jnp.allclose(alpha.sum(axis=-1), jnp.ones(BATCH), atol=1e-5)

    def test_attn_weights_shape(self) -> None:
        """Per-head attention weights must have shape (batch, num_heads, 3)."""
        num_heads = 4
        router = DecisionRouter(embed_dim=EMBED, num_heads=num_heads)
        h = jax.random.normal(RNG, (BATCH, EMBED))
        reprs = [jax.random.normal(jax.random.PRNGKey(i), (BATCH, EMBED)) for i in range(3)]

        variables = router.init(jax.random.PRNGKey(0), h, *reprs)
        _alpha, attn_weights = router.apply(variables, h, *reprs)

        assert attn_weights.shape == (BATCH, num_heads, 3)
        sums = attn_weights.sum(axis=-1)
        assert jnp.allclose(sums, jnp.ones_like(sums), atol=1e-5)


class TestFusionLoss:
    """Tests for :func:`fusion_model.loss.fusion_loss`."""

    def test_loss_is_scalar(self) -> None:
        """Total loss must be a scalar."""
        logits = jax.random.normal(RNG, (BATCH, MAX_CELLS, NUM_COLOURS))
        targets = jax.random.randint(jax.random.PRNGKey(1), (BATCH, MAX_CELLS), 0, NUM_COLOURS)
        alphas = jax.nn.softmax(jax.random.normal(jax.random.PRNGKey(2), (BATCH, 3)), axis=-1)
        scores = jax.nn.softmax(jax.random.normal(jax.random.PRNGKey(3), (BATCH, N_SLOTS)), axis=-1)

        loss, loss_dict = fusion_loss(logits, targets, alphas, scores)
        assert loss.shape == ()
        assert "total" in loss_dict

    def test_aux_loss_uses_per_cell_ce(self) -> None:
        """Auxiliary loss must correctly compute per-cell CE for each pathway."""
        logits = jax.random.normal(RNG, (BATCH, MAX_CELLS, NUM_COLOURS))
        targets = jax.random.randint(jax.random.PRNGKey(1), (BATCH, MAX_CELLS), 0, NUM_COLOURS)
        targets = targets.at[:, -4:].set(-1)
        alphas = jax.nn.softmax(jax.random.normal(jax.random.PRNGKey(2), (BATCH, 3)), axis=-1)
        scores = jax.nn.softmax(jax.random.normal(jax.random.PRNGKey(3), (BATCH, N_SLOTS)), axis=-1)

        metadata = {
            "logits_mem": jax.random.normal(jax.random.PRNGKey(4), (BATCH, MAX_CELLS, NUM_COLOURS)),
            "logits_rule": jax.random.normal(jax.random.PRNGKey(5), (BATCH, MAX_CELLS, NUM_COLOURS)),
            "logits_guess": jax.random.normal(jax.random.PRNGKey(6), (BATCH, MAX_CELLS, NUM_COLOURS)),
        }

        loss, loss_dict = fusion_loss(logits, targets, alphas, scores, metadata=metadata)
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
        demo_inputs = jax.random.randint(RNG, (BATCH, max_demos, G, G), 0, NUM_COLOURS)
        demo_outputs = jax.random.randint(jax.random.PRNGKey(1), (BATCH, max_demos, G, G), 0, NUM_COLOURS)
        demo_mask = jnp.ones((BATCH, max_demos), dtype=jnp.bool_)
        test_input = jax.random.randint(jax.random.PRNGKey(2), (BATCH, G, G), 0, NUM_COLOURS)

        init_rng, dropout_rng = jax.random.split(jax.random.PRNGKey(0))
        variables = model.init(
            {"params": init_rng, "dropout": dropout_rng},
            demo_inputs, demo_outputs, demo_mask, test_input,
            training=True,
        )

        (logits, alpha, meta), _ = model.apply(
            variables,
            demo_inputs, demo_outputs, demo_mask, test_input,
            training=True,
            rngs={"dropout": dropout_rng},
            mutable=["state"],
        )

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
        demo_inputs = jax.random.randint(RNG, (BATCH, max_demos, G, G), 0, NUM_COLOURS)
        demo_outputs = jax.random.randint(jax.random.PRNGKey(1), (BATCH, max_demos, G, G), 0, NUM_COLOURS)
        demo_mask = jnp.array([[True, True, False]] * BATCH)
        test_input = jax.random.randint(jax.random.PRNGKey(2), (BATCH, G, G), 0, NUM_COLOURS)
        test_input = test_input.at[:, 3:, :].set(-1)

        init_rng, dropout_rng = jax.random.split(jax.random.PRNGKey(0))
        variables = model.init(
            {"params": init_rng, "dropout": dropout_rng},
            demo_inputs, demo_outputs, demo_mask, test_input,
            training=True,
        )

        logits, alpha, _ = model.apply(
            variables,
            demo_inputs, demo_outputs, demo_mask, test_input,
            training=False,
        )

        assert logits.shape == (BATCH, MAX_CELLS, NUM_COLOURS)
        assert alpha.shape == (BATCH, 3)


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
            assert len(ds) == 5

    def test_getitem_returns_expected_keys(self) -> None:
        """Each sample must contain the standard ARC array dict."""
        with tempfile.TemporaryDirectory() as tmp:
            path = _make_parquet_file(tmp, n_tasks=3)
            ds = ParquetARCDataset([path], max_grid_size=4)
            sample = ds[0]

            expected_keys = {
                "demo_inputs", "demo_outputs", "demo_mask",
                "test_input", "test_output", "input_size", "output_size",
            }
            assert set(sample.keys()) == expected_keys

    def test_getitem_shapes(self) -> None:
        """Array shapes must match the configured grid size and demo count."""
        G = 4
        max_demos = 3
        with tempfile.TemporaryDirectory() as tmp:
            path = _make_parquet_file(tmp, n_tasks=2)
            ds = ParquetARCDataset([path], max_grid_size=G, max_demos=max_demos)
            sample = ds[0]

            assert sample["demo_inputs"].shape == (max_demos, G, G)
            assert sample["demo_outputs"].shape == (max_demos, G, G)
            assert sample["demo_mask"].shape == (max_demos,)
            assert sample["test_input"].shape == (G, G)
            assert sample["test_output"].shape == (G, G)
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
            path2 = os.path.join(tmp, "test_data2.parquet")
            import shutil
            shutil.copy(path1, path2)

            ds = ParquetARCDataset([path1, path2])
            assert len(ds) == 6

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

            s0 = ds[0]
            s1 = ds[1]
            assert not np.array_equal(s0["test_input"], s1["test_input"])
