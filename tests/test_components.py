"""Unit tests for the individual Fusion Model components.

Each test creates a component with small dimensions, feeds it a random
batch, and verifies that the output shapes and value ranges are correct.
These tests run on CPU and do not require any data files.
"""

from __future__ import annotations

import torch

from fusion_model.decision import DecisionRouter
from fusion_model.guess import GuessComponent
from fusion_model.loss import FusionLoss
from fusion_model.memory import RuleMemory
from fusion_model.model import FusionModel
from fusion_model.rule_engine import RuleGenerator

# Shared test dimensions — kept small so tests run in milliseconds.
BATCH = 4
EMBED = 64
N_CLASSES = 28
N_SLOTS = 8
RANK = 4
VOCAB = 50
MAX_Q_LEN = 10


class TestRuleMemory:
    """Tests for :class:`fusion_model.memory.RuleMemory`."""

    def test_output_shapes(self) -> None:
        """Logits, blended correction, and retrieval info must have the expected shapes."""
        mem = RuleMemory(embed_dim=EMBED, num_classes=N_CLASSES, num_slots=N_SLOTS, rank=RANK)
        h = torch.randn(BATCH, EMBED)
        logits, blended, info = mem(h)

        assert logits.shape == (BATCH, N_CLASSES)
        assert blended.shape == (BATCH, EMBED)
        assert info["scores"].shape == (BATCH, N_SLOTS)

    def test_scores_sum_to_one(self) -> None:
        """Retrieval scores must be valid softmax probabilities."""
        mem = RuleMemory(embed_dim=EMBED, num_classes=N_CLASSES, num_slots=N_SLOTS, rank=RANK)
        h = torch.randn(BATCH, EMBED)
        _, _, info = mem(h)
        sums = info["scores"].sum(dim=-1)
        assert torch.allclose(sums, torch.ones(BATCH), atol=1e-5)


class TestRuleGenerator:
    """Tests for :class:`fusion_model.rule_engine.RuleGenerator`."""

    def test_output_shapes(self) -> None:
        """Logits, confidence, and correction must have the expected shapes."""
        gen = RuleGenerator(embed_dim=EMBED, num_classes=N_CLASSES, rank=RANK)
        h = torch.randn(BATCH, EMBED)
        logits, confidence, correction, _proposal = gen(h)

        assert logits.shape == (BATCH, N_CLASSES)
        assert confidence.shape == (BATCH, 1)
        assert correction.shape == (BATCH, EMBED)

    def test_confidence_range(self) -> None:
        """Confidence must lie in [0, 1] (sigmoid output)."""
        gen = RuleGenerator(embed_dim=EMBED, num_classes=N_CLASSES, rank=RANK)
        h = torch.randn(BATCH, EMBED)
        _, confidence, _, _proposal = gen(h)
        assert (confidence >= 0.0).all() and (confidence <= 1.0).all()

    def test_proposal_shapes_after_history_fill(self) -> None:
        """After enough history, propose_rule must return correctly shaped tensors."""
        min_hist = 16
        gen = RuleGenerator(
            embed_dim=EMBED, num_classes=N_CLASSES, rank=RANK,
            history_size=64, min_history=min_hist,
        )
        gen.train()

        for _ in range(min_hist // BATCH + 1):
            h_fill = torch.randn(BATCH, EMBED)
            decisions = torch.randint(0, N_CLASSES, (BATCH,))
            gen.update_history(h_fill, decisions)

        h = torch.randn(BATCH, EMBED)
        proposal = gen.propose_rule(h)

        assert proposal is not None
        assert proposal["key"].shape == (BATCH, EMBED)
        assert proposal["A"].shape == (BATCH, EMBED, RANK)
        assert proposal["B"].shape == (BATCH, RANK, EMBED)
        assert proposal["confidence"].shape == (BATCH, 1)
        assert (proposal["confidence"] >= 0.0).all() and (proposal["confidence"] <= 1.0).all()

    def test_proposal_none_before_min_history(self) -> None:
        """propose_rule must return None when history is below min_history."""
        gen = RuleGenerator(embed_dim=EMBED, num_classes=N_CLASSES, rank=RANK, min_history=64)
        h = torch.randn(BATCH, EMBED)
        assert gen.propose_rule(h) is None


class TestGuessComponent:
    """Tests for :class:`fusion_model.guess.GuessComponent`."""

    def test_output_shapes(self) -> None:
        """Output logits and pooled representation must have expected shapes."""
        guess = GuessComponent(embed_dim=EMBED, num_classes=N_CLASSES, num_tokens=8)
        h = torch.randn(BATCH, EMBED)
        logits, pooled = guess(h)
        assert logits.shape == (BATCH, N_CLASSES)
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
        alpha = router(h, mem_repr, rule_repr, guess_repr)

        assert alpha.shape == (BATCH, 3)
        assert torch.allclose(alpha.sum(dim=-1), torch.ones(BATCH), atol=1e-5)


class TestFusionLoss:
    """Tests for :class:`fusion_model.loss.FusionLoss`."""

    def test_loss_is_scalar(self) -> None:
        """Total loss must be a scalar tensor."""
        criterion = FusionLoss()
        logits = torch.randn(BATCH, N_CLASSES)
        targets = torch.randint(0, N_CLASSES, (BATCH,))
        alphas = torch.softmax(torch.randn(BATCH, 3), dim=-1)
        scores = torch.softmax(torch.randn(BATCH, N_SLOTS), dim=-1)

        loss, loss_dict = criterion(logits, targets, alphas, scores)
        assert loss.shape == ()
        assert "total" in loss_dict


class TestFusionModel:
    """Integration tests for :class:`fusion_model.model.FusionModel`."""

    def test_forward_shapes(self) -> None:
        """Full forward pass must produce correctly shaped outputs."""
        model = FusionModel(
            vocab_size=VOCAB,
            embed_dim=EMBED,
            num_classes=N_CLASSES,
            num_rule_slots=N_SLOTS,
            rule_rank=RANK,
        )
        images = torch.randn(BATCH, 3, 224, 224)
        questions = torch.randint(0, VOCAB, (BATCH, MAX_Q_LEN))

        logits, alpha, meta = model(images, questions)

        assert logits.shape == (BATCH, N_CLASSES)
        assert alpha.shape == (BATCH, 3)
        assert "logits_mem" in meta
        assert "logits_rule" in meta
        assert "logits_guess" in meta
