"""FusionModel — orchestrator that wires all components together.

The Fusion Model is a multimodal architecture for Visual Question Answering.
It accepts an **image** and a **question** and produces a probability
distribution over a fixed set of answers.

**Forward pass overview:**

1. **Image encoding** — a pre-trained ResNet-18 extracts a spatial feature
   map.  Early layers (conv1 through layer2) are frozen; layer3 and layer4
   are fine-tuned.  The resulting ``(batch, 512, 7, 7)`` map is projected
   to ``embed_dim`` channels via a 1x1 convolution.
2. **Question encoding** — word embeddings are fed through a GRU; the last
   hidden state is projected to ``embed_dim``.
3. **Multi-head spatial attention** — the question embedding attends over
   the 7x7 spatial grid via ``num_attn_heads`` parallel attention heads.
   Each head can focus on a different image region (e.g. subject vs.
   reference object), and their outputs are concatenated into a single
   attended image vector of shape ``(batch, embed_dim)``.
4. **Multimodal fusion** — the attended image vector and question vector
   are combined (element-wise product + linear projection) into a single
   shared embedding ``h``.
5. **Three expert pathways** each process ``h`` independently:
   - :class:`~fusion_model.memory.RuleMemory` — retrieves stored rules.
   - :class:`~fusion_model.rule_engine.RuleGenerator` — produces ephemeral
     corrections *and* proposes persistent rules for the memory bank.
   - :class:`~fusion_model.guess.GuessComponent` — self-attention predictor.
6. **Rule commitment** — if the RuleGenerator's proposal confidence exceeds
   a learnable threshold, the proposed rule is written into the
   lowest-utility slot of the memory bank.
7. **DecisionRouter** — produces softmax mixture weights ``alpha`` over the
   three pathways.
8. **History update** — the current ``(h, prediction)`` pair is appended to
   the RuleGenerator's circular history buffer (training only).
9. **Output** — the final logits are the weighted sum of the pathway logits.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torchvision.models as models

from .decision import DecisionRouter
from .guess import GuessComponent
from .memory import RuleMemory
from .rule_engine import RuleGenerator


class FusionModel(nn.Module):
    """End-to-end Fusion Model for CLEVR Visual Question Answering.

    :param vocab_size: Number of unique words in the question vocabulary
        (including ``<PAD>`` and ``<UNK>``).
    :param embed_dim: Internal embedding dimensionality shared by all
        expert pathways.
    :param num_classes: Number of answer classes (28 for CLEVR).
    :param num_rule_slots: Number of rule slots in the memory bank.
    :param rule_rank: Low-rank dimension used by both memory and generator.
    :param q_embed_dim: Word-embedding dimension for question tokens.
    :param q_hidden_dim: GRU hidden size for the question encoder.
    :param num_attn_heads: Number of parallel attention heads for the
        question-guided spatial attention over the image feature map.
    :param history_size: Capacity of the RuleGenerator's circular history
        buffer for ``(h, prediction)`` pairs.
    """

    def __init__(
        self,
        vocab_size: int = 100,
        embed_dim: int = 256,
        num_classes: int = 28,
        num_rule_slots: int = 128,
        rule_rank: int = 16,
        q_embed_dim: int = 128,
        q_hidden_dim: int = 256,
        num_attn_heads: int = 4,
        history_size: int = 512,
    ) -> None:
        super().__init__()

        # ── Image encoder (partially fine-tuned ResNet-18) ────────────────
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # Frozen early layers: learn generic low-level features (edges,
        # colours, textures) that transfer well across domains.
        self.backbone_frozen = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2,
        )
        for param in self.backbone_frozen.parameters():
            param.requires_grad = False

        # Fine-tuned late layers: adapt high-level features to CLEVR's
        # synthetic objects, producing a (batch, 512, 7, 7) spatial map.
        self.backbone_finetune = nn.Sequential(resnet.layer3, resnet.layer4)

        # 1x1 conv projects each spatial position: 512 → embed_dim.
        self.image_proj = nn.Conv2d(512, embed_dim, kernel_size=1)

        # ── Multi-head spatial attention ─────────────────────────────────
        # The question embedding queries over the 7x7 spatial grid.  Each
        # head can attend to a different image region independently, letting
        # the model reason about multiple objects or spatial relationships.
        self.spatial_attn = nn.MultiheadAttention(
            embed_dim, num_attn_heads, batch_first=True,
        )

        # ── Question encoder (embedding + GRU) ──────────────────────────
        self.word_embed = nn.Embedding(vocab_size, q_embed_dim, padding_idx=0)
        self.question_gru = nn.GRU(q_embed_dim, q_hidden_dim, batch_first=True, bidirectional=False)
        # Project GRU hidden -> embed_dim.
        self.question_proj = nn.Linear(q_hidden_dim, embed_dim)

        # ── Multimodal fusion ────────────────────────────────────────────
        # Element-wise product followed by a linear layer to blend modalities.
        self.fusion_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
        )

        # ── Expert pathways ──────────────────────────────────────────────
        self.memory = RuleMemory(
            embed_dim=embed_dim,
            num_classes=num_classes,
            num_slots=num_rule_slots,
            rank=rule_rank,
        )
        self.rule_gen = RuleGenerator(
            embed_dim=embed_dim,
            num_classes=num_classes,
            rank=rule_rank,
            history_size=history_size,
        )
        self.guess = GuessComponent(
            embed_dim=embed_dim,
            num_classes=num_classes,
        )

        # ── Decision router ──────────────────────────────────────────────
        self.router = DecisionRouter(embed_dim=embed_dim)

    def forward(
        self,
        images: torch.Tensor,
        questions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Run the full Fusion Model forward pass.

        :param images: Batch of images ``(batch, 3, 224, 224)``.
        :param questions: Batch of tokenised questions ``(batch, max_q_len)``.
        :return: Tuple of ``(logits, alphas, metadata)`` where *logits* has
            shape ``(batch, num_classes)``, *alphas* has shape ``(batch, 3)``
            and *metadata* holds per-pathway logits and retrieval info.
        """
        # ── Encode image into spatial feature map ─────────────────────────
        with torch.no_grad():
            x = self.backbone_frozen(images)
        feat_map = self.backbone_finetune(x)          # (batch, 512, 7, 7)
        feat_map = self.image_proj(feat_map)           # (batch, embed_dim, 7, 7)
        spatial = feat_map.flatten(2).permute(0, 2, 1) # (batch, 49, embed_dim)

        # ── Encode question ──────────────────────────────────────────────
        word_emb = self.word_embed(questions)  # (batch, seq_len, q_embed_dim)
        _, q_hidden = self.question_gru(word_emb)  # q_hidden: (1, batch, q_hidden_dim)
        q_emb = self.question_proj(q_hidden.squeeze(0))  # (batch, embed_dim)

        # ── Multi-head spatial attention ─────────────────────────────────
        # Each head independently attends to different spatial regions,
        # allowing the model to focus on multiple objects or relationships.
        img_emb, _ = self.spatial_attn(
            query=q_emb.unsqueeze(1), key=spatial, value=spatial,
        )                                    # (batch, 1, embed_dim)
        img_emb = img_emb.squeeze(1)         # (batch, embed_dim)

        # ── Fuse modalities via element-wise product + projection ────────
        h = self.fusion_proj(img_emb * q_emb)  # (batch, embed_dim)

        # ── Expert pathways ──────────────────────────────────────────────
        logits_mem, retrieval_info = self.memory(h)
        logits_rule, confidence, proposal = self.rule_gen(h)
        logits_guess = self.guess(h)

        # ── Route and blend ──────────────────────────────────────────────
        alpha = self.router(h, retrieval_info["top_key"], confidence)

        # Weighted mixture: each alpha slice is (batch, 1) for broadcasting.
        logits = (
            alpha[:, 0:1] * logits_mem + alpha[:, 1:2] * logits_rule + alpha[:, 2:3] * logits_guess
        )

        # ── Rule commitment ──────────────────────────────────────────────
        committed = False
        if self.training and proposal is not None:
            prop_conf = proposal["confidence"]  # (batch, 1)
            threshold = proposal["commit_threshold"]  # scalar
            best_idx = int(prop_conf.argmax(dim=0).item())
            best_conf = prop_conf[best_idx, 0]

            if best_conf.item() > threshold.item():
                slot = self.memory.get_lowest_utility_slot()
                self.memory.commit_rule(
                    slot,
                    proposal["key"][best_idx],
                    proposal["A"][best_idx],
                    proposal["B"][best_idx],
                )
                committed = True

        # ── Update history buffer with current predictions ───────────────
        if self.training:
            preds = logits.detach().argmax(dim=-1)  # (batch,)
            self.rule_gen.update_history(h, preds)

        metadata: dict[str, Any] = {
            "logits_mem": logits_mem,
            "logits_rule": logits_rule,
            "logits_guess": logits_guess,
            "retrieval_scores": retrieval_info["scores"],
            "rule_confidence": confidence,
            "proposal": proposal,
            "committed": committed,
        }
        return logits, alpha, metadata
