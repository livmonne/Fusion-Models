# Fusion Model — Visual Question Answering on CLEVR

A neural architecture that combines **rule memory**, **rule generation**,
**self-attention guessing**, and a **learned decision router** into a single
end-to-end trainable model, applied to the
[CLEVR](https://cs.stanford.edu/people/jcjohns/clevr/) Visual Question
Answering benchmark.

## Architecture

```
Image (3×224×224)                  Question ("How many red cubes …")
       │                                      │
  ResNet-18 early (frozen)              Embedding + GRU
  conv1 → layer2                              │
       │                              question_proj (256→256)
  ResNet-18 late (fine-tuned)                 │
  layer3 → layer4                             │
       │                                      │
  1×1 conv (512→256)                          │
       │                                      │
  spatial map (49×256)                   q_emb (256)
       │                                      │
       └── multi-head spatial attention (4 heads) ──┘
                      │
               img_emb (256)  ×  q_emb (256)
                      │               │
               fusion_proj (256→256)  │  (residual)
                      │               │
                      └───── + ───────┘
                      │
                      h  (shared embedding)
                      │
       ┌──────────────┼──────────────┐
       ▼              ▼              ▼
  RuleMemory    RuleGenerator   GuessComponent
       │              │              │
  logits + repr  logits + repr  logits + repr
       │              │              │
       ▼              ▼              ▼
       ┌──────────────┼──────────────┐
       │    DecisionRouter (cross-attention)
       │    query = h
       │    keys  = [mem_repr, rule_repr, guess_repr]
       │              │
       │       α = attn weights (3)
       └──────────────┘
                      │
     p(y|x) = α_mem·p_mem + α_rule·p_rule + α_guess·p_guess
```

### Components

| Component | Role |
|---|---|
| **RuleMemory** | Bank of 128 learned low-rank rules with trigger embeddings. Differentiable soft-attention retrieval **gated by memory strength** (frequency × recency). Learnable decay/reinforcement rates and recency half-life. Automatic pruning of forgotten slots. Supports soft-blend commitment of proposed rules. Exposes blended correction vector for routing. |
| **RuleGenerator** | Proposes ephemeral one-shot rules as low-rank corrections per input. Uses a three-stage cross-attention pipeline (history, decision, synthesis) over a circular history buffer to propose persistent rules with a learned soft commit weight. Exposes correction vector for routing. |
| **GuessComponent** | Self-attention over pseudo-tokens followed by an MLP head for fuzzy patterns. Exposes pooled representation for routing. |
| **DecisionRouter** | Cross-attention router: uses ``h`` as query and each pathway's intermediate representation as keys to produce input-dependent mixture weights. |

### Memory Strength (Biologically-Inspired Decay & Reinforcement)

Each rule-memory slot carries a **strength** value in `[0, 1]` that
modulates how much it contributes during retrieval.  Strength is the
product of two independent signals inspired by neuroscience:

| Signal | What it captures | Mechanism |
|---|---|---|
| **Frequency** | How often the slot is triggered | Running accumulator: decayed each step by a **learnable decay rate**, boosted by a **learnable reinforcement rate** × batch-mean retrieval score. |
| **Recency** | How recently the slot was last strongly activated | Exponential decay: `exp(-ln2 × steps_since_activation / half_life)` where `half_life` is a **learnable parameter**. |

**`strength = frequency_score × recency_score`**

All three dynamics parameters (decay rate, reinforcement rate, recency
half-life) are **learnable** — stored as unconstrained logits and mapped
through sigmoid/exp so the model discovers its own optimal
forgetting/consolidation schedule via gradient descent.  A strength
regularisation term in the loss prevents degenerate regimes (total amnesia
or total saturation).

Slots whose strength drops below a configurable threshold are **pruned**:
their parameters are re-initialised with small random values and given a
moderate starting frequency, recycling capacity for new rules.

## Quick Start

```bash
# Install dependencies (requires uv: https://docs.astral.sh/uv/)
make install

# Run linter + type-checker
make lint

# Run tests
make test

# Train on CLEVR (quick debug run with 100 samples)
make train

# Full training
uv run python train.py --clevr_root CLEVR_v1.0 --epochs 20

# Train on only 25% of the data (stratified by answer class)
uv run python train.py --clevr_root CLEVR_v1.0 --train_fraction 0.25

# Train on 10% and add the other 90% to the validation set
uv run python train.py --clevr_root CLEVR_v1.0 --train_fraction 0.10 --extend_val

# Evaluate and generate visualisation plots
uv run python evaluate.py --clevr_root CLEVR_v1.0
```

On Windows (PowerShell):

```powershell
.\Make.ps1 install
.\Make.ps1 lint
.\Make.ps1 test
.\Make.ps1 train
```

## Data-Efficiency Experiments

Use `--train_fraction` to train on a subset of the data and see how the model
performs with less supervision.  The split is **stratified by answer class** so
every one of the 28 answer types is represented proportionally (with at least
one sample each).

| Flag | Description |
|---|---|
| `--train_fraction F` | Keep fraction `F` of the training set (0.0–1.0, default 1.0). |
| `--extend_val` | Append the unused training samples to the validation set. |

Both flags are deterministic — controlled by `--seed` — so runs are
reproducible.  When `--train_fraction 1.0` (the default), neither flag has any
effect and the pipeline behaves exactly as before.

## Project Structure

```
fusion_model/
    __init__.py         # Package re-exports
    model.py            # FusionModel orchestrator (image+question encoders + fusion)
    memory.py           # RuleMemory (low-rank rule bank + differentiable retrieval)
    rule_engine.py      # RuleGenerator (ephemeral hypothesis proposer)
    guess.py            # GuessComponent (self-attention predictor)
    decision.py         # DecisionRouter (softmax mixture weights)
    loss.py             # FusionLoss (task + regularisation terms)
tasks/
    clevr.py            # CLEVR dataset loader, tokeniser, and answer vocabulary
tests/
    test_components.py  # Unit tests for all model components
train.py                # Training loop
evaluate.py             # Evaluation + matplotlib visualisations
pyproject.toml          # UV project config (deps, ruff, mypy, pytest)
Makefile                # Unix make targets
Make.ps1                # PowerShell equivalent
.pre-commit-config.yaml # Pre-commit hooks (ruff + mypy)
```
