# Fusion Model — Abstract Reasoning on ARC-AGI-2

A neural architecture that combines **rule memory**, **rule generation**,
**self-attention guessing**, and a **learned decision router** into a single
end-to-end trainable model, applied to the
[ARC-AGI-2](https://github.com/arcprize/ARC-AGI-2) abstract reasoning
benchmark.

## Architecture

```
Demo Pairs (input/output grids)                Test Input Grid
       │                                              │
  Cell Embedding (10 colours → embed_dim)        Cell Embedding
  + 2-D Sinusoidal Positional Encoding           + 2-D Pos Enc
  + Type Embedding (demo_in / demo_out)          + Type Embed (test_in)
       │                                              │
  ┌────┴────┐                                         │
  │ Concat  │  (input + output tokens per demo)       │
  │ pair    │                                         │
  └────┬────┘                                         │
       │                                              │
  Transformer Encoder                                 │
  (self-attention × N layers)                         │
       │                                              │
  Demo Context (all demo tokens concatenated)         │
       │                                              │
       └── Multi-Head Cross-Attention (× N layers) ──┘
           test tokens query demo context
                      │
               x = cross-attended test tokens
                   (batch, seq, embed_dim)
                      │
              ┌───────┴────────┐
              │                │
              │         Mean Pool (mask padding)
              │                │
              │         pool_proj (embed_dim → embed_dim)
              │                │
              │                h  (pooled embedding, batch, embed_dim)
              │                │
       ┌──────┼────────────────┼──────────────┐
       ▼      ▼                ▼              ▼
  RuleMemory(x, h)    RuleGenerator(x, h)  GuessComponent(x)
       │                    │                  │
  (B, seq, 10)        (B, seq, 10)       (B, seq, 10)
  + mem_repr (B,E)    + rule_repr (B,E)  + guess_repr (B,E)
       │                    │                  │
       ▼                    ▼                  ▼
       ┌────────────────────┼──────────────────┐
       │  DecisionRouter (multi-head cross-attention, 4 heads)
       │  query  = h
       │  keys   = [mem_repr, rule_repr, guess_repr]
       │  values = [mem_repr, rule_repr, guess_repr]
       │              │
       │  ┌───────────┴───────────┐
       │  │  MHA context (256)    │  per-head attn weights
       │  │       │               │  (batch, 4, 3) — logged
       │  │       + h (residual)  │
       │  │       │               │
       │  │  LayerNorm (256)      │
       │  │       │               │
       │  │  MLP (256→256→3)      │
       │  │  Linear→GELU→Linear   │
       │  │       │               │
       │  │  softmax / τ          │
       │  └───────┴───────────────┘
       │       α (batch, 3)
       └──────────────┘
                      │
     Blended logits = α₀·logits_mem + α₁·logits_rule + α₂·logits_guess
     Per-cell colour predictions: (batch, seq, 10)
```

### Components

| Component | Role |
|---|---|
| **Cell Embedding + Pos Enc** | Maps each grid cell (0–9) to a learned vector with 2-D sinusoidal positional encoding and type embeddings to distinguish demo inputs, demo outputs, and test inputs. |
| **Transformer Encoder** | Processes concatenated demo input+output token sequences via self-attention, allowing the model to learn the transformation pattern within each demo pair. |
| **Multi-Head Cross-Attention** | Test input tokens cross-attend to the demo context. Each head can focus on different aspects of the demonstrated transformation (colour mapping, spatial pattern, etc.). This is the core mechanism for transferring the inferred rule to the test input. |
| **RuleMemory** | Bank of 128 learned low-rank rules with trigger embeddings. Receives the full spatial sequence `x` and pooled `h`: retrieval scores are computed from `h` via **strength-gated** soft attention (frequency × recency), while low-rank corrections `A @ (B @ x_token)` and per-slot classification heads are applied independently to every spatial token, producing per-cell colour logits `(batch, seq, 10)`. Learnable decay/reinforcement rates and recency half-life. Automatic pruning of forgotten slots. Supports soft-blend commitment of proposed rules. Returns a mean-pooled correction vector as its router representation. |
| **RuleGenerator** | Produces ephemeral one-shot rules as low-rank corrections: an MLP generates A, B matrices from pooled `h`, then the correction `A @ (B @ x_token)` is applied per spatial token, yielding per-cell logits `(batch, seq, 10)`. Maintains a circular history buffer of `(embedding, decision, outcome)` triples and uses a three-stage cross-attention pipeline — operating entirely on history, not the current input — to propose persistent rules: (1) historical inputs attend over historical decisions, (2) that result attends over outcome signals (per-sample loss), (3) a learned synthesis query fuses the two. Produces a soft commit weight for blending into the weakest memory slot. Returns a mean-pooled correction vector as its router representation. |
| **GuessComponent** | Multi-head self-attention over the full spatial token sequence `x`, followed by a per-token MLP classification head for fuzzy pattern matching. Returns per-cell logits `(batch, seq, 10)` and a mean-pooled representation for routing. |
| **DecisionRouter** | Multi-head cross-attention router (4 heads): uses ``h`` as query and each pathway's pooled representation as keys *and* values.  A residual connection adds ``h`` back to the attention context, followed by LayerNorm, so the routing MLP always sees both the raw input and the pathway-informed context.  A two-layer MLP (Linear → GELU → Linear) maps the normalised vector to three routing logits, enabling nonlinear feature interactions.  Per-head attention weights are returned for interpretability. |

### Spatial Expert Pathways

A key design principle is that **every expert pathway operates on the full
spatial token sequence** `x (batch, seq, embed_dim)` rather than a single
pooled vector.  This preserves per-cell positional information all the way
through to the output, allowing each expert to make position-dependent
predictions.

The pooled vector `h` is still computed (via masked mean-pooling + a
projection) and used for:
- **Retrieval key matching** in RuleMemory (which rules to activate)
- **Ephemeral MLP input** in RuleGenerator (what correction to produce)
- **Router query** in DecisionRouter (how to blend the experts)
- **History buffer entries** in RuleGenerator (for rule proposal)

Each expert also returns a pooled representation `(batch, embed_dim)` for
the router — derived by mean-pooling its internal per-token features.

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
forgetting/consolidation schedule via gradient descent.  During each
forward pass the new frequency is computed differentiably from the rate
parameters and used in the strength gating, providing the gradient path
`loss → scores → strength → new_freq → rate logits`.  A strength
regularisation term in the loss prevents degenerate regimes (total amnesia
or total saturation).

Slots whose strength drops below a configurable threshold are **pruned**:
their parameters are re-initialised with small random values and given a
moderate starting frequency, recycling capacity for new rules.

### Decision Router (Multi-Head Cross-Attention Routing)

The DecisionRouter determines how much each expert pathway contributes to
the final answer.  It does this via **multi-head cross-attention** over the
three pathway representations, followed by a learned projection.

#### Why cross-attention instead of an MLP?

A naive approach would concatenate the three pathway representations into a
flat vector and pass it through an MLP to produce three mixture weights.
This works, but the MLP sees a fixed-size input regardless of what the
pathways are "saying" — it can only learn static feature-position mappings.

Cross-attention is fundamentally different: the shared embedding `h` acts
as a **query** that *asks* each pathway "what can you offer for this
input?"  Each pathway's representation acts as a **key** (controlling when
it attracts attention) and a **value** (controlling what information it
communicates).  This means the routing decision is **input-dependent** by
construction — the same pathway can be upweighted or downweighted depending
on the specific task.

#### How routing weights are produced

1. **Multi-head attention** — `h` queries over the 3 pathway tokens
   (key = value = pathway representations) across `num_heads` independent
   subspaces.

2. **Residual connection + LayerNorm** — the original shared embedding `h`
   is added back to the attention context and normalised.

3. **Two-layer MLP** — maps the normalised vector to 3 logits.

4. **Temperature-scaled softmax** — produces the final routing
   coefficients `α ∈ [0, 1]³` that sum to 1.

### How the Pieces Fit Together

The architecture follows a **perceive → specialise → arbitrate** pipeline:

1. **Perceive** — grid cells are embedded with positional and type
   information.  Demo pairs are encoded by a shared Transformer encoder.
   Multi-head cross-attention transfers the inferred transformation from
   demos to the test input, producing spatial tokens `x (batch, seq, E)`.

2. **Pool** — the cross-attended test tokens are mean-pooled (masking
   padding) into a global summary vector `h (batch, E)`.  Both `x` and `h`
   are passed downstream.

3. **Specialise** — three expert pathways process the spatial tokens `x`
   independently, each producing per-cell colour logits `(batch, seq, 10)`:
   - **RuleMemory** retrieves and applies stored rules per token.
   - **RuleGenerator** synthesises one-shot rules and applies them per token.
   - **GuessComponent** uses self-attention over spatial tokens for fuzzy
     pattern matching.

4. **Arbitrate** — the DecisionRouter uses multi-head cross-attention
   (querying each expert's pooled representation with `h`) to produce
   per-sample routing weights `α`.  The final prediction is a soft
   mixture: `logits = α₀·mem + α₁·rule + α₂·guess` applied per cell.

5. **Output** — the blended logits are per-cell colour predictions
   `(batch, seq, num_colours)` for the output grid.

6. **Consolidate** — during training, the RuleGenerator proposes new
   rules for permanent storage in the RuleMemory.  The per-sample loss
   is fed back into the history buffer as an outcome signal so the
   proposer can learn which input→decision pairings were effective.
   History update is deferred until after the loss is computed.

### Loss Function

The training objective balances seven terms:

| Term | Purpose | Effect |
|---|---|---|
| **Task loss** | Per-cell cross-entropy (ignoring padding) | Main learning signal |
| **Guess penalty** | Penalises `mean(α_guess)` | Prevents over-reliance on the guess fallback |
| **Storage cost** | Approximate L0 over slot usage | Encourages sparse, specialised memory slots |
| **Entropy bonus** | Maximises `H(α)` | Prevents routing collapse early in training |
| **Auxiliary losses** | Per-cell cross-entropy on each pathway's own logits (ignoring padding) | Keeps all pathways learning even when the router ignores them |
| **Commitment reg.** | Penalises deviation from target commit rate | Prevents the rule proposer from committing too aggressively or never |
| **Strength reg.** | Penalises deviation from target mean strength | Prevents total amnesia or total saturation of memory slots |

## Quick Start

```bash
# Install dependencies (requires uv: https://docs.astral.sh/uv/)
make install

# Clone the ARC-AGI-2 dataset
git clone https://github.com/arcprize/ARC-AGI-2.git arc-agi-2
ln -s arc-agi-2/data data

# Run linter + type-checker
make lint

# Run tests
make test

# Train on ARC-AGI-2 (quick debug run)
make train

# Full training
uv run python train.py --data_root data --epochs 40

# Evaluate and generate visualisation plots
uv run python evaluate.py --data_root data
```

## ARC-AGI-2 Dataset

The [ARC-AGI-2](https://github.com/arcprize/ARC-AGI-2) dataset contains:

- **1,000 training tasks** — demonstrate the task format and Core Knowledge
  priors.
- **120 public evaluation tasks** — for testing models on unseen tasks.

Each task is a JSON file with demonstration input/output pairs and test
input(s).  Grids are rectangular matrices of integers 0–9 (up to 30×30).
The goal is to produce the correct output grid by inferring the
transformation rule from the demonstrations.

## Project Structure

```
fusion_model/
    __init__.py         # Package re-exports
    model.py            # FusionModel orchestrator (grid encoder + cross-attention + fusion)
    memory.py           # RuleMemory (low-rank rule bank + differentiable retrieval)
    rule_engine.py      # RuleGenerator (ephemeral hypothesis proposer)
    guess.py            # GuessComponent (self-attention predictor)
    decision.py         # DecisionRouter (softmax mixture weights)
    loss.py             # FusionLoss (task + regularisation terms)
tasks/
    arc.py              # ARC-AGI-2 dataset loader and grid utilities
tests/
    test_components.py  # Unit tests for all model components
train.py                # Training loop
evaluate.py             # Evaluation + matplotlib visualisations
pyproject.toml          # UV project config (deps, ruff, mypy, pytest)
Makefile                # Unix make targets
Make.ps1                # PowerShell equivalent
.pre-commit-config.yaml # Pre-commit hooks (ruff + mypy)
```
