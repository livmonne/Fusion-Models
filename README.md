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
     p(y|x) = α_mem·p_mem + α_rule·p_rule + α_guess·p_guess
```

### Components

| Component | Role |
|---|---|
| **RuleMemory** | Bank of 128 learned low-rank rules with trigger embeddings. Differentiable soft-attention retrieval **gated by memory strength** (frequency × recency). Learnable decay/reinforcement rates and recency half-life. Automatic pruning of forgotten slots. Supports soft-blend commitment of proposed rules. Exposes blended correction vector for routing. |
| **RuleGenerator** | Proposes ephemeral one-shot rules as low-rank corrections per input. Uses a three-stage cross-attention pipeline (history, decision, synthesis) over a circular history buffer to propose persistent rules with a learned soft commit weight. Exposes correction vector for routing. |
| **GuessComponent** | Self-attention over pseudo-tokens followed by an MLP head for fuzzy patterns. Exposes pooled representation for routing. |
| **DecisionRouter** | Multi-head cross-attention router (4 heads): uses ``h`` as query and each pathway's intermediate representation as keys *and* values.  A residual connection adds ``h`` back to the attention context, followed by LayerNorm, so the routing MLP always sees both the raw input and the pathway-informed context.  A two-layer MLP (Linear → GELU → Linear) maps the normalised vector to three routing logits, enabling nonlinear feature interactions.  Per-head attention weights are returned for interpretability. |

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
on the specific image-question pair.

#### Why multiple heads?

With a single attention head, the router can only evaluate the pathways
along one learned comparison axis (e.g. "which pathway's key is most
similar to my query?").  With multiple heads, each head operates in its own
**subspace** and can learn a different evaluation criterion:

- One head might focus on **semantic relevance** — which pathway's
  representation aligns best with the current input.
- Another might track **confidence signals** — detecting when a pathway's
  value representation indicates high certainty.
- A third might evaluate **complementarity** — identifying when pathways
  are saying contradictory things and a tiebreaker is needed.

The multi-head attention combines these perspectives into a single rich
context vector before the final routing decision.

#### How routing weights are produced

1. **Multi-head attention** — `h` queries over the 3 pathway tokens
   (key = value = pathway representations) across `num_heads` independent
   subspaces.  This produces a context vector of shape `(batch, embed_dim)`
   that encodes *what information* the router extracted from the pathways.

2. **Residual connection + LayerNorm** — the original shared embedding `h`
   is added back to the attention context and the sum is normalised:
   `fused = LayerNorm(context + h)`.  This ensures the routing MLP always
   has direct access to the raw input alongside the pathway-informed
   context, and provides a clean gradient path back to the upstream
   encoders without bottlenecking through the attention softmax.

3. **Two-layer MLP** — a Linear → GELU → Linear network maps the
   normalised vector to 3 logits (one per pathway).  The hidden nonlinearity
   lets the router learn feature interactions that a single linear projection
   cannot capture (e.g. "memory confidence is high *and* the question is
   about counting").

4. **Temperature-scaled softmax** — the logits are divided by a learnable
   temperature `τ` and passed through softmax to produce the final routing
   coefficients `α ∈ [0, 1]³` that sum to 1.

The key insight is that the routing decision passes through the **value
projections**, not just the attention weights.  Each pathway's key controls
*when* it attracts attention, but its value controls *what it communicates*
to the routing decision.  This separation lets the router make decisions
based on richer information than dot-product similarity alone.

#### Interpretability

The router returns two outputs:

| Output | Shape | Description |
|---|---|---|
| `alpha` | `(batch, 3)` | Final routing coefficients used for the mixture. |
| `attn_weights` | `(batch, num_heads, 3)` | Raw per-head attention distributions over the three pathways. |

The per-head attention weights reveal *how* each head is evaluating the
pathways — even though they are not directly used as routing coefficients,
they show which pathways each head considers relevant.  Comparing
`attn_weights` against `alpha` can reveal how the value projections and
`alpha_proj` layer transform raw attention into final routing decisions.

### How the Pieces Fit Together

The architecture follows a **perceive → specialise → arbitrate** pipeline:

1. **Perceive** — the image encoder and question encoder independently
   process their respective modalities.  Multi-head spatial attention lets
   the question "look at" different image regions (e.g. the subject vs. a
   reference object), producing a single attended image vector.

2. **Fuse** — the attended image vector and question embedding are combined
   via element-wise product (capturing multiplicative interactions) followed
   by a linear projection.  A residual connection from the question
   embedding ensures that raw linguistic signal is always available
   downstream, even if the multiplicative fusion loses it.

3. **Specialise** — three expert pathways process the fused embedding `h`
   independently.  Each pathway is designed for a different reasoning
   strategy:
   - **RuleMemory** retrieves and applies stored rules — good for
     recurring patterns the model has seen before.
   - **RuleGenerator** synthesises one-shot rules on the fly — good for
     novel situations that require compositional reasoning.
   - **GuessComponent** uses self-attention over learned pseudo-tokens —
     a flexible fallback for fuzzy pattern matching.

4. **Arbitrate** — the DecisionRouter uses multi-head cross-attention to
   decide how much to trust each pathway for the current input.  The final
   prediction is a soft mixture weighted by the routing coefficients.

5. **Consolidate** — during training, the RuleGenerator can propose new
   rules for permanent storage in the RuleMemory.  A learned commit weight
   controls how aggressively new rules overwrite weak memory slots,
   creating a feedback loop where successful ephemeral rules graduate into
   long-term memory.

### Loss Function

The training objective balances seven terms:

| Term | Purpose | Effect |
|---|---|---|
| **Task loss** | Cross-entropy on the blended output | Main learning signal |
| **Guess penalty** | Penalises `mean(α_guess)` | Prevents over-reliance on the guess fallback |
| **Storage cost** | Approximate L0 over slot usage | Encourages sparse, specialised memory slots |
| **Entropy bonus** | Maximises `H(α)` | Prevents routing collapse early in training |
| **Auxiliary losses** | Cross-entropy on each pathway's own logits | Keeps all pathways learning even when the router ignores them |
| **Commitment reg.** | Penalises deviation from target commit rate | Prevents the rule proposer from committing too aggressively or never |
| **Strength reg.** | Penalises deviation from target mean strength | Prevents total amnesia or total saturation of memory slots |

The entropy bonus and guess penalty work in tension: entropy encourages
uniform routing (explore all pathways), while the guess penalty discourages
one specific pathway.  Together they push the router toward a balanced
exploration of the rule-based pathways.

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
