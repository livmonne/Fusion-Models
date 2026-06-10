# Fusion Model — Abstract Reasoning on ARC-AGI-2

A neural architecture that combines **rule memory**, **rule generation**,
**deep spatial guessing** (with local attention and FiLM conditioning), and
a **learned decision router** into a single end-to-end trainable model,
applied to the [ARC-AGI-2](https://github.com/arcprize/ARC-AGI-2) abstract
reasoning benchmark.

## What is ARC-AGI-2?

ARC-AGI-2 (Abstraction and Reasoning Corpus) is a benchmark designed to
test an AI's ability to *generalise* from very few examples.  Each task
gives you a handful of **demonstration pairs** — an input grid and the
correct output grid — plus a fresh **test input**.  Your job is to figure
out the hidden transformation rule from the demos and apply it to the test
input to produce the right output.

Grids are small rectangular matrices (up to 30×30) of integers 0–9,
visualised as coloured cells.  There are 10 possible colours.

**Why is it hard?**  Unlike typical ML benchmarks where you can memorise
statistical patterns across thousands of examples, each ARC task is
essentially a *new puzzle*.  The model must learn to learn — extracting a
rule from just 2–5 examples and applying it in a single forward pass.

## High-Level Idea

The Fusion Model tackles this by running **three specialist pathways** in
parallel and letting a **router** decide how to blend their answers:

1. **RuleMemory** — a persistent bank of reusable rules learned over time.
2. **RuleGenerator** — invents brand-new one-shot rules on the fly.
3. **GuessComponent** — a deep pattern-matcher that handles anything too
   fuzzy for explicit rules.

The final output is a weighted mixture of all three, where the weights are
learned per-input by the router.

## Architecture

The following diagram traces a single forward pass from raw grids to the
final blended prediction.

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
       ┌──────┼────────────────┼──────────────────┐
       ▼      ▼                ▼                   ▼
  RuleMemory(x, h)    RuleGenerator(x, h)  GuessComponent(x, h, H, W)
  (MH cross-attn
   retrieval, 4 heads)
       │                    │                      │
  (B, seq, 10)        (B, seq, 10)           (B, seq, 10)
  + mem_repr (B,E)    + rule_repr (B,E)      + guess_repr (B,E)
       │                    │                      │
       ▼                    ▼                      ▼
       ┌────────────────────┼──────────────────────┐
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

### Components at a Glance

| Component | What it does (one sentence) |
|---|---|
| **Cell Embedding + Pos Enc** | Converts each grid cell (0–9) into a vector, adds 2-D positional info so the model knows where each cell sits. |
| **Transformer Encoder** | Reads concatenated demo input+output pairs via self-attention in a single batched pass (all valid demos at once) so the model can understand *how* the input was transformed. |
| **Multi-Head Cross-Attention** | Lets the test input tokens "ask questions" of the demo context — this is the core mechanism that transfers the inferred rule to the test input. |
| **RuleMemory** | A persistent bank of 128 reusable low-rank rules, each with a learned trigger.  Rules are matched via **multi-head cross-attention** (4 heads by default) and applied per-cell.  Slot strength decays and reinforces over time (biologically-inspired). |
| **RuleGenerator** | Invents a one-shot low-rank correction on the fly and maintains a history of past attempts to propose persistent rules for the memory bank. |
| **GuessComponent** | A deep spatial predictor: a stack of Transformer layers that alternate between *local* (windowed) and *global* self-attention, with FiLM conditioning from the pooled task embedding `h`, followed by a per-cell MLP head. |
| **DecisionRouter** | Uses multi-head cross-attention to decide how much weight each pathway gets.  Produces per-input routing coefficients `α ∈ [0, 1]³` that sum to 1. |

---

## Component Deep Dives

### 1. Grid Encoding (Cell Embedding + Positional Encoding)

Before anything else, raw grid values need to be turned into vectors the
model can process.

- **Cell embedding**: a standard lookup table (`nn.Embedding`) maps each
  colour value (0–9) to a learned vector of size `embed_dim` (default 256).
  Padding cells (value −1) map to a special PAD embedding.
- **2-D sinusoidal positional encoding**: following the classic Transformer
  approach but extended to 2D — the first half of each vector encodes the
  row position, the second half encodes the column.  This gives the model
  spatial awareness without learning anything.
- **Type embedding**: a small learned embedding distinguishes three roles —
  demo input, demo output, and test input — so the model knows what kind
  of grid it's looking at.

The result is a sequence of token vectors `(batch, H×W, embed_dim)` for
each grid.

### 2. Demo Pair Encoding (Batched Transformer Encoder)

All demonstration pairs are embedded, concatenated (input + output), and
processed through a shared Transformer encoder **in a single batched
forward pass** — all valid demos across the batch are stacked into one
tensor of shape `(n_valid, 2*H*W, embed_dim)`.  Invalid demo slots
(from the padding mask) are excluded from the encoder pass entirely to
avoid wasting compute.  This gives a ~D× throughput improvement over the
naïve approach of looping over each demo index sequentially.

All encoded demo sequences are then concatenated across demos into a
single **demo context** tensor.

### 3. Cross-Attention (Test ← Demo Context)

The test input tokens cross-attend to the demo context over multiple
layers (4 by default).  In cross-attention, the test tokens are the
**queries** and the demo context provides the **keys** and **values**.

Think of it this way: each test cell is asking "given what I saw in the
demos, what should I become?"  Each cross-attention head can focus on a
different aspect — one head might track colour mappings, another might
track spatial shifts.

After cross-attention, we have:
- `x` — per-cell spatial tokens `(batch, seq, embed_dim)`
- `h` — a global summary vector obtained by mean-pooling `x` (masking out
  padding) and projecting it through a small MLP.

Both `x` and `h` are passed to all three expert pathways.

### 4. Expert Pathway: RuleMemory

**Idea**: store a library of reusable transformation rules and look up the
right ones for each input.

Each of the 128 memory slots stores:
- A **key** (trigger embedding) — controls *when* this rule fires.
- Two small matrices **A** and **B** — their product `A @ (B @ x)` is a
  low-rank correction applied to each token.  This is the same idea as
  LoRA: instead of a full weight matrix, we store two thin factors.
  During the forward pass, retrieval scores are folded into A/B *before*
  expanding over the spatial dimension, so peak memory is `O(B × seq × E)`
  instead of the naïve `O(B × seq × S × E)` (a ~128× reduction for
  default settings).
- A **classification head** — converts the corrected embedding into
  per-cell colour logits.

#### Multi-Head Cross-Attention Retrieval

Retrieval uses **multi-head cross-attention** (4 heads by default) rather
than a single dot-product.  The pooled task embedding `h` is the *query*;
the slot keys serve as both *keys* and *values* through separate learned
projections.

```
h  (B, E) ──► Q projection ──► Q (B, H, head_dim) ─┐
                                                     ├─► scaled dot-product attention
keys (S, E) ──► K projection ──► K (S, H, head_dim) ─┘
            ──► V projection ──► V (S, H, head_dim)
                                        │
                          softmax per head → head_attn (B, H, S)
                                │                    │
                         weighted sum of V      learned head combination
                                │                    │
                        retrieved (B, H, D)     scores (B, S)
                                │
                    concat heads → out_proj → context (B, E)
```

**Why multiple heads?**  A single dot-product compresses "relevance" into
one number per slot.  With *H* heads, the model gets *H* independent
channels to assess relevance — one head might attend to colour
transformations, another to spatial layout, a third to symmetry patterns.
The per-head attention maps are merged into final slot scores via a
*learned* head-combination vector (softmax over `H` learnable weights),
so the model can up-weight the most informative heads over training.

The cross-attention also produces a **context vector** (the standard MHA
output) that is added to the mean-pooled blended correction and
LayerNorm'd before being passed to the decision router.  This gives the
router a richer signal about what the memory pathway found.

**Memory strength** is modulated by two signals inspired by neuroscience:

| Signal | Captures | Mechanism |
|---|---|---|
| **Frequency** | How often a slot is used | Running accumulator with learnable decay + reinforcement rates |
| **Recency** | How recently a slot was activated | Exponential decay with a learnable half-life |

`strength = frequency × recency` — slots that are rarely used or haven't
been triggered recently will fade, and once they drop below a threshold
they are **pruned** (reset with fresh random parameters), recycling
capacity for new rules.

All three dynamics parameters (decay rate, reinforcement rate, half-life)
are **learnable** — the model discovers its own optimal
forgetting/consolidation schedule via gradient descent.

### 5. Expert Pathway: RuleGenerator

**Idea**: sometimes no stored rule fits, so we invent one on the spot.

An MLP takes the pooled `h` and directly outputs A and B matrices for a
one-shot low-rank correction, plus a scalar confidence.  The correction is
applied per-token just like in RuleMemory, but it exists only for this
forward pass.

The generator also maintains a **circular history buffer** of recent
`(embedding, decision, outcome)` triples.  Decision identifiers are
computed by a **learned linear projection** from the mean per-cell logits
to a compact vocabulary index, which is far less lossy than a simple
hash.  A three-stage cross-attention pipeline — operating entirely on
history, *not* the current input — proposes persistent rules:

1. Historical embeddings attend over historical decisions.
2. That result attends over outcome signals (per-sample loss).
3. A learned synthesis query fuses the two.

The output is a proposed rule (key, A, B) plus a soft **commit weight**.
During training, the proposed rule is **soft-blended** into the weakest
memory slot, giving the bank a warm start for newly discovered patterns.

### 6. Expert Pathway: GuessComponent (Deep Spatial Predictor)

**Idea**: not every pattern can be captured by a crisp rule.  Sometimes
the model needs to "just look at the grid and figure it out."

The GuessComponent is a stack of Transformer layers (3 by default) with
two key design choices:

1. **Alternating local/global attention** — even-numbered layers restrict
   each token to attending only within a *window* on the original 2-D grid
   (Chebyshev distance, default radius 3 cells).  This forces fine-grained
   local pattern detection.  Odd-numbered layers use standard unrestricted
   global attention for long-range integration.  The alternation gives the
   model both close-up and birds-eye views.  The local attention masks are
   LRU-cached by `(grid_h, grid_w)` to avoid recomputing them every
   forward pass.

2. **FiLM conditioning** — after each layer, the pooled task embedding `h`
   is used to compute per-token scale (`gamma`) and shift (`beta`)
   parameters via Feature-wise Linear Modulation.  This injects global
   task context (the same signal the other pathways receive) into the
   spatial representations, so the guess pathway knows *what kind of task*
   it's working on.

After the layer stack, a per-token MLP head maps to `num_colours` logits,
and the tokens are mean-pooled into a representation for the router.

### 7. DecisionRouter (Multi-Head Cross-Attention Routing)

The router decides how much each expert contributes to the final answer.

**Why cross-attention instead of a simple MLP?**  With an MLP, the routing
decision would be a static function of concatenated representations.
Cross-attention is fundamentally different: `h` acts as a *query* that
asks each pathway "what can you offer for this input?"  The routing
decision is therefore **input-dependent by construction**.

How it works:

1. **Multi-head attention** (4 heads) — `h` queries the 3 pathway
   representations (key = value = pathway pooled outputs).
2. **Residual + LayerNorm** — adds `h` back and normalises.
3. **Two-layer MLP** — maps to 3 logits (one per pathway).
4. **Temperature-scaled softmax** — produces routing weights
   `α ∈ [0, 1]³` that sum to 1.  The temperature is learnable.

The final prediction: `logits = α₀·logits_mem + α₁·logits_rule + α₂·logits_guess`.

---

## How the Pieces Fit Together

The architecture follows a **perceive → specialise → arbitrate** pipeline:

1. **Perceive** — grid cells are embedded with positional and type
   information.  Demo pairs are encoded by a shared Transformer encoder.
   Cross-attention transfers the inferred transformation from demos to the
   test input, producing spatial tokens `x (batch, seq, E)`.

2. **Pool** — the cross-attended test tokens are masked mean-pooled
   (ignoring padding cells) and then projected through a learned linear
   layer + GELU activation to produce the global summary vector
   `h (batch, E)`.  Both `x` and `h` are passed downstream.

3. **Specialise** — three expert pathways process the spatial tokens `x`
   independently (all also receive `h`), each producing per-cell colour
   logits `(batch, seq, 10)`:
   - **RuleMemory** retrieves and applies stored rules per token.
   - **RuleGenerator** synthesises one-shot rules and applies them per
     token.
   - **GuessComponent** runs deep local/global attention with FiLM for
     fuzzy pattern matching.

4. **Arbitrate** — the DecisionRouter uses multi-head cross-attention
   (querying each expert's pooled representation with `h`) to produce
   per-sample routing weights `α`.  The final prediction is a soft
   mixture: `logits = α₀·mem + α₁·rule + α₂·guess` applied per cell.

5. **Output** — the blended logits are per-cell colour predictions
   `(batch, seq, num_colours)` for the output grid.

6. **Consolidate** (training only) — the RuleGenerator proposes new rules
   for permanent storage in the RuleMemory.  The per-sample loss is fed
   back into the history buffer as an outcome signal so the proposer can
   learn which input→decision pairings were effective.

---

## Loss Function

The training objective balances seven terms.  Each addresses a specific
failure mode:

| Term | What it penalises | Why it matters |
|---|---|---|
| **Task loss** | Per-cell cross-entropy (ignoring padding) | Main learning signal — predict the right colours |
| **Guess penalty** | `mean(α_guess)` | Prevents the model from being lazy and always falling back on guessing |
| **Storage cost** | Approximate L0 over slot usage | Encourages sparse, specialised memory slots instead of using them all |
| **Entropy bonus** | Negative `H(α)` | Prevents routing collapse — keeps all pathways active early in training |
| **Auxiliary losses** | Per-cell CE on each pathway's *own* logits | Keeps all pathways learning even when the router is ignoring one |
| **Commitment reg.** | Deviation from target commit rate | Prevents the rule proposer from committing too aggressively (or never) |
| **Strength reg.** | Deviation from target mean strength | Prevents total amnesia or total saturation of memory slots |

---

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

# Full training (JSON dataset)
uv run python train.py --data_root data --epochs 40

# Train with augmented parquet dataset
uv run python train.py --parquet_dir data/parquet --data_root data --epochs 40

# Train with specific parquet files and gradient accumulation
uv run python train.py \
    --parquet_files data/parquet/seeds_original.parquet data/parquet/rearc.parquet \
    --data_root data --batch_size 4 --grad_accum 32 --epochs 40

# Evaluate and generate visualisation plots
uv run python evaluate.py --data_root data

# Generate a Kaggle submission
uv run python submit.py --challenges arc-agi_evaluation_challenges.json
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

### Augmented Data (Parquet)

The training script also supports loading augmented datasets stored as
`.parquet` files (e.g. the
[Giotto ARC-AGI dataset](https://zenodo.org/records/18508333) with ~1.2M
synthetic tasks).  Each parquet file must have columns `id` (string) and
`task` (JSON string in the standard ARC format).

```
data/
    evaluation/              # 120 JSON files (always needed for validation)
    parquet/                 # augmented parquet files
        seeds_original.parquet
        rearc.parquet
        ...
```

Use `--parquet_dir data/parquet/` to load all parquet files in a directory,
or `--parquet_files file1.parquet file2.parquet` to select specific files.
Validation always reads from `data/evaluation/` (JSON).

The parquet loader uses **lazy loading**: only a lightweight index is built
at init (~20s for 1.2M tasks), and JSON is parsed on-the-fly per sample.
RAM usage stays proportional to the compressed parquet size (~1.3 GB)
rather than the fully materialised Python objects.

### Training Dynamics

The training script includes:

- **LR warmup**: Linear warmup over the first `min(5, epochs // 4)` epochs
  from `lr × 0.01` to `lr`, followed by cosine annealing.
- **Gradient accumulation**: `--grad_accum N` sets the effective batch size
  to `N`.  With `--batch_size 4 --grad_accum 32`, the model processes 4
  samples at a time but accumulates gradients over 8 steps before updating.
  `N` must be >= `batch_size` and divisible by it.
- **Epoch-average routing weights**: The logged routing weights (`alpha`)
  are averaged over the entire epoch, not just the last batch.
- **TPU/XLA auto-detection**: The training script automatically detects
  `torch_xla` and uses XLA devices (TPU) when available.  Falls back to
  CUDA → Apple MPS → CPU in that order.

## Project Structure

```
fusion_model/
    __init__.py         # Package re-exports
    model.py            # FusionModel orchestrator (grid encoder + cross-attention + fusion)
    memory.py           # RuleMemory (low-rank rule bank + differentiable retrieval)
    rule_engine.py      # RuleGenerator (ephemeral hypothesis proposer + history buffer)
    guess.py            # GuessComponent (FiLM-conditioned local/global attention predictor)
    decision.py         # DecisionRouter (cross-attention softmax mixture weights)
    loss.py             # FusionLoss (task + 6 regularisation terms)
tasks/
    arc.py              # ARC-AGI-2 dataset loaders (JSON + Parquet) and grid utilities
tests/
    test_components.py  # Unit tests for all model components
train.py                # Training loop (AdamW + warmup + cosine LR + grad accumulation)
evaluate.py             # Evaluation + matplotlib visualisations
submit.py               # Kaggle submission.json generator
pyproject.toml          # UV project config (deps, ruff, mypy, pytest)
Makefile                # Unix make targets
Make.ps1                # PowerShell equivalent
.pre-commit-config.yaml # Pre-commit hooks (ruff + mypy)
```
