# Fusion Model — Few-Shot Abstract Reasoning on Grid Tasks

A neural architecture that combines **rule memory**, **rule generation**,
a **history-based rule proposer**, and **deep spatial guessing** into a
single end-to-end trainable model, arbitrated by a **learned decision
router** that is grounded in **measured demo verification**.  Developed
and tested on the ARC format ([ARC-AGI](https://github.com/fchollet/ARC-AGI) /
[ARC-AGI-2](https://github.com/arcprize/ARC-AGI-2)), but the architecture
applies to any few-shot grid-transformation domain.

## The Task Format

Each task gives the model a handful of **demonstration pairs** — an input
grid and the correct output grid — plus a fresh **test input**.  The model
must infer the hidden transformation rule from the demos and apply it to
the test input in a single forward pass.

Grids are small rectangular matrices (up to 30×30) of integers 0–9,
visualised as coloured cells.  **The output grid's size may differ from
the input's** — the model predicts the output dimensions too.

**Why is it hard?**  Unlike typical ML benchmarks where you can memorise
statistical patterns across thousands of examples, each task is
essentially a *new puzzle*.  The model must learn to learn — extracting a
rule from just 2–5 examples and applying it immediately.

## High-Level Idea

The Fusion Model runs **four specialist pathways** in parallel and lets a
**router** decide how to blend their answers:

1. **RuleMemory** (`mem`) — a persistent bank of reusable rules learned
   over time, retrieved by matching the current *task*.
2. **RuleGenerator** (`rule`) — invents a brand-new one-shot rule on the
   fly from the current task embedding.
3. **RuleProposer** (`prop`) — synthesises a rule from a buffer of *past
   attempts and their outcomes* — the module that learns from its own
   mistakes.  Its rule is **tried on every input** (it is a routed
   pathway), and rules that demonstrably work are committed into the
   RuleMemory bank.
4. **GuessComponent** (`guess`) — a deep pattern-matcher that handles
   anything too fuzzy for explicit rules.

The final output is a weighted mixture of all four, where the weights are
produced per-input by the router — informed by both learned signals *and*
each pathway's **measured ability to reproduce a held-out demonstration**
(leave-one-out verification).  Verification is what makes "rule" mean
something: a rule is good if and only if it reproduces demos it did not
see.

## Architecture

The following diagram traces a single forward pass from raw grids to the
final blended prediction.

```
Demo Pairs (input/output grids)                Test Input Grid
       │                                              │
  Cell Embedding (10 colours → embed_dim)        Cell Embedding
  + 2-D Sinusoidal Positional Encoding           + 2-D Pos Enc
  + Type Embedding (demo_in / demo_out)          + Type Embed (test_in)
  + Demo-Index Embedding                              │
       │                                              │
  Transformer Encoder (per pair,                      │
  padding masked, × N layers)                         │
       │                                              │
       ├── masked mean per pair ──► t (task embedding,│B, E)
       │                            "what is the rule?"
       │                                              │
  Demo Context (all demo tokens, pad-masked)          │
       │                                              │
       └── Multi-Head Cross-Attention (× N layers) ──┘
           test tokens query demo context
                      │
               x = cross-attended test tokens (B, seq, E)
               h = masked mean pool → instance embedding (B, E)
                      │
       ┌──────────────┼──────────────────┬─────────────────┐
       ▼              ▼                  ▼                 ▼
  RuleMemory      RuleGenerator     RuleProposer      GuessComponent
  (x, t)          (x, t)            (x, history)      (x, t, H, W)
  retrieve slots  generate A,B      synthesise A,B    local/global attn
  blend low-rank  one-shot          from past         + FiLM(t)
  corrections     correction        (task, decision,
       │              │              outcome) triples     │
  head(x + corr)  head(x + corr)    head(x + corr)    MLP head
       │              │                  │                │
  (B, seq, 10) + pooled representation for the router (each pathway)
       │              │                  │                │
       └──────────────┴────────┬─────────┴────────────────┘
                               │
        Leave-one-out verification (optional, default on):
        hold out one demo, predict its output with all four
        pathways → per-pathway fit = −CE on the held-out demo
                               │
       DecisionRouter: multi-head cross-attention (query = h,
       keys/values = pathway reprs) → MLP → logits
       + fit_scale · fit  →  softmax / τ  →  α (B, 4)
                               │
     Blended logits = α₀·mem + α₁·rule + α₂·prop + α₃·guess
     Per-cell colour predictions (B, seq, 10)
                               │
     Size head: MLP([t ‖ h]) → output height & width logits
```

### Components at a Glance

| Component | What it does (one sentence) |
|---|---|
| **Cell Embedding + Pos Enc** | Converts each grid cell (0–9) into a vector; adds 2-D positional, grid-type, and demo-index information. |
| **Transformer Encoder** | Reads concatenated demo input+output pairs via self-attention (padding masked) so the model can understand *how* the input was transformed. |
| **Task embedding `t`** | Masked mean of the encoded demo pairs — a summary of the *transformation* itself, independent of the test instance; conditions retrieval, generation, and FiLM. |
| **Multi-Head Cross-Attention** | Lets the test input tokens "ask questions" of the demo context — the core mechanism that transfers the inferred rule to the test input. |
| **RuleMemory** | A persistent bank of low-rank rules (64 slots by default), each with a learned trigger key; retrieved via multi-head cross-attention from `t`, blended, and decoded through a residual base path.  Slot strength decays and reinforces over time (biologically-inspired). |
| **RuleGenerator** | Generates a one-shot low-rank correction from `t` — a fresh hypothesis for this specific task. |
| **RuleProposer** | Cross-attends over a circular history of `(task, decision, outcome)` triples to synthesise a rule from *what worked before*; the rule is applied as a fourth routed pathway, and successful ones are committed to RuleMemory. |
| **GuessComponent** | A deep spatial predictor: Transformer layers alternating *local* (windowed) and *global* self-attention, FiLM-conditioned on `t`. |
| **Verification** | Holds out one demonstration, predicts its output with every pathway, and measures the real cross-entropy — a grounded quality signal for routing and an extra training signal for all pathways. |
| **Size head** | Predicts the output grid's height and width from `[t ‖ h]` — required because outputs frequently differ in size from inputs. |
| **DecisionRouter** | Multi-head cross-attention over pathway representations plus the measured verification fit, producing routing coefficients `α ∈ [0, 1]⁴` that sum to 1. |

---

## Component Deep Dives

### 1. Grid Encoding

- **Cell embedding**: a lookup table maps each colour value (0–9) to a
  learned vector of size `embed_dim` (default 256).  Padding cells
  (value −1) map to a dedicated PAD embedding.
- **2-D sinusoidal positional encoding**: the first half of each vector
  encodes the row position, the second half the column.
- **Type embedding**: distinguishes demo-input, demo-output, and
  test-input grids.
- **Demo-index embedding**: distinguishes tokens from different
  demonstrations inside the concatenated demo context, so the model can
  align patterns *within* a single demo pair.

### 2. Demo Pair Encoding and the Task Embedding `t`

For each demonstration pair, the input and output token sequences are
concatenated and fed through a shared Transformer encoder with
**key-padding masks** (padding cells are excluded from attention).  Two
products come out of this stage:

- the **demo context** — all encoded demo tokens, used as keys/values for
  cross-attention (absent demos and padding cells are masked);
- the **task embedding `t`** — each pair is masked-mean-pooled, the pools
  are averaged over valid demos, and the result is projected.  Because
  `t` is built *purely from the demonstrations*, it describes the
  transformation rule rather than any particular input — which is exactly
  the right key for rule retrieval and generation.  (The *instance*
  embedding `h`, pooled from the cross-attended test tokens, is kept
  separate and used for routing and size prediction.)

### 3. Cross-Attention (Test ← Demo Context)

The test input tokens cross-attend to the demo context over multiple
layers (4 by default), with invalid context positions masked.  Each test
cell asks "given what I saw in the demos, what should I become?"

### 4. Expert Pathway: RuleMemory

**Idea**: store a library of reusable transformation rules and look up the
right ones for each task.

Each memory slot stores a **key** (trigger embedding) and two thin
matrices **A, B** whose product `A @ (B @ x)` is a LoRA-style low-rank
correction applied per token.  Retrieval is **multi-head cross-attention**
from the task embedding `t` over the key bank: each head independently
assesses relevance and a learned combination merges them into per-slot
scores.

The score-weighted blend of slot corrections is added back onto the token
stream and decoded by a shared head:

```
logits_mem = head(LayerNorm(x + Σ_s score_s · A_s(B_s x)))
```

The **residual base path** matters: an earlier design decoded each slot's
correction in isolation, which forced every prediction through a rank-16
bottleneck (LoRA without the base weights) and structurally handicapped
the rule pathways against the guess pathway.  Blending before decoding
also avoids materialising a `(batch, seq, slots, embed)` tensor, keeping
peak memory low.

**Memory strength** is modulated by two signals inspired by neuroscience:

| Signal | Captures | Mechanism |
|---|---|---|
| **Frequency** | How often a slot is used | Running accumulator with learnable decay + reinforcement rates |
| **Recency** | How recently a slot was activated | Exponential decay with a learnable half-life |

`strength = frequency × recency` — weak slots fade and are eventually
**pruned** (reset to a no-op rule), recycling capacity.  The dynamics are
**tuned to the slot count**: retrieval scores are a softmax, so the mean
slot score is `1/num_slots`; the reinforcement rate is initialised so an
average slot equilibrates at strength ≈ 0.5 rather than hovering at the
prune threshold, and the recency-activation threshold is *relative* to
the uniform share (2× by default).  Pruning runs rarely (every 500 steps)
because it resets parameters the optimiser still has momentum for.

### 5. Expert Pathway: RuleGenerator

**Idea**: sometimes no stored rule fits, so we invent one on the spot.

An MLP takes the task embedding `t` and outputs A and B matrices for a
one-shot low-rank correction.  The correction is applied per token and
decoded through the same residual base-path pattern
(`head(LayerNorm(x + correction))`).  It exists only for this forward
pass.

### 6. Expert Pathway: RuleProposer (history → rule, tried every step)

**Idea**: keep a record of what was attempted and how it went, and distil
recurring successes into explicit rules.  This is the pathway that *learns
from its own mistakes*.

The proposer maintains a **circular history buffer** of
`(task, decision, outcome)` triples:

- **task** — the task embedding `t` of a past sample;
- **decision** — what the model did: the colour histogram of its
  prediction concatenated with the routing weights `α` (which pathway it
  trusted);
- **outcome** — the per-sample task loss, **standardised over the buffer
  at read time** so the proposer sees relative quality rather than the
  shrinking absolute loss scale.

A three-stage cross-attention pipeline — operating entirely on history,
*not* the current input — synthesises a rule `(key, A, B)`:

1. Historical task embeddings attend over historical decisions.
2. The result attends over the standardised outcome signals.
3. A learned synthesis query fuses the two.

**The crucial design point:** the proposed rule is **applied to the
current input** and decoded into its own per-cell logits
(`head(LayerNorm(x + correction))`), which the router can select and the
auxiliary loss grades.  The proposer therefore "tries" a rule on every
forward pass, and task gradient flows back through the entire
history-attention pipeline.  (An earlier revision only consumed proposals
through a no-grad commit, which left the proposer's rule content
untrained — the machinery looked sophisticated but learned nothing.)

**Commitment** into RuleMemory is **outcome-gated and deferred**.  After
the backward pass, the training loop hands the model the proposal
pathway's *measured* per-sample loss (`model.apply_outcomes`).  A rule is
committed only when the best sample in the batch beat an exponential
moving average of recent proposal losses — i.e. rules are stored because
they demonstrably worked, not because a similarity heuristic fired.  A
learned commit weight (regularised toward a target rate) controls the
soft blend into the weakest memory slot.  Deferring commits to the
optimiser-step boundary also keeps the in-place slot update from
corrupting gradients mid-step.

The proposal pathway activates once the history buffer holds
`min_history` entries (64 by default); until then the router masks its
weight to exactly zero.

### 7. Expert Pathway: GuessComponent

**Idea**: not every pattern can be captured by a crisp rule.

A stack of Transformer layers (3 by default) with:

1. **Alternating local/global attention** — even layers restrict
   attention to a Chebyshev-distance window on the 2-D grid; odd layers
   are global.  Padding cells are masked out of attention (every token
   keeps its self-connection, so no attention row is ever fully masked).
2. **FiLM conditioning** — after each layer, the task embedding `t`
   produces per-channel scale and shift, injecting "what kind of task is
   this" into the spatial representations.

### 8. Leave-One-Out Verification (grounded routing)

The demonstrations contain ground truth the model can check itself
against — the architecture uses them for exactly that:

1. One demonstration `d*` is held out (random per sample during training;
   averaged over all demos during evaluation).
2. Its tokens are removed from the cross-attention context and the task
   pooling; its *input* grid is embedded like a test input and
   cross-attended to the remaining demos.
3. All four pathways predict the held-out *output*, and the per-pathway
   cross-entropy is measured.

The measured fit (centred negative CE) is:

- **fed to the router** through a learned scale — an *objective* "this
  pathway actually reproduced a demo it didn't see" signal on top of the
  learned routing; and
- **added to the loss** — every pathway is trained to transform held-out
  demo inputs into demo outputs, which is a free, perfectly-labelled
  augmentation of exactly the right skill.

Samples with fewer than two demonstrations skip verification (their fit
is zero and they are excluded from the verification loss).  Disable with
`--no_verify` to save ~40% step time at the cost of the grounded signal.

### 9. Output Size Head

A small MLP reads `[t ‖ h]` and classifies the output grid's height and
width (1–30 each).  At inference, predictions are cropped to the
predicted size; when the predicted output exceeds the test-input canvas,
the sample is re-run on an enlarged canvas so every output cell has a
token position (two-pass inference in `submit.py`).

### 10. DecisionRouter

The router decides how much each expert contributes:

1. **Multi-head attention** — `h` queries the four pathway
   representations (key = value = pathway pooled outputs).
2. **Residual + LayerNorm**, then a **two-layer MLP** maps to 4 logits.
3. **Verification bias** — `fit_scale · fit` is added to the logits when
   verification ran.
4. **Temperature-scaled softmax** produces `α ∈ [0, 1]⁴`; the inactive
   proposal pathway is masked to exactly zero.

Final prediction: `logits = α₀·mem + α₁·rule + α₂·prop + α₃·guess`.

---

## Loss Function

The training objective balances nine terms:

| Term | What it penalises | Why it matters |
|---|---|---|
| **Task loss** | Per-cell cross-entropy (ignoring padding) | Main learning signal — predict the right colours |
| **Size loss** | CE on predicted output height/width | Without it, only same-size transformations are possible |
| **Verification loss** | Per-pathway CE on the held-out demo | Trains every pathway to actually *reproduce* demonstrations |
| **Guess penalty** | `mean(α_guess)` | Prevents lazily falling back on pattern matching |
| **Storage cost** | Entropy of the retrieval distribution | Peaky retrieval = specialised slots (an earlier `u·(1−u)` form rewarded winner-take-all collapse) |
| **Entropy bonus** | Negative `H(α)` | Prevents routing collapse — keeps pathways alive early |
| **Auxiliary losses** | Per-cell CE on each pathway's *own* logits | Keeps all pathways learning even when the router ignores one |
| **Commitment reg.** | Deviation from target commit rate | Prevents the proposer from committing too aggressively (or never) |
| **Strength reg.** | Deviation from target mean strength | Prevents total amnesia or total saturation of memory slots |

---

## Quick Start

```bash
# Install dependencies (requires uv: https://docs.astral.sh/uv/)
make install

# Clone an ARC dataset
git clone https://github.com/arcprize/ARC-AGI-2.git arc-agi-2
ln -s arc-agi-2/data data

# Run linter + type-checker
make lint

# Run tests
make test

# Quick debug run (fits a 16 GB laptop; 16x16 grid subset)
make train

# Full training (JSON dataset)
uv run python train.py --data_root data --epochs 40

# Train with augmented parquet dataset on a single GPU (e.g. A40):
uv run python train.py --parquet_dir data/parquet --data_root data \
    --batch_size 8 --grad_accum 32 --epochs 2 --amp --val_every 2000

# Evaluate and generate visualisation plots
uv run python evaluate.py --data_root data

# Generate a Kaggle submission
uv run python submit.py --challenges arc-agi_evaluation_challenges.json
```

### Hardware guidance

Attention cost is dominated by the demo context (up to
`max_demos × 2 × H × W` keys), so the grid-size cap is the main knob:

| Hardware | Suggested settings |
|---|---|
| **MacBook (M1/M2, 16 GB)** | `--max_grid_size 16 --batch_size 2 --num_workers 0` — trains on the small-grid subset; great for development and small-scale experiments |
| **Single A40/A6000-class GPU (40–48 GB)** | full `--max_grid_size 30`, `--batch_size 8 --grad_accum 32 --amp` (bfloat16 autocast + fused SDPA attention) |

Tasks with grids larger than `--max_grid_size` are **skipped, not
truncated**, so a reduced canvas trains on a consistent subset (the
loaders print how many tasks were filtered).

### Training dynamics

- **Per-step LR schedule**: linear warmup (default `min(2000, 3%)` of
  total optimiser steps) then cosine annealing to 5% of the peak LR.
  Per-epoch scheduling is useless when one epoch is hundreds of
  thousands of steps.
- **Mid-epoch validation**: `--val_every N` validates (and checkpoints)
  every N optimiser steps — strongly recommended for million-sample
  parquet datasets.
- **Gradient accumulation**: `--grad_accum N` sets the effective batch
  size.  Rule commits only happen on optimiser-step boundaries.
- **Checkpoint selection**: best `(exact-match solve rate, cell
  accuracy)` — cell accuracy alone is flattering (90% cells can still be
  0% solved tasks).
- The per-epoch log prints the four routing weights, per-pathway
  auxiliary losses, the verification loss, and the number of rules
  committed — watch `alpha(...)` to see the router's division of labour
  emerge.

## Data

### JSON (ARC format)

```
data/
    training/       # task JSON files
    evaluation/     # task JSON files (always used for validation)
```

Each task is a JSON file with demonstration input/output pairs and test
input(s).

### Augmented Data (Parquet)

The training script also supports loading augmented datasets stored as
`.parquet` files (e.g. the
[Giotto ARC-AGI dataset](https://zenodo.org/records/18508333) or
[re-ARC](https://github.com/michaelhodel/re-arc) exports).  Each parquet
file must have columns `id` (string) and `task` (JSON string in the
standard ARC format).

Use `--parquet_dir data/parquet/` to load all parquet files in a
directory, or `--parquet_files file1.parquet file2.parquet` to select
specific files.  Validation always reads from `data/evaluation/` (JSON).

The parquet loader is **lazy and worker-safe**: only a lightweight index
is built at init; each DataLoader worker re-opens the files on first
access (no multi-gigabyte tables are pickled to spawned workers), and
JSON is parsed on-the-fly per sample.

### Recommended experiment: stratified rule generalisation

The architecture makes a sharp, testable prediction: **RuleMemory should
win on new instances of *seen* rules, while RuleGenerator/Guess should
win on *unseen* rules** — with the router's α shifting accordingly.  To
test it, don't train on the full augmented dump; stratify it:

1. Sample N rule families × M instances from re-ARC (e.g. 100 × 500).
2. Hold out (a) unseen instances of seen rules and (b) entire unseen
   rule families.
3. Compare solve rates and mean α per pathway on (a) vs (b), and check
   slot specialisation in `rule_utility.png` / the retrieval head maps.

This gives a direct readout of the fusion hypothesis at a fraction of
the compute of training on millions of samples.

## Project Structure

```
fusion_model/
    __init__.py         # Package re-exports (incl. PATHWAY_NAMES order)
    common.py           # Shared helpers: masked pooling, per-sample CE
    model.py            # FusionModel orchestrator (encoding, task embedding,
                        #   cross-attention, verification, size head, fusion)
    memory.py           # RuleMemory (low-rank rule bank + multi-head retrieval
                        #   + strength dynamics + pruning)
    rule_engine.py      # RuleGenerator (ephemeral rules) + RuleProposer
                        #   (history buffer, proposal pathway, commit gating)
    guess.py            # GuessComponent (FiLM + local/global attention)
    decision.py         # DecisionRouter (4-way, verification-grounded)
    loss.py             # FusionLoss (task + size + verification + 6 regularisers)
tasks/
    arc.py              # Dataset loaders (JSON + lazy parquet), grid utilities,
                        #   size filtering
tests/
    test_components.py  # Unit + regression tests for all components
train.py                # Training loop (AdamW, per-step warmup/cosine, AMP,
                        #   grad accumulation, outcome feedback, solve-rate val)
evaluate.py             # Evaluation + matplotlib visualisations
submit.py               # Kaggle submission generator (two-pass size inference)
pyproject.toml          # UV project config (deps, ruff, mypy, pytest)
Makefile                # Unix make targets
Make.ps1                # PowerShell equivalent
.pre-commit-config.yaml # Pre-commit hooks (ruff + mypy)
```
