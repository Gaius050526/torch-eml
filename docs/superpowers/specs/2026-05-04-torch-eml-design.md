# torch-eml Design Spec

## Overview

`torch-eml` is an open-source PyTorch library that implements EML (Exp-Minus-Log) tree heads — small, trainable binary trees where every node computes `eml(x, y) = exp(x) - ln(y)`. After training, the tree can be pruned and its weights snapped to clean values, producing a closed-form symbolic equation that is human-readable, auditable, and formally verifiable.

**Goal:** A generic, trunk-agnostic interpretable head that turns any feature extractor (neural net, LLM, classical model) into a neuro-symbolic system that outputs equations, not approximations.

**License:** Apache 2.0

## Background

Odrzywołek (2026) proved that `eml(x, y) = exp(x) - ln(y)`, together with the constant 1, can express every standard elementary function (exp, log, trig, hyperbolic, and their inverses) as a binary tree of identical nodes. This is the continuous-mathematics analog of the NAND gate in digital logic.

Ipek (2026) proposed embedding EML trees as interpretable heads on conventional neural network trunks, showing that weight snapping can collapse trained EML trees into closed-form symbolic expressions.

`torch-eml` is the first open-source implementation of this architecture as a reusable library.

## Architecture

### File Structure

```
torch-eml/
├── torch_eml/
│   ├── __init__.py          # Public API exports
│   ├── node.py              # EMLNode — single eml(x,y) computation
│   ├── tree.py              # EMLTree — binary tree of EMLNodes
│   ├── head.py              # EMLHead — high-level nn.Module
│   ├── symbolic.py          # Weight snapping + symbolic expression extraction
│   ├── pruning.py           # Post-training tree pruning
│   └── trunk.py             # LLM trunk adapters
├── examples/
│   ├── 01_symbolic_regression.py
│   ├── 02_drop_in_head.py
│   └── 03_llm_trunk.py
├── tests/
│   ├── test_node.py
│   ├── test_tree.py
│   ├── test_head.py
│   ├── test_symbolic.py
│   ├── test_pruning.py
│   └── test_trunk.py
├── README.md
├── pyproject.toml
└── LICENSE
```

### Components

#### 1. EMLNode (`node.py`)

A single `nn.Module` computing `eml(x, y) = exp(w_l * x + b_l) - ln(|w_r * y + b_r| + epsilon)`.

- **Parameters:** `w_left`, `w_right`, `bias_left`, `bias_right` (4 learnable floats)
- **Epsilon:** `1e-7` default, configurable, prevents log(0)
- **Gradient flow:** Standard PyTorch autograd through exp and log

#### 2. EMLTree (`tree.py`)

A complete binary tree of `EMLNode` modules at a specified depth.

- **Construction:** `EMLTree(depth=4)` creates 15 nodes (2^depth - 1) and 16 leaf positions (2^depth)
- **Forward pass:** Bottom-up evaluation. Leaf values feed into the deepest nodes, outputs propagate up to the root, producing a single scalar per batch element.
- **Leaf inputs:** Accepts a tensor of shape `[batch, 2^depth]`. Each leaf position maps to one element.
- **Node access:** `tree.nodes` is a `nn.ModuleList`. Nodes are indexed level-order for easy traversal.

#### 3. EMLHead (`head.py`)

The high-level API. An `nn.Module` that wraps leaf projection + EMLTree + output.

- **Constructor:** `EMLHead(n_inputs, depth=4)`
- **Leaf projection:** `nn.Linear(n_inputs, 2^depth)` — maps arbitrary input dimensions to the tree's leaf count. Each leaf becomes a learned weighted combination of all inputs.
- **Forward:** `input [batch, n_inputs] → projection → tree → output [batch, 1]`
- **Convenience methods:**
  - `head.snap(tolerance=0.05, interactive=False)` — prune + snap + return symbolic expression
  - `head.to_symbolic()` — extract symbolic expression from current weights (without snapping)
  - `head.prune(threshold=0.01)` — remove low-contribution branches

#### 4. Symbolic Extraction (`symbolic.py`)

Converts trained EML tree weights into human-readable symbolic expressions.

**Snap targets:**
```python
SNAP_TARGETS = {
    0, 1, -1, 2, -2, 0.5, -0.5,
    math.pi, -math.pi,
    math.e, -math.e,
    1/3, -1/3, 2/3,
    math.sqrt(2), math.log(2),
}
```

**Auto snap** (`snap()`):
1. For each weight, find nearest snap target within tolerance
2. If no target within tolerance, keep raw float
3. Re-evaluate accuracy on validation data (if provided)
4. If accuracy drop exceeds threshold, warn
5. Return `SymbolicExpression` object

**Interactive snap** (`snap(interactive=True)`):
1. For each weight, show top 3 candidate snap values with accuracy impact
2. User selects per-weight or accepts defaults
3. Produce summary report

**SymbolicExpression object:**
- `.string` — human-readable string: `"exp(0.5*x0) - ln(exp(x2) - ln(x5))"`
- `.sympy` — SymPy expression object (can be simplified, differentiated, integrated)
- `.latex` — LaTeX string for papers
- `.python` — pure Python function string (no torch dependency)

#### 5. Pruning (`pruning.py`)

Post-training tree simplification.

- **Contribution scoring:** For each node, measure output variance contribution across a calibration dataset. Nodes whose removal changes the output by less than `threshold` are prunable.
- **Pruning:** Replace prunable subtrees with their mean constant output.
- **Result:** A smaller tree that produces the same output within tolerance.
- **API:** `tree.prune(threshold=0.01, calibration_data=X)` mutates the tree in-place. Returns a `PruneReport` with stats on nodes removed and accuracy impact.

#### 6. LLM Trunk Adapter (`trunk.py`)

Thin wrapper that turns an LLM API into a numerical feature extractor.

**Constructor:**
```python
LLMTrunk(
    provider="anthropic",       # "anthropic" or "openai"
    model="claude-opus-4-6",
    features=[
        {"name": "market_size_log", "description": "log10 of TAM in USD"},
        {"name": "team_years", "description": "combined years of relevant experience"},
    ]
)
```

**How it works:**
- Constructs a system prompt forcing JSON output with the specified feature names as numeric values
- Calls the LLM API with the user-provided text
- Parses JSON response, validates all features are present and numeric
- Retries once on malformed output
- Returns `torch.Tensor` of shape `[1, n_features]`

**What it does NOT do:**
- No prompt engineering framework
- No conversation management
- No caching or batching
- No vendor lock-in — users can skip the trunk and pass raw tensors

**Dependencies:** `anthropic` and `openai` are optional extras (`pip install torch-eml[anthropic]`).

## Data Flow

### Training
```
Input tensor [batch, n_features]
       │
  Leaf Projection (nn.Linear)
       │
  [batch, 2^depth] leaf values
       │
  EML Tree (bottom-up, all eml nodes)
       │
  [batch, 1] output
       │
  Loss → backprop → update all weights
```

### Post-Training
```
Trained tree → prune(threshold) → snap(tolerance) → to_symbolic()
                                                          │
                                              SymbolicExpression
                                              ├── .string
                                              ├── .sympy
                                              ├── .latex
                                              └── .python
```

### LLM Pipeline
```
Unstructured text → LLMTrunk.extract() → tensor → EMLHead → score + equation
                                                                     │
                                                        LLM explains equation
                                                        (user's responsibility)
```

## Testing

| Test file | What it covers |
|---|---|
| `test_node.py` | EMLNode forward/backward, gradient flow, numerical stability near zero |
| `test_tree.py` | Tree construction at various depths, forward pass output shapes, deeper trees produce different outputs |
| `test_head.py` | End-to-end: random input → train on known function → snap → verify symbolic output |
| `test_symbolic.py` | Weight snapping accuracy, snap target matching, SymPy conversion, LaTeX output, Python codegen |
| `test_pruning.py` | Pruned trees produce same output within tolerance, dead branches removed, node count reduced |
| `test_trunk.py` | Mock LLM responses, feature extraction, malformed response handling, retry logic |

## Dependencies

**Required:**
- `torch >= 2.0`
- `sympy`

**Optional:**
- `anthropic` — for LLM trunk with Claude (`pip install torch-eml[anthropic]`)
- `openai` — for LLM trunk with OpenAI (`pip install torch-eml[openai]`)

## Examples

### 01: Symbolic Regression
Given `(x, y)` pairs sampled from `y = sin(x)`, train an EMLHead to discover the equation. After training + snapping, print the symbolic expression and verify it matches `sin(x)`.

### 02: Drop-in Head
Take a pretrained ResNet with its classification head removed. Attach an EMLHead. Fine-tune on a small dataset. Snap weights and extract a symbolic scoring function that explains the classification.

### 03: LLM Trunk
Use `LLMTrunk` with Claude to extract 8 features from a startup pitch description. Feed into a trained EMLHead. Output a score and the symbolic equation that produced it. Print both.

## README Structure

1. One-paragraph description + the EML formula
2. `pip install torch-eml`
3. 10-line quickstart (symbolic regression on `sin(x)`)
4. Three example sections with code + expected output
5. API reference (EMLHead, EMLTree, EMLNode, snap, prune, to_symbolic)
6. How it works (the math, with diagrams)
7. Citations (Odrzywołek 2026, Ipek 2026)
8. Contributing guide
9. License

## Non-Goals (v1)

- Docs site (README + docstrings + examples are sufficient)
- Learnable tree topology (fixed tree + pruning is sufficient)
- GPU-optimized EML kernels (standard PyTorch ops are fine for now)
- Training loop utilities (users bring their own training loop)
- Visualization dashboard (print/log is fine for v1)

## References

- A. Odrzywołek, "All elementary functions from a single binary operator," arXiv:2503.20022 (2026)
- E. Ipek, "Hardware-Efficient Neuro-Symbolic Networks with the Exp-Minus-Log Operator," arXiv:2604.13871 (2026)
