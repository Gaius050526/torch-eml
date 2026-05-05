# torch-eml

**EML tree heads for interpretable neuro-symbolic models.**

`torch-eml` implements trainable binary trees where every node computes `eml(x, y) = exp(x) - ln(y)`. After training, the tree can be pruned and its weights snapped to clean values, producing a **closed-form symbolic equation** — human-readable, auditable, and formally verifiable.

```
eml(x, y) = eˣ - ln(y)
```

One function. Recursed. Every elementary function emerges.

## Install

```bash
pip install torch-eml
```

With LLM trunk support:
```bash
pip install "torch-eml[anthropic]"   # Claude
pip install "torch-eml[openai]"      # GPT
pip install "torch-eml[all]"         # everything
```

## Quickstart

```python
import torch
from torch_eml import EMLHead

# Train on y = sin(x)
X = torch.linspace(-3.14, 3.14, 500).unsqueeze(1)
y = torch.sin(X)

head = EMLHead(n_inputs=1, depth=3)
optimizer = torch.optim.Adam(head.parameters(), lr=0.005)

for _ in range(2000):
    loss = torch.nn.functional.mse_loss(head(X), y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Extract the equation
expr = head.snap(tolerance=0.1)
print(expr.string)   # symbolic equation
print(expr.latex)     # LaTeX for papers
print(expr.python)    # pure Python function
```

## Architecture

```
Input features [batch, n]
       │
  ┌────▼──────┐
  │ Projection │  nn.Linear(n, 2^depth)
  └────┬───────┘
       │
  ┌────▼──────┐
  │  EML Tree  │  Binary tree of identical eml(x,y) nodes
  │  (trained) │  Bottom-up evaluation → single scalar
  └────┬───────┘
       │
  [batch, 1] output
       │
  After training:
       │
  prune() → snap() → to_symbolic()
       │
  Closed-form equation
```

## API

### `EMLHead(n_inputs, depth=4)`

High-level module. Projects inputs to tree leaves, evaluates tree.

```python
head = EMLHead(n_inputs=8, depth=4)
output = head(torch.randn(32, 8))  # [32, 1]

# After training:
head.prune(threshold=0.05, calibration_data=X)
expr = head.snap(tolerance=0.1)
expr = head.to_symbolic(input_names=["market", "team", "growth"])
```

### `EMLTree(depth=4)`

Binary tree of EMLNodes. For researchers who want direct tree access.

```python
tree = EMLTree(depth=3)
leaves = torch.randn(32, 8)  # 2^3 = 8 leaves
output = tree(leaves)  # [32, 1]
```

### `EMLNode()`

Single node: `eml(x, y) = exp(w_l * x + b_l) - ln(|w_r * y + b_r| + ε)`

```python
node = EMLNode()
output = node(x, y)  # element-wise
```

### `SymbolicExpression`

Returned by `snap()` and `to_symbolic()`.

```python
expr.string   # "exp(0.5*x0) - ln(exp(x2) - ln(x5))"
expr.sympy    # SymPy expression (simplify, differentiate, etc.)
expr.latex    # LaTeX string
expr.python   # "import math\n\ndef f(x0, x1, ...):\n    return ..."
```

### `snap(tree, tolerance=0.05)`

Snap weights to clean values (0, 1, -1, π, e, √2, etc.).

### `prune(tree, threshold=0.01, calibration_data=X)`

Remove low-contribution branches. Returns `PruneReport`.

## Examples

See [`examples/`](examples/) for complete runnable scripts:

1. **Symbolic Regression** — Discover `sin(x)` from data
2. **Drop-in Head** — Replace MLP classification head with interpretable EML head
3. **LLM Trunk** — Claude extracts features → EML head produces a scoring equation

## How It Works

Odrzywołek (2026) proved that `eml(x, y) = exp(x) - ln(y)`, together with the constant 1, can express every standard elementary function as a binary tree of identical nodes — the continuous-mathematics analog of the NAND gate.

`torch-eml` makes this trainable:

1. **Train** a fixed-depth binary tree of `eml` nodes on your data
2. **Prune** branches that don't contribute meaningfully
3. **Snap** weights to clean values (3.14159 → π, 0.999 → 1)
4. **Extract** the resulting symbolic equation

The equation is not an approximation — it's a closed-form expression you can publish, audit, differentiate, and verify.

## Citations

```bibtex
@article{odrzywolel2026,
  title={All elementary functions from a single binary operator},
  author={Odrzywo{\l}ek, Andrzej},
  journal={arXiv preprint arXiv:2503.20022},
  year={2026}
}

@article{ipek2026,
  title={Hardware-Efficient Neuro-Symbolic Networks with the Exp-Minus-Log Operator},
  author={Ipek, Eymen},
  journal={arXiv preprint arXiv:2604.13871},
  year={2026}
}
```

## Contributing

Contributions welcome. Please open an issue first to discuss what you'd like to change.

```bash
git clone https://github.com/gaius050526/torch-eml.git
cd torch-eml
pip install -e ".[dev]"
pytest
```

## License

Apache 2.0
