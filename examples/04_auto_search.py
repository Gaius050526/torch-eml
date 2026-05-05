"""Auto-tune depth and structure search demo."""

import logging
import torch
from torch_eml.auto import auto_depth, search
from torch_eml.viz import save_html

logging.basicConfig(level=logging.INFO)

print("=== Auto-Depth Tuning ===\n")

# Generate data: y = sin(x)
torch.manual_seed(42)
X = torch.linspace(-3.14, 3.14, 500).unsqueeze(1)
y = torch.sin(X)

# Auto-tune: tries depths 2-5, picks best
result = auto_depth(n_inputs=1, X=X, y=y, depths=(2, 3, 4, 5), epochs=1000, lr=0.005)
print(f"\nBest depth: {result.depth}")
print(f"Validation loss: {result.val_loss:.6f}")

print("\n=== Full Structure Search ===\n")

# Search: auto-depth + prune + fine-tune + snap
result = search(
    n_inputs=1, X=X, y=y,
    max_depth=5, epochs=1000, finetune_epochs=300,
    lr=0.005, prune_threshold=0.05, snap_tolerance=0.1,
)

print(f"\nFinal equation: {result.expression.string}")
print(f"Final loss: {result.val_loss:.6f}")

# Save visualization
path = save_html(
    result.head,
    "eml_tree.html",
    title="Discovered Tree",
    equation=result.expression.string,
)
print(f"\nVisualization saved to: {path}")
print("Open it in a browser to see the tree structure.")
