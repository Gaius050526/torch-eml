"""Rediscover physics formulas from noisy experimental data.

Demonstrates torch-eml recovering known laws from synthetic measurements:
  1. Kepler's Third Law: T = a^(3/2)
  2. Inverse Square Law: F = k / r²

No physics knowledge is encoded — the EML tree discovers the relationships from data alone.
"""

import logging
import torch
from torch_eml import search, save_html

logging.basicConfig(level=logging.INFO)
torch.manual_seed(42)


# ============================================================
# 1. Kepler's Third Law: T = a^(3/2)
# ============================================================
print("=" * 60)
print("1. KEPLER'S THIRD LAW: T = a^(3/2)")
print("=" * 60)

# Normalized data: small range, near unit scale
a = torch.linspace(0.5, 3.0, 400).unsqueeze(1)
T = a ** 1.5 + 0.01 * torch.randn(400, 1)  # small noise

result = search(
    n_inputs=1, X=a, y=T,
    max_depth=4, epochs=2000, finetune_epochs=500,
    lr=0.005, prune_threshold=0.1, snap_tolerance=0.15,
)

print(f"\n  Discovered: T = {result.expression.string}")
print(f"  Loss: {result.val_loss:.6f}")

save_html(result.head, "kepler_tree.html",
          title="Kepler's Third Law", equation=result.expression.string)
print(f"  Tree saved to kepler_tree.html\n")


# ============================================================
# 2. Inverse Square Law: F = 1 / r²
# ============================================================
print("=" * 60)
print("2. INVERSE SQUARE LAW: F = 1 / r²")
print("=" * 60)

r = torch.linspace(0.5, 3.0, 400).unsqueeze(1)
F = 1.0 / r ** 2 + 0.01 * torch.randn(400, 1)

result = search(
    n_inputs=1, X=r, y=F,
    max_depth=4, epochs=2000, finetune_epochs=500,
    lr=0.005, prune_threshold=0.1, snap_tolerance=0.15,
)

print(f"\n  Discovered: F = {result.expression.string}")
print(f"  Loss: {result.val_loss:.6f}")

save_html(result.head, "force_tree.html",
          title="Inverse Square Law", equation=result.expression.string)
print(f"  Tree saved to force_tree.html")

print("\n" + "=" * 60)
print("Open the HTML files in a browser to see the tree structures.")
print("=" * 60)
