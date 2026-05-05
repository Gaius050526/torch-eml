"""Head-to-head: MLP vs EML on the same regression task.

Both models fit y = exp(sin(x)). One gives you an equation, the other doesn't.
"""

import logging
import torch
import torch.nn as nn

from torch_eml import EMLHead

logging.basicConfig(level=logging.INFO)
torch.manual_seed(42)

# ============================================================
# Dataset: y = exp(sin(x)) — a natural composition for EML
# ============================================================
X = torch.linspace(-3.14, 3.14, 500).unsqueeze(1)
y = torch.exp(torch.sin(X))

X_train, X_test = X[:400], X[400:]
y_train, y_test = y[:400], y[400:]

# ============================================================
# Model 1: MLP (3 layers, 64 hidden)
# ============================================================
print("=" * 60)
print("MLP (1 -> 64 -> 64 -> 1)")
print("=" * 60)

mlp = nn.Sequential(
    nn.Linear(1, 64), nn.ReLU(),
    nn.Linear(64, 64), nn.ReLU(),
    nn.Linear(64, 1),
)
opt = torch.optim.Adam(mlp.parameters(), lr=0.01)

for step in range(2000):
    loss = nn.functional.mse_loss(mlp(X_train), y_train)
    opt.zero_grad()
    loss.backward()
    opt.step()
    if (step + 1) % 500 == 0:
        print(f"  step {step+1}: train_loss={loss.item():.6f}")

with torch.no_grad():
    mlp_test_loss = nn.functional.mse_loss(mlp(X_test), y_test).item()

print(f"\n  Test loss: {mlp_test_loss:.6f}")
print(f"  Parameters: {sum(p.numel() for p in mlp.parameters())}")
print(f"  Equation: ???  (black box)\n")

# ============================================================
# Model 2: EML Head
# ============================================================
print("=" * 60)
print("EML Head (depth=4)")
print("=" * 60)

head = EMLHead(n_inputs=1, depth=4)
opt = torch.optim.Adam(head.parameters(), lr=0.005)

for step in range(3000):
    loss = nn.functional.mse_loss(head(X_train), y_train)
    opt.zero_grad()
    loss.backward()
    opt.step()
    if (step + 1) % 500 == 0:
        print(f"  step {step+1}: train_loss={loss.item():.6f}")

with torch.no_grad():
    eml_test_loss = nn.functional.mse_loss(head(X_test), y_test).item()

print(f"\n  Test loss (pre-snap): {eml_test_loss:.6f}")

# Prune and extract
head.prune(threshold=0.05, calibration_data=X_train)
expr = head.snap(tolerance=0.1, validation_data=(X_train, y_train))

with torch.no_grad():
    eml_snapped_loss = nn.functional.mse_loss(head(X_test), y_test).item()

print(f"  Test loss (post-snap): {eml_snapped_loss:.6f}")
print(f"  Parameters: {sum(p.numel() for p in head.parameters())}")
print(f"\n  Equation: {expr.string}")
print(f"  LaTeX:    {expr.latex}")

# ============================================================
# Summary
# ============================================================
print(f"\n{'=' * 60}")
print("COMPARISON")
print("=" * 60)
print(f"  {'':25} {'MLP':>15} {'EML':>15}")
print(f"  {'-'*25} {'-'*15} {'-'*15}")
print(f"  {'Test MSE':25} {mlp_test_loss:>15.6f} {eml_snapped_loss:>15.6f}")
print(f"  {'Parameters':25} {sum(p.numel() for p in mlp.parameters()):>15} {sum(p.numel() for p in head.parameters()):>15}")
print(f"  {'Interpretable':25} {'No':>15} {'Yes':>15}")
print(f"  {'Publishable equation':25} {'No':>15} {'Yes':>15}")
