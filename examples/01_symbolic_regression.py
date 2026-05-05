"""Discover y = sin(x) from data using an EML head."""

import torch
from torch_eml import EMLHead

# Generate training data
torch.manual_seed(42)
X = torch.linspace(-3.14, 3.14, 500).unsqueeze(1)  # [500, 1]
y = torch.sin(X)  # [500, 1]

# Create and train EML head
head = EMLHead(n_inputs=1, depth=3)
optimizer = torch.optim.Adam(head.parameters(), lr=0.005)

print("Training...")
for step in range(2000):
    pred = head(X)
    loss = torch.nn.functional.mse_loss(pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (step + 1) % 500 == 0:
        print(f"  Step {step+1}: loss={loss.item():.6f}")

# Prune and snap
print("\nPruning...")
report = head.prune(threshold=0.05, calibration_data=X)
print(f"  Nodes: {report.nodes_before} -> {report.nodes_after}")

print("\nSnapping weights...")
expr = head.snap(tolerance=0.1, validation_data=(X, y))

print(f"\nDiscovered equation:")
print(f"  String: {expr.string}")
print(f"  LaTeX:  {expr.latex}")
print(f"\nPython function:")
print(expr.python)
