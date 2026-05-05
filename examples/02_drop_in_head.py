"""Replace an MLP head with an EML head for interpretable classification."""

import logging
import torch
import torch.nn as nn
from torch_eml import EMLHead

logging.basicConfig(level=logging.INFO)

# Simple MLP trunk (pretend this is a pretrained feature extractor)
class MLPTrunk(nn.Module):
    def __init__(self, in_features: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)

# Generate synthetic data: 2-class classification
torch.manual_seed(42)
X = torch.randn(500, 10)
y = (X[:, 0] + X[:, 1] - X[:, 2] > 0).float().unsqueeze(1)  # [500, 1]

# Build model: MLP trunk + EML head
trunk = MLPTrunk(in_features=10, hidden=16)
head = EMLHead(n_inputs=16, depth=3)

model = nn.Sequential(trunk, head)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

print("Training MLP trunk + EML head...")
for step in range(500):
    pred = torch.sigmoid(model(X))
    loss = nn.functional.binary_cross_entropy(pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (step + 1) % 100 == 0:
        acc = ((pred > 0.5).float() == y).float().mean()
        print(f"  Step {step+1}: loss={loss.item():.4f}, acc={acc.item():.3f}")

# Extract symbolic equation from the EML head
print("\nSnapping EML head weights...")
expr = head.snap(tolerance=0.1)

print(f"\nScoring equation (from EML head):")
print(f"  {expr.string}")
print(f"\nNote: inputs to this equation are the trunk's 16 hidden features,")
print(f"not the original 10 raw features.")
