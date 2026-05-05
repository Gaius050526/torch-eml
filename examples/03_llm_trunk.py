"""LLM trunk + EML head for startup scoring (uses mock data, no API key needed)."""

import torch
from torch_eml import EMLHead

# In a real scenario, you'd use:
#   from torch_eml.trunk import LLMTrunk
#   trunk = LLMTrunk(provider="anthropic", model="claude-opus-4-6", features=[...])
#   features = trunk.extract("Here is our pitch deck...")
#
# For this example, we simulate the trunk output directly.

print("=== LLM Trunk + EML Head Demo ===\n")
print("Simulating LLM feature extraction (no API key needed)...\n")

# Simulated features an LLM might extract from pitch decks
feature_names = [
    "market_size_log",    # log10 of TAM
    "team_years",         # combined team experience
    "revenue_growth",     # QoQ growth rate
    "burn_multiple",      # burn / net new ARR
    "competitor_count",   # number of direct competitors
    "defensibility",      # moat score 0-1
    "traction_score",     # normalized traction metric
    "timing_score",       # market timing score 0-1
]

# Generate synthetic training data (as if extracted by LLM from 200 pitch decks)
torch.manual_seed(42)
n_companies = 200
X = torch.randn(n_companies, len(feature_names))

# Synthetic "investor score" based on a known formula
y = (
    0.3 * X[:, 0]           # market size matters
    + 0.2 * X[:, 2]         # growth matters
    - 0.15 * X[:, 3]        # high burn is bad
    + 0.1 * X[:, 5]         # defensibility helps
    + 0.05 * X[:, 7]        # timing matters a little
).unsqueeze(1)

# Train EML head to discover the scoring formula
head = EMLHead(n_inputs=len(feature_names), depth=3)
optimizer = torch.optim.Adam(head.parameters(), lr=0.01)

print("Training EML head on labeled deal data...")
for step in range(1000):
    pred = head(X)
    loss = torch.nn.functional.mse_loss(pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (step + 1) % 250 == 0:
        print(f"  Step {step+1}: loss={loss.item():.6f}")

# Prune and snap
print("\nPruning...")
report = head.prune(threshold=0.05, calibration_data=X)
print(f"  Nodes: {report.nodes_before} -> {report.nodes_after}")

print("\nSnapping to clean weights...")
expr = head.snap(tolerance=0.1, validation_data=(X, y))

print(f"\nDiscovered scoring equation:")
print(f"  {expr.string}")
print(f"\nLaTeX (for papers):")
print(f"  {expr.latex}")

# Score a new company
print("\n--- Scoring a new company ---")
new_company = torch.tensor([[
    4.5,   # market_size_log: $30B TAM
    15.0,  # team_years: 15 years combined
    0.9,   # revenue_growth: 90% QoQ
    1.2,   # burn_multiple: 1.2x
    5.0,   # competitor_count: 5
    0.7,   # defensibility: 0.7
    0.8,   # traction_score: 0.8
    0.6,   # timing_score: 0.6
]])

with torch.no_grad():
    score = head(new_company).item()
print(f"  Score: {score:.4f}")
print(f"  Equation used: {expr.string}")
