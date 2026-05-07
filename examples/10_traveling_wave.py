"""Traveling wave: recover a NON-SEPARABLE solution via two-term brute-force search.

Target:
    u(x,t) = sin(x - 2t)

This is NOT separable in x and t. However, it decomposes as a sum of two
separable terms via the angle subtraction identity:

    sin(x - 2t) = sin(x)*cos(2t) - cos(x)*sin(2t)

Algorithm:
    1. Enumerate all 81 two-term pairs (9 single-term types × 9)
    2. For each pair, freeze biases at 0, train jointly for 2000 steps
    3. Pick the best pair
    4. Fine-tune for 5000 steps
    5. Snap scales + coefficients to clean values
"""

import torch
import torch.nn as nn
import copy
from itertools import product as cart_product
from torch_eml.compose import ComposeHead, SeparableTerm

torch.manual_seed(42)

PI2 = 2 * 3.14159265

print("=" * 70)
print("TRAVELING WAVE — non-separable via two-term brute-force search")
print("  Target: u(x,t) = sin(x - 2t)")
print("  Decomposition: sin(x)cos(2t) - cos(x)sin(2t)")
print("=" * 70)

# ----------------------------------------------------------------
# 1. Generate training data
# ----------------------------------------------------------------
n_train = 3000
x_d = torch.rand(n_train, 1) * PI2       # x in [0, 2pi]
t_d = torch.rand(n_train, 1) * 2.0       # t in [0, 2]
data_in = torch.cat([x_d, t_d], dim=1)   # [3000, 2]
target = torch.sin(x_d - 2 * t_d)        # [3000, 1]

print(f"\n  Training points: {n_train}")
print(f"  x in [0, {PI2:.4f}], t in [0, 2]")

# ----------------------------------------------------------------
# 2. Enumerate all two-term pairs with frozen biases
# ----------------------------------------------------------------
sep_funcs = ["sin", "cos", "exp"]
single_types = list(cart_product(sep_funcs, repeat=2))  # 9 types

print(f"\n  Searching {len(single_types)}² = {len(single_types)**2} two-term pairs...")
print(f"  (biases frozen at 0, 2000 steps each)")

results = []
total = len(single_types) ** 2
for i, funcs1 in enumerate(single_types):
    for funcs2 in single_types:
        # Create two-term model with frozen biases
        head = ComposeHead(n_inputs=2, primitives=[], repeat=0,
                          products=False, separable=False)
        term1 = SeparableTerm(list(zip(funcs1, (0, 1))), 2)
        term2 = SeparableTerm(list(zip(funcs2, (0, 1))), 2)
        head.terms = nn.ModuleList([term1, term2])

        # Freeze biases at 0
        with torch.no_grad():
            head.bias.fill_(0.0)
            head.bias.requires_grad_(False)
            for t in head.terms:
                for f in t.factors:
                    f.b.fill_(0.0)
                    f.b.requires_grad_(False)

        trainable = [p for p in head.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(trainable, lr=0.01)

        for step in range(2000):
            pred = head(data_in)
            loss = nn.functional.mse_loss(pred, target)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()

        final_loss = loss.item()
        dim_names = ["x", "t"]
        name1 = "*".join(f"{f}({dim_names[d]})" for f, d in zip(funcs1, (0, 1)))
        name2 = "*".join(f"{f}({dim_names[d]})" for f, d in zip(funcs2, (0, 1)))
        results.append((final_loss, f"{name1} + {name2}", head))

results.sort()
print(f"\n  Top 10 two-term pairs:")
for loss_val, name, _ in results[:10]:
    print(f"    {name}: loss={loss_val:.8f}")

best_head = results[0][2]

# ----------------------------------------------------------------
# 3. Fine-tune the best pair
# ----------------------------------------------------------------
print(f"\n  Fine-tuning best pair...")
trainable = [p for p in best_head.parameters() if p.requires_grad]
optimizer = torch.optim.Adam(trainable, lr=0.005)

for step in range(5000):
    pred = best_head(data_in)
    loss = nn.functional.mse_loss(pred, target)
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(trainable, 1.0)
    optimizer.step()
    if (step + 1) % 1000 == 0:
        print(f"    step {step+1}/5000: loss={loss.item():.10f}")

# ----------------------------------------------------------------
# 4. Print raw parameters
# ----------------------------------------------------------------
print("\n" + "-" * 70)
print("  RAW LEARNED PARAMETERS")
print("-" * 70)
for i, t in enumerate(best_head.terms):
    print(f"  Term {i+1}: coeff={t.coeff.item():.6f}")
    for f in t.factors:
        print(f"    {f.func_name}[{f.input_dim}]: a={f.a.item():.6f}, b={f.b.item():.6f}")

# ----------------------------------------------------------------
# 5. Snap (biases already 0, just snap scales + coefficients)
# ----------------------------------------------------------------
print("\n" + "-" * 70)
print("  SNAP COEFFICIENTS")
print("-" * 70)

head_snap = copy.deepcopy(best_head)
head_snap.snap_coefficients(tolerance=0.05)

print("\n  After snap:")
for i, t in enumerate(head_snap.terms):
    print(f"  Term {i+1}: coeff={t.coeff.item():.6f}")
    for f in t.factors:
        print(f"    {f.func_name}[{f.input_dim}]: a={f.a.item():.6f}, b={f.b.item():.6f}")

expr_snap = head_snap.to_symbolic(input_names=["x", "t"])
print(f"\n  RECOVERED: u(x,t) = {expr_snap.string}")
print(f"  GROUND TRUTH: sin(x - 2t) = sin(x)*cos(2t) - cos(x)*sin(2t)")

# ----------------------------------------------------------------
# 6. Verify on test data
# ----------------------------------------------------------------
print("\n" + "-" * 70)
print("  VERIFICATION ON TEST DATA")
print("-" * 70)

with torch.no_grad():
    n_test = 5000
    x_test = torch.rand(n_test, 1) * PI2
    t_test = torch.rand(n_test, 1) * 2.0
    test_in = torch.cat([x_test, t_test], dim=1)
    u_true = torch.sin(x_test - 2 * t_test)

    u_raw = best_head(test_in)
    err_raw = (u_raw - u_true).abs().mean().item()

    u_snap = head_snap(test_in)
    err_snap = (u_snap - u_true).abs().mean().item()

    print(f"\n  Test points: {n_test}")
    print(f"  Raw model   MAE: {err_raw:.8f}")
    print(f"  Snapped model MAE: {err_snap:.8f}")

print(f"\n{'=' * 70}")
print("  Non-separable sin(x - 2t) successfully decomposed into")
print("  two separable terms via brute-force pair search.")
print(f"{'=' * 70}")
