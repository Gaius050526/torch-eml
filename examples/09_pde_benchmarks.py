"""PDE Benchmarks: exact recovery of separable PDE solutions via ComposeHead.

Tests two classical PDEs with known separable solutions:

1. 1D Heat Equation: u(x,t) = sin(pi*x) * exp(-pi^2 * t)
   Domain: x in [0,1], t in [0,0.5]
   Basis: {sin, cos, exp}, 3^2 = 9 candidate pairs

2. 1D Wave Equation: u(x,t) = sin(pi*x) * cos(pi*t)
   Domain: x in [0,1], t in [0,2]
   Basis: {sin, cos, exp}, 3^2 = 9 candidate pairs

Strategy: brute-force search over all separable pair candidates, pick the
one with lowest loss, fine-tune, normalize, snap, and compare with ground truth.
"""

import torch
import torch.nn as nn
import copy
from itertools import product as cart_product
from torch_eml.compose import ComposeHead, SeparableTerm

torch.manual_seed(42)

PI = 3.14159265358979


def find_best_pair(data_in, target, n_inputs, n_steps=2000, label=""):
    """Try all 9 separable pair candidates and return the best one."""
    sep_funcs = ["sin", "cos", "exp"]
    results = []

    print(f"\n  Searching over 9 separable pair candidates ({label})...")

    for funcs in cart_product(sep_funcs, repeat=2):
        dims = (0, 1)
        # Create a ComposeHead with just this one separable term
        head = ComposeHead(n_inputs=n_inputs, primitives=[], repeat=0,
                           products=False, separable=False)
        term = SeparableTerm(list(zip(funcs, dims)), n_inputs)
        head.terms = nn.ModuleList([term])

        optimizer = torch.optim.Adam(head.parameters(), lr=0.01)
        for step in range(n_steps):
            pred = head(data_in)
            loss = nn.functional.mse_loss(pred, target)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()

        final_loss = loss.item()
        dim_names = ["x", "t"]
        fname = " * ".join(f"{f}({dim_names[d]})" for f, d in zip(funcs, dims))
        results.append((final_loss, fname, head))

    results.sort()
    print(f"    Top 5 candidates:")
    for loss_val, fname, _ in results[:5]:
        print(f"      {fname}: loss = {loss_val:.8f}")

    return results[0][2]  # return best head


# ============================================================
# PDE 1: 1D Heat Equation
# u(x,t) = sin(pi*x) * exp(-pi^2 * t)
# Domain: x in [0,1], t in [0,0.5]
# ============================================================
print("=" * 60)
print("PDE 1: 1D HEAT EQUATION")
print("  u(x,t) = sin(pi*x) * exp(-pi^2 * t)")
print("  Domain: x in [0,1], t in [0,0.5]")
print("=" * 60)

# Generate 2000 training points
n_train = 2000
x_heat = torch.rand(n_train, 1)          # x in [0, 1]
t_heat = torch.rand(n_train, 1) * 0.5    # t in [0, 0.5]
data_heat = torch.cat([x_heat, t_heat], dim=1)
u_heat_target = torch.sin(PI * x_heat) * torch.exp(-PI**2 * t_heat)

# Search over all 9 pair candidates
head_heat = find_best_pair(data_heat, u_heat_target, n_inputs=2, label="heat eq")

# Fine-tune the best candidate for 5000 steps
print("\n  Fine-tuning best candidate...")
optimizer = torch.optim.Adam(head_heat.parameters(), lr=0.005)
for step in range(5000):
    pred = head_heat(data_heat)
    loss = nn.functional.mse_loss(pred, u_heat_target)
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(head_heat.parameters(), 1.0)
    optimizer.step()
    if (step + 1) % 1000 == 0:
        print(f"    step {step+1}: loss = {loss.item():.10f}")

# Raw symbolic expression
expr_heat_raw = head_heat.to_symbolic(input_names=["x", "t"])
print(f"\n  Raw recovered expression:")
print(f"    u(x,t) = {expr_heat_raw.string}")

# Normalize, snap, and get clean expression
head_heat_snap = copy.deepcopy(head_heat)
for t in head_heat_snap.terms:
    if isinstance(t, SeparableTerm):
        t.normalize()
head_heat_snap.snap_coefficients(tolerance=0.05)

expr_heat_snap = head_heat_snap.to_symbolic(input_names=["x", "t"])
print(f"\n  Snapped recovered expression:")
print(f"    u(x,t) = {expr_heat_snap.string}")
print(f"\n  Ground truth:")
print(f"    u(x,t) = sin(pi*x) * exp(-pi^2 * t)")

# Snapped parameters
print("\n  Snapped parameters:")
print(f"    bias = {head_heat_snap.bias.item():.6f}")
for t in head_heat_snap.terms:
    print(f"    coeff = {t.coeff.item():.6f}")
    if hasattr(t, 'factors'):
        for f in t.factors:
            print(f"      {f.func_name}[dim={f.input_dim}]: a = {f.a.item():.6f}, b = {f.b.item():.6f}")

# Test on 5000 held-out points
with torch.no_grad():
    x_test = torch.rand(5000, 1)
    t_test = torch.rand(5000, 1) * 0.5
    test_heat = torch.cat([x_test, t_test], dim=1)
    u_heat_true = torch.sin(PI * x_test) * torch.exp(-PI**2 * t_test)

    heat_err_raw = (head_heat(test_heat) - u_heat_true).abs().mean().item()
    heat_err_snap = (head_heat_snap(test_heat) - u_heat_true).abs().mean().item()

    print(f"\n  Test error (5000 held-out points):")
    print(f"    Raw  MAE: {heat_err_raw:.8f}")
    print(f"    Snap MAE: {heat_err_snap:.8f}")


# ============================================================
# PDE 2: 1D Wave Equation
# u(x,t) = sin(pi*x) * cos(pi*t)
# Domain: x in [0,1], t in [0,2]
# ============================================================
print(f"\n{'=' * 60}")
print("PDE 2: 1D WAVE EQUATION")
print("  u(x,t) = sin(pi*x) * cos(pi*t)")
print("  Domain: x in [0,1], t in [0,2]")
print("=" * 60)

# Generate 2000 training points
x_wave = torch.rand(n_train, 1)          # x in [0, 1]
t_wave = torch.rand(n_train, 1) * 2.0    # t in [0, 2]
data_wave = torch.cat([x_wave, t_wave], dim=1)
u_wave_target = torch.sin(PI * x_wave) * torch.cos(PI * t_wave)

# Search over all 9 pair candidates
head_wave = find_best_pair(data_wave, u_wave_target, n_inputs=2, label="wave eq")

# Fine-tune the best candidate for 5000 steps
print("\n  Fine-tuning best candidate...")
optimizer = torch.optim.Adam(head_wave.parameters(), lr=0.005)
for step in range(5000):
    pred = head_wave(data_wave)
    loss = nn.functional.mse_loss(pred, u_wave_target)
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(head_wave.parameters(), 1.0)
    optimizer.step()
    if (step + 1) % 1000 == 0:
        print(f"    step {step+1}: loss = {loss.item():.10f}")

# Raw symbolic expression
expr_wave_raw = head_wave.to_symbolic(input_names=["x", "t"])
print(f"\n  Raw recovered expression:")
print(f"    u(x,t) = {expr_wave_raw.string}")

# Normalize, snap, and get clean expression
head_wave_snap = copy.deepcopy(head_wave)
for t in head_wave_snap.terms:
    if isinstance(t, SeparableTerm):
        t.normalize()
head_wave_snap.snap_coefficients(tolerance=0.05)

expr_wave_snap = head_wave_snap.to_symbolic(input_names=["x", "t"])
print(f"\n  Snapped recovered expression:")
print(f"    u(x,t) = {expr_wave_snap.string}")
print(f"\n  Ground truth:")
print(f"    u(x,t) = sin(pi*x) * cos(pi*t)")

# Snapped parameters
print("\n  Snapped parameters:")
print(f"    bias = {head_wave_snap.bias.item():.6f}")
for t in head_wave_snap.terms:
    print(f"    coeff = {t.coeff.item():.6f}")
    if hasattr(t, 'factors'):
        for f in t.factors:
            print(f"      {f.func_name}[dim={f.input_dim}]: a = {f.a.item():.6f}, b = {f.b.item():.6f}")

# Test on 5000 held-out points
with torch.no_grad():
    x_test = torch.rand(5000, 1)
    t_test = torch.rand(5000, 1) * 2.0
    test_wave = torch.cat([x_test, t_test], dim=1)
    u_wave_true = torch.sin(PI * x_test) * torch.cos(PI * t_test)

    wave_err_raw = (head_wave(test_wave) - u_wave_true).abs().mean().item()
    wave_err_snap = (head_wave_snap(test_wave) - u_wave_true).abs().mean().item()

    print(f"\n  Test error (5000 held-out points):")
    print(f"    Raw  MAE: {wave_err_raw:.8f}")
    print(f"    Snap MAE: {wave_err_snap:.8f}")


# ============================================================
# Summary
# ============================================================
print(f"\n{'=' * 60}")
print("SUMMARY")
print("=" * 60)
print(f"  Heat equation:")
print(f"    Ground truth:  sin(pi*x) * exp(-pi^2 * t)")
print(f"    Recovered:     {expr_heat_snap.string}")
print(f"    Raw  MAE:      {heat_err_raw:.8f}")
print(f"    Snap MAE:      {heat_err_snap:.8f}")
print(f"\n  Wave equation:")
print(f"    Ground truth:  sin(pi*x) * cos(pi*t)")
print(f"    Recovered:     {expr_wave_snap.string}")
print(f"    Raw  MAE:      {wave_err_raw:.8f}")
print(f"    Snap MAE:      {wave_err_snap:.8f}")
print(f"{'=' * 60}")
