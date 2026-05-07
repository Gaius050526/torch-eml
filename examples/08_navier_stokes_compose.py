"""Navier-Stokes with ComposeHead: closed-form solutions via EML primitives.

Taylor-Green vortex (2D, decaying):
    u(x,y,t) =  sin(x) * cos(y) * exp(-2νt)
    v(x,y,t) = -cos(x) * sin(y) * exp(-2νt)

Strategy: try each separable term individually, pick the one with lowest loss,
then fine-tune it. This avoids the multi-term local minimum problem.
"""

import logging
import torch
import torch.nn as nn

from torch_eml.compose import ComposeHead, SeparableTerm

logging.basicConfig(level=logging.INFO)
torch.manual_seed(42)

nu = 0.1
PI2 = 2 * 3.14159265

print("=" * 60)
print("TAYLOR-GREEN VORTEX — closed-form via ComposeHead")
print("=" * 60)

# Training data
n = 3000
x_d = torch.rand(n, 1) * PI2
y_d = torch.rand(n, 1) * PI2
t_d = torch.rand(n, 1) * 1.0
data_in = torch.cat([x_d, y_d, t_d], dim=1)

u_target = torch.sin(x_d) * torch.cos(y_d) * torch.exp(-2 * nu * t_d)
v_target = -torch.cos(x_d) * torch.sin(y_d) * torch.exp(-2 * nu * t_d)


def find_best_single_term(data_in, target, n_inputs=3, n_steps=2000, label=""):
    """Try all separable triple terms, return the best-fitting one."""
    from itertools import combinations, product as cart_product

    sep_funcs = ["sin", "cos", "exp"]
    best_loss = float("inf")
    best_head = None

    print(f"\n  Searching for best single term ({label})...")
    results = []

    for funcs in cart_product(sep_funcs, repeat=3):
        dims = (0, 1, 2)
        # Create a ComposeHead with just this one term
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
        fname = "*".join(f"{f}({['x','y','t'][d]})" for f, d in zip(funcs, dims))
        results.append((final_loss, fname, head))

    results.sort()
    print(f"    Top 5 terms:")
    for loss_val, fname, _ in results[:5]:
        print(f"      {fname}: loss={loss_val:.8f}")

    return results[0][2]  # return best head


# --- Find best single term for u and v ---
head_u = find_best_single_term(data_in, u_target, label="u")
head_v = find_best_single_term(data_in, v_target, label="v")

# --- Fine-tune ---
print("\n  Fine-tuning best terms...")
params = list(head_u.parameters()) + list(head_v.parameters())
optimizer = torch.optim.Adam(params, lr=0.005)

for step in range(5000):
    u_pred = head_u(data_in)
    v_pred = head_v(data_in)
    loss = nn.functional.mse_loss(u_pred, u_target) + \
           nn.functional.mse_loss(v_pred, v_target)

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(params, 1.0)
    optimizer.step()

    if (step + 1) % 1000 == 0:
        print(f"    step {step+1}: loss={loss.item():.10f}")

# --- Output ---
print("\n  Raw parameters:")
for label, head in [("u", head_u), ("v", head_v)]:
    print(f"    {label} (bias={head.bias.item():.6f}):")
    for t in head.terms:
        print(f"      {t.func_name}: coeff={t.coeff.item():.6f}")
        if hasattr(t, 'factors'):
            for f in t.factors:
                print(f"        {f.func_name}[{f.input_dim}]: a={f.a.item():.6f}, b={f.b.item():.6f}")

expr_u = head_u.to_symbolic(input_names=["x", "y", "t"])
expr_v = head_v.to_symbolic(input_names=["x", "y", "t"])

print(f"\n  CLOSED-FORM SOLUTION:")
print(f"  u(x,y,t) = {expr_u.string}")
print(f"  v(x,y,t) = {expr_v.string}")
print(f"\n  Known:  u =  sin(x)*cos(y)*exp(-0.2t)")
print(f"  Known:  v = -cos(x)*sin(y)*exp(-0.2t)")

# --- Snap for clean version ---
import copy
from torch_eml.compose import SeparableTerm
head_u_snap = copy.deepcopy(head_u)
head_v_snap = copy.deepcopy(head_v)
# Normalize: absorb exp(b) into coeff, canonicalize sin/cos phases
for head in [head_u_snap, head_v_snap]:
    for t in head.terms:
        if isinstance(t, SeparableTerm):
            t.normalize()

print("\n  After normalization (before snap):")
for label, head in [("u", head_u_snap), ("v", head_v_snap)]:
    for t in head.terms:
        print(f"    {label}: {t.func_name}: coeff={t.coeff.item():.6f}")
        if hasattr(t, 'factors'):
            for f in t.factors:
                print(f"      {f.func_name}[{f.input_dim}]: a={f.a.item():.6f}, b={f.b.item():.6f}")

head_u_snap.snap_coefficients(tolerance=0.05)
head_v_snap.snap_coefficients(tolerance=0.05)

expr_u_snap = head_u_snap.to_symbolic(input_names=["x", "y", "t"])
expr_v_snap = head_v_snap.to_symbolic(input_names=["x", "y", "t"])
print(f"\n  SNAPPED CLEAN VERSION:")
print(f"  u(x,y,t) = {expr_u_snap.string}")
print(f"  v(x,y,t) = {expr_v_snap.string}")

# Show snapped parameters
print("\n  Snapped parameters:")
for label, head in [("u", head_u_snap), ("v", head_v_snap)]:
    print(f"    {label} (bias={head.bias.item():.6f}):")
    for t in head.terms:
        print(f"      {t.func_name}: coeff={t.coeff.item():.6f}")
        if hasattr(t, 'factors'):
            for f in t.factors:
                print(f"        {f.func_name}[{f.input_dim}]: a={f.a.item():.6f}, b={f.b.item():.6f}")

# --- Verify ---
with torch.no_grad():
    xt = torch.rand(5000, 1) * PI2
    yt = torch.rand(5000, 1) * PI2
    tt = torch.rand(5000, 1)
    test_in = torch.cat([xt, yt, tt], dim=1)

    u_true = torch.sin(xt) * torch.cos(yt) * torch.exp(-2 * nu * tt)
    v_true = -torch.cos(xt) * torch.sin(yt) * torch.exp(-2 * nu * tt)

    u_err = (head_u(test_in) - u_true).abs().mean().item()
    v_err = (head_v(test_in) - v_true).abs().mean().item()
    u_err_s = (head_u_snap(test_in) - u_true).abs().mean().item()
    v_err_s = (head_v_snap(test_in) - v_true).abs().mean().item()

    print(f"\n  Raw error  — u: {u_err:.8f}, v: {v_err:.8f}")
    print(f"  Snap error — u: {u_err_s:.8f}, v: {v_err_s:.8f}")

print(f"\n{'=' * 60}")
