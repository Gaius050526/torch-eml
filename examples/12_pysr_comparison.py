"""ComposeHead vs PySR: symbolic regression comparison.

Compares our ComposeHead brute-force single-term search against PySR
on the Taylor-Green vortex u-component:

    u(x, y, t) = sin(x) * cos(y) * exp(-0.2 * t)

Both methods receive the same training data and are evaluated on the
same held-out test set.
"""

import copy
import time
import logging

import numpy as np
import torch
import torch.nn as nn
from itertools import product as cart_product

from torch_eml.compose import ComposeHead, SeparableTerm

logging.basicConfig(level=logging.INFO)
torch.manual_seed(42)

nu = 0.1
PI2 = 2 * 3.14159265

# ------------------------------------------------------------------ #
#  Shared data                                                        #
# ------------------------------------------------------------------ #
print("=" * 70)
print("ComposeHead vs PySR — Taylor-Green vortex u-component")
print("  u(x,y,t) = sin(x)*cos(y)*exp(-0.2*t)")
print("=" * 70)

# Training data (3000 points)
n_train = 3000
x_tr = torch.rand(n_train, 1) * PI2
y_tr = torch.rand(n_train, 1) * PI2
t_tr = torch.rand(n_train, 1) * 1.0
X_train = torch.cat([x_tr, y_tr, t_tr], dim=1)
y_train = torch.sin(x_tr) * torch.cos(y_tr) * torch.exp(-2 * nu * t_tr)

# Test data (5000 points)
n_test = 5000
x_te = torch.rand(n_test, 1) * PI2
y_te = torch.rand(n_test, 1) * PI2
t_te = torch.rand(n_test, 1) * 1.0
X_test = torch.cat([x_te, y_te, t_te], dim=1)
y_test = torch.sin(x_te) * torch.cos(y_te) * torch.exp(-2 * nu * t_te)

# Numpy copies for PySR
X_train_np = X_train.numpy()
y_train_np = y_train.squeeze(-1).numpy()
X_test_np = X_test.numpy()
y_test_np = y_test.squeeze(-1).numpy()

# ------------------------------------------------------------------ #
#  Method 1: ComposeHead (brute-force single-term search)             #
# ------------------------------------------------------------------ #
print("\n" + "-" * 70)
print("METHOD 1: ComposeHead (brute-force single-term search)")
print("-" * 70)


def find_best_single_term(data_in, target, n_inputs=3, n_steps=2000, label=""):
    """Try all separable triple terms, return the best-fitting one."""
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
        fname = "*".join(
            f"{f}({['x','y','t'][d]})" for f, d in zip(funcs, dims)
        )
        results.append((final_loss, fname, head))

    results.sort()
    print(f"    Top 5 terms:")
    for loss_val, fname, _ in results[:5]:
        print(f"      {fname}: loss={loss_val:.8f}")

    return results[0][2]  # return best head


compose_start = time.time()

head_u = find_best_single_term(X_train, y_train, label="u")

# --- Fine-tune ---
print("\n  Fine-tuning best term...")
optimizer = torch.optim.Adam(head_u.parameters(), lr=0.005)

for step in range(5000):
    u_pred = head_u(X_train)
    loss = nn.functional.mse_loss(u_pred, y_train)
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(head_u.parameters(), 1.0)
    optimizer.step()

    if (step + 1) % 1000 == 0:
        print(f"    step {step+1}: loss={loss.item():.10f}")

# --- Normalize and snap ---
head_u_snap = copy.deepcopy(head_u)
for t in head_u_snap.terms:
    if isinstance(t, SeparableTerm):
        t.normalize()
head_u_snap.snap_coefficients(tolerance=0.05)

compose_time = time.time() - compose_start

# --- Evaluate on test set ---
with torch.no_grad():
    compose_pred = head_u_snap(X_test)
    compose_error = (compose_pred - y_test).abs().mean().item()
    compose_pred_raw = head_u(X_test)
    compose_error_raw = (compose_pred_raw - y_test).abs().mean().item()

expr_u = head_u_snap.to_symbolic(input_names=["x", "y", "t"])
compose_expr = expr_u.string

print(f"\n  ComposeHead expression (snapped): {compose_expr}")
print(f"  ComposeHead test MAE (raw):       {compose_error_raw:.8f}")
print(f"  ComposeHead test MAE (snapped):   {compose_error:.8f}")
print(f"  ComposeHead training time:        {compose_time:.1f}s")

# ------------------------------------------------------------------ #
#  Method 2: PySR                                                     #
# ------------------------------------------------------------------ #
print("\n" + "-" * 70)
print("METHOD 2: PySR")
print("-" * 70)

pysr_expr = None
pysr_error = None
pysr_time = None

try:
    from pysr import PySRRegressor
    import time as _time  # already imported, but be explicit

    model = PySRRegressor(
        niterations=100,
        binary_operators=["+", "*", "-", "/"],
        unary_operators=["sin", "cos", "exp"],
        populations=30,
        maxsize=20,
        timeout_in_seconds=300,
        progress=True,
    )

    start = time.time()
    model.fit(X_train_np, y_train_np)
    pysr_time = time.time() - start

    pysr_expr = str(model.sympy())
    print(f"\n  PySR best expression: {pysr_expr}")
    print(f"  PySR training time:   {pysr_time:.1f}s")

    # Evaluate on test set
    pysr_pred = model.predict(X_test_np)
    pysr_error = np.mean(np.abs(pysr_pred - y_test_np))
    print(f"  PySR test MAE:        {pysr_error:.8f}")

except ImportError:
    print("\n  PySR is not installed. To include PySR in the comparison, run:")
    print("      pip install pysr")
    print("  Skipping PySR benchmark.\n")

# ------------------------------------------------------------------ #
#  Comparison table                                                   #
# ------------------------------------------------------------------ #
print("\n" + "=" * 70)
print("COMPARISON TABLE")
print("=" * 70)

header = f"{'Method':<14} | {'Expression':<40} | {'Test MAE':<12} | {'Time (s)':<10} | {'Interpretable?'}"
sep = "-" * len(header)
print(header)
print(sep)

# ComposeHead row
compose_expr_short = compose_expr if len(compose_expr) <= 40 else compose_expr[:37] + "..."
print(
    f"{'ComposeHead':<14} | {compose_expr_short:<40} | "
    f"{compose_error:<12.8f} | {compose_time:<10.1f} | {'Yes'}"
)

# PySR row
if pysr_expr is not None:
    pysr_expr_short = pysr_expr if len(pysr_expr) <= 40 else pysr_expr[:37] + "..."
    print(
        f"{'PySR':<14} | {pysr_expr_short:<40} | "
        f"{pysr_error:<12.8f} | {pysr_time:<10.1f} | {'Yes'}"
    )
else:
    print(
        f"{'PySR':<14} | {'(not installed)':<40} | "
        f"{'N/A':<12} | {'N/A':<10} | {'--'}"
    )

print(sep)
print(f"\nGround truth: u(x,y,t) = sin(x)*cos(y)*exp(-0.2*t)")
print("=" * 70)
