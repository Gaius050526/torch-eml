"""Noise robustness and graceful failure modes.

Part 1: Noise robustness — recover sin(x)*cos(y)*exp(-0.2*t) under increasing noise.
Part 2: Wrong basis — attempt recovery with only {exp, ln} (no trig).
Part 3: Non-separable target — sin(x - 2*t) cannot be expressed as a single separable term.
"""

import copy
import logging

import torch
import torch.nn as nn

from torch_eml.compose import ComposeHead, SeparableTerm

logging.basicConfig(level=logging.WARNING)
torch.manual_seed(42)

PI2 = 2 * 3.14159265


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_taylor_green_data(n, seed=None):
    """Generate training data for u(x,y,t) = sin(x)*cos(y)*exp(-0.2*t)."""
    if seed is not None:
        torch.manual_seed(seed)
    x = torch.rand(n, 1) * PI2
    y = torch.rand(n, 1) * PI2
    t = torch.rand(n, 1) * 1.0
    data_in = torch.cat([x, y, t], dim=1)
    u_true = torch.sin(x) * torch.cos(y) * torch.exp(-0.2 * t)
    return data_in, u_true


def brute_force_search(data_in, target, basis, n_inputs, n_steps=2000, lr=0.01):
    """Try all separable triple-product candidates, return sorted results.

    Each candidate: f(x_0) * g(x_1) * h(x_2) for f,g,h in basis, dims 0,1,2.
    Returns list of (loss, label, head) sorted by loss.
    """
    from itertools import product as cart_product

    dims = list(range(n_inputs))
    results = []

    for funcs in cart_product(basis, repeat=n_inputs):
        specs = list(zip(funcs, dims))
        head = ComposeHead(n_inputs=n_inputs, primitives=[], repeat=0,
                           products=False, separable=False)
        term = SeparableTerm(specs, n_inputs)
        head.terms = nn.ModuleList([term])

        optimizer = torch.optim.Adam(head.parameters(), lr=lr)
        for _ in range(n_steps):
            pred = head(data_in)
            loss = nn.functional.mse_loss(pred, target)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()

        final_loss = loss.item()
        dim_names = [f"x{d}" for d in dims]
        label = "*".join(f"{f}({dim_names[d]})" for f, d in specs)
        results.append((final_loss, label, head))

    results.sort(key=lambda r: r[0])
    return results


def fine_tune(head, data_in, target, n_steps=3000, lr=0.005):
    """Fine-tune a head on data."""
    optimizer = torch.optim.Adam(head.parameters(), lr=lr)
    for _ in range(n_steps):
        pred = head(data_in)
        loss = nn.functional.mse_loss(pred, target)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        optimizer.step()
    return loss.item()


def normalize_and_snap(head, tolerance=0.05):
    """Deep-copy, normalize, and snap a head."""
    head_snap = copy.deepcopy(head)
    for t in head_snap.terms:
        if isinstance(t, SeparableTerm):
            t.normalize()
    head_snap.snap_coefficients(tolerance=tolerance)
    return head_snap


def eval_snap_error(head, test_in, u_true):
    """Mean absolute error on clean test data."""
    with torch.no_grad():
        pred = head(test_in)
        return (pred - u_true).abs().mean().item()


def check_exact_recovery(head_snap, input_names=("x", "y", "t")):
    """Check whether the snapped expression exactly matches the target."""
    expr = head_snap.to_symbolic(input_names=list(input_names))
    s = expr.string
    # The exact target: sin(x)*cos(y)*exp(-0.2*t)
    # After snap, bias should be 0 and expression should contain sin, cos, exp
    # We check numerically on a grid
    import math
    xs = [0.5, 1.0, 2.0, 3.0, 5.0]
    max_err = 0.0
    for xv in xs:
        for yv in xs:
            for tv in [0.0, 0.5, 1.0]:
                true_val = math.sin(xv) * math.cos(yv) * math.exp(-0.2 * tv)
                test_pt = torch.tensor([[xv, yv, tv]])
                with torch.no_grad():
                    pred_val = head_snap(test_pt).item()
                max_err = max(max_err, abs(pred_val - true_val))
    return max_err < 0.02


# ---------------------------------------------------------------------------
# Part 1: Noise robustness
# ---------------------------------------------------------------------------

print("=" * 70)
print("PART 1: NOISE ROBUSTNESS ON TAYLOR-GREEN u-COMPONENT")
print("  Target: u(x,y,t) = sin(x)*cos(y)*exp(-0.2*t)")
print("=" * 70)

noise_levels = [0.0, 0.001, 0.01, 0.05, 0.1, 0.2]
basis_trig = ["sin", "cos", "exp"]
summary_rows = []

# Shared clean test set
torch.manual_seed(9999)
xt = torch.rand(5000, 1) * PI2
yt = torch.rand(5000, 1) * PI2
tt = torch.rand(5000, 1) * 1.0
test_in = torch.cat([xt, yt, tt], dim=1)
u_test_true = torch.sin(xt) * torch.cos(yt) * torch.exp(-0.2 * tt)

for i, sigma in enumerate(noise_levels):
    seed = 100 + i
    print(f"\n--- sigma = {sigma} (seed={seed}) ---")

    # Generate data with noise
    torch.manual_seed(seed)
    data_in, u_true = make_taylor_green_data(3000, seed=seed)
    torch.manual_seed(seed + 1000)
    u_noisy = u_true + sigma * torch.randn_like(u_true)

    # Brute-force search (27 candidates)
    results = brute_force_search(data_in, u_noisy, basis_trig, n_inputs=3,
                                 n_steps=2000, lr=0.01)
    print(f"  Best candidate: {results[0][1]} (search loss={results[0][0]:.8f})")

    # Fine-tune best
    best_head = results[0][2]
    final_loss = fine_tune(best_head, data_in, u_noisy, n_steps=3000, lr=0.005)
    print(f"  After fine-tuning: loss={final_loss:.8f}")

    # Normalize + snap
    head_snap = normalize_and_snap(best_head, tolerance=0.05)
    snap_err = eval_snap_error(head_snap, test_in, u_test_true)
    recovered = check_exact_recovery(head_snap)

    expr = head_snap.to_symbolic(input_names=["x", "y", "t"])
    print(f"  Snap error (clean test): {snap_err:.8f}")
    print(f"  Exact recovery: {'YES' if recovered else 'NO'}")
    print(f"  Expression: {expr.string}")

    summary_rows.append((sigma, final_loss, snap_err, recovered))

# Summary table
print("\n" + "=" * 70)
print("SUMMARY TABLE")
print("=" * 70)
print(f"{'sigma':>8s}  {'train_loss':>12s}  {'snap_err':>12s}  {'recovered':>10s}")
print("-" * 48)
for sigma, train_loss, snap_err, recovered in summary_rows:
    print(f"{sigma:8.3f}  {train_loss:12.8f}  {snap_err:12.8f}  {'YES' if recovered else 'NO':>10s}")


# ---------------------------------------------------------------------------
# Part 2: Wrong basis
# ---------------------------------------------------------------------------

print("\n" + "=" * 70)
print("PART 2: WRONG BASIS — only {exp, ln}, no trig functions")
print("  Target: u(x,y,t) = sin(x)*cos(y)*exp(-0.2*t)")
print("=" * 70)

torch.manual_seed(200)
data_in, u_true = make_taylor_green_data(3000, seed=200)

wrong_basis = ["exp", "ln"]
results_wrong = brute_force_search(data_in, u_true, wrong_basis, n_inputs=3,
                                   n_steps=2000, lr=0.01)

print("\n  All candidates (sorted by loss):")
for loss_val, label, _ in results_wrong:
    print(f"    {label}: loss={loss_val:.8f}")

best_loss_wrong = results_wrong[0][0]
print(f"\n  Best candidate loss: {best_loss_wrong:.6f}")

if best_loss_wrong > 0.1:
    print("  >> All candidates have loss > 0.1.")
    print("  >> The method correctly fails: the target cannot be represented")
    print("     with only {exp, ln} (trig functions are required).")
else:
    print(f"  >> Unexpected: best loss is {best_loss_wrong:.6f} (expected > 0.1)")


# ---------------------------------------------------------------------------
# Part 3: Non-separable target (graceful failure)
# ---------------------------------------------------------------------------

print("\n" + "=" * 70)
print("PART 3: NON-SEPARABLE TARGET — sin(x - 2*t)")
print("  A traveling wave that is NOT a single separable product.")
print("=" * 70)

torch.manual_seed(300)
n = 3000
x_wave = torch.rand(n, 1) * PI2
t_wave = torch.rand(n, 1) * 1.0
data_wave = torch.cat([x_wave, t_wave], dim=1)
u_wave = torch.sin(x_wave - 2.0 * t_wave)

wave_basis = ["sin", "cos", "exp"]
results_wave = brute_force_search(data_wave, u_wave, wave_basis, n_inputs=2,
                                  n_steps=2000, lr=0.01)

print("\n  All candidates (sorted by loss):")
for loss_val, label, _ in results_wave:
    print(f"    {label}: loss={loss_val:.8f}")

best_loss_wave = results_wave[0][0]

# Fine-tune the best to give it every chance
best_wave_head = results_wave[0][2]
best_loss_wave = fine_tune(best_wave_head, data_wave, u_wave, n_steps=3000, lr=0.005)

print(f"\n  Best single-term loss after fine-tuning: {best_loss_wave:.6f}")
print(f"\n  No single separable term fits the data (best loss = {best_loss_wave:.2f}).")
print("  This correctly indicates the target is not a single-term separable")
print("  function. Multi-term sequential fitting (see example 10) is required.")
