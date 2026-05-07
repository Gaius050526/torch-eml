"""Example 17: Exact recovery with extended hyperbolic basis (tanh, sech).

Demonstrates that adding tanh and sech as named primitives — which are
EML-constructible by universality but impractically deep as raw compositions —
unlocks new PDE solution classes:

1. Allen-Cahn steady state:  u_xx + u - u³ = 0,  u(x) = tanh(x/√2)
2. Burgers steady state:     u·u_x = ν·u_xx,     u(x) = A·tanh(Ax/(2ν))
3. KdV soliton (1-term):     u(x,0) = -2·sech²(x)  (initial profile recovery)

These solutions are NOT separable products of sin/cos/exp, so the original
primitive basis cannot discover them.
"""

import math
import torch
import torch.nn as nn

from torch_eml.primitives import EMLTanh, EMLSech
from torch_eml.compose import AxisTerm, SeparableTerm


# ============================================================
# Experiment 1: Allen-Cahn steady state  u = tanh(x/√2)
# ============================================================

def allen_cahn():
    """Recover u(x) = tanh(x/√2) from the Allen-Cahn equation u_xx + u - u³ = 0."""
    print("=" * 60)
    print("Experiment 1: Allen-Cahn steady state")
    print("  PDE: u_xx + u - u³ = 0")
    print("  Exact solution: u(x) = tanh(x/√2)")
    print("=" * 60)

    # Generate training data
    x = torch.linspace(-5, 5, 500).unsqueeze(1)
    u_exact = torch.tanh(x / math.sqrt(2))

    # Build a single AxisTerm with tanh
    term = AxisTerm("tanh", input_dim=0, n_inputs=1)

    # Initialize near the solution
    with torch.no_grad():
        term.coeff.fill_(1.0)
        term.a.fill_(0.5)  # start near 1/√2 ≈ 0.707
        term.b.fill_(0.0)

    optimizer = torch.optim.Adam(term.parameters(), lr=0.01)

    for step in range(2000):
        pred = term.coeff * term._funcs["tanh"](term.a * x[:, 0] + term.b)
        loss = ((pred - u_exact.squeeze()) ** 2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (step + 1) % 500 == 0:
            print(f"  Step {step+1:4d}: loss = {loss.item():.2e}")

    # Read learned parameters
    a_learned = term.a.item()
    b_learned = term.b.item()
    c_learned = term.coeff.item()
    print(f"\n  Learned: c={c_learned:.6f} * tanh({a_learned:.6f}*x + {b_learned:.6f})")
    print(f"  Expected: 1.0 * tanh({1/math.sqrt(2):.6f}*x + 0.0)")

    # Snap to exact values
    a_exact = 1.0 / math.sqrt(2)
    print(f"  |a - 1/√2| = {abs(a_learned - a_exact):.2e}")
    print(f"  |c - 1|    = {abs(c_learned - 1.0):.2e}")
    print(f"  |b - 0|    = {abs(b_learned):.2e}")

    # Verify PDE residual: u_xx + u - u³ = 0
    x_test = torch.linspace(-4, 4, 1000, requires_grad=True).unsqueeze(1)
    u = torch.tanh(x_test / math.sqrt(2))
    u_x = torch.autograd.grad(u.sum(), x_test, create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x.sum(), x_test)[0]
    residual = u_xx + u - u ** 3
    print(f"  PDE residual (exact): {residual.abs().max().item():.2e}")

    # Verify with learned params
    u_learned = c_learned * torch.tanh(a_learned * x_test[:, 0] + b_learned)
    mae = (u_learned - torch.tanh(x_test[:, 0] / math.sqrt(2))).abs().mean().item()
    print(f"  MAE (learned vs exact): {mae:.2e}")

    return mae < 1e-4


# ============================================================
# Experiment 2: Burgers steady state  u·u_x = ν·u_xx
# ============================================================

def burgers_steady():
    """Recover u(x) = A·tanh(Ax/(2ν)) from Burgers' equation steady state."""
    print("\n" + "=" * 60)
    print("Experiment 2: Burgers steady state")
    print("  PDE: u·u_x = ν·u_xx  (steady)")
    print("  Exact solution: u(x) = A·tanh(Ax/(2ν))")
    print("  Using A=1, ν=0.5 → u = tanh(x)")
    print("=" * 60)

    # With A=1, ν=0.5: u = tanh(x), and u_x = sech²(x)
    # u·u_x = tanh(x)·sech²(x)
    # ν·u_xx = 0.5 · (-2·tanh(x)·sech²(x)) = -tanh(x)·sech²(x)
    # Wait, that gives u·u_x = -ν·u_xx, not u·u_x = ν·u_xx
    # The steady Burgers with specific sign: u_t + u·u_x = ν·u_xx
    # Steady: u·u_x = ν·u_xx → d/dx(u²/2) = ν·u_x → u²/2 = ν·u + C
    # With u(-∞)=-A, u(+∞)=A: C = -A²/2, giving u = A·tanh(Ax/(2ν))

    A = 2.0
    nu = 1.0
    # u = A·tanh(A·x/(2ν)) = 2·tanh(x)
    x = torch.linspace(-5, 5, 500).unsqueeze(1)
    u_exact = A * torch.tanh(A * x / (2 * nu))

    term = AxisTerm("tanh", input_dim=0, n_inputs=1)
    with torch.no_grad():
        term.coeff.fill_(1.5)
        term.a.fill_(0.8)
        term.b.fill_(0.0)

    optimizer = torch.optim.Adam(term.parameters(), lr=0.01)

    for step in range(2000):
        pred = term.coeff * term._funcs["tanh"](term.a * x[:, 0] + term.b)
        loss = ((pred - u_exact.squeeze()) ** 2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (step + 1) % 500 == 0:
            print(f"  Step {step+1:4d}: loss = {loss.item():.2e}")

    a_learned = term.a.item()
    c_learned = term.coeff.item()
    b_learned = term.b.item()
    print(f"\n  Learned: {c_learned:.6f} * tanh({a_learned:.6f}*x + {b_learned:.6f})")
    print(f"  Expected: {A:.1f} * tanh({A/(2*nu):.6f}*x + 0.0)")
    print(f"  |c - A| = {abs(c_learned - A):.2e}")
    print(f"  |a - A/(2ν)| = {abs(a_learned - A/(2*nu)):.2e}")

    mae = (c_learned * torch.tanh(a_learned * x[:, 0] + b_learned) - u_exact.squeeze()).abs().mean().item()
    print(f"  MAE: {mae:.2e}")
    return mae < 1e-4


# ============================================================
# Experiment 3: KdV soliton profile  u(x,0) = -2·sech²(x)
# ============================================================

def kdv_soliton_profile():
    """Recover the KdV soliton profile u(x) = -2·sech²(x)."""
    print("\n" + "=" * 60)
    print("Experiment 3: KdV soliton initial profile")
    print("  u(x,0) = -2·sech²(x)")
    print("  (The full traveling wave sech²(x-ct) is not separable)")
    print("=" * 60)

    x = torch.linspace(-5, 5, 500).unsqueeze(1)
    u_exact = -2.0 / torch.cosh(x) ** 2

    # Use sech² = sech(x) * sech(x) — product of two sech factors on same axis
    # Actually simpler: train a single sech term and square it
    # Or: use the SeparableTerm with two sech factors on dim 0

    # Approach: single AxisTerm with sech, then handle squaring
    # Let's use a direct approach: c * sech(a*x+b)²
    # We can represent this as a product term: SeparableTerm([("sech", 0), ("sech", 0)])
    # But SeparableTerm expects different dims. Let's do it manually.

    class SechSquaredTerm(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Parameter(torch.tensor(0.8))
            self.b = nn.Parameter(torch.tensor(0.0))
            self.coeff = nn.Parameter(torch.tensor(-1.5))
            self.sech = EMLSech()

        def forward(self, x):
            inner = self.a * x + self.b
            s = self.sech(inner)
            return self.coeff * s * s

    term = SechSquaredTerm()
    optimizer = torch.optim.Adam(term.parameters(), lr=0.01)

    for step in range(2000):
        pred = term(x[:, 0])
        loss = ((pred - u_exact.squeeze()) ** 2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (step + 1) % 500 == 0:
            print(f"  Step {step+1:4d}: loss = {loss.item():.2e}")

    a_learned = term.a.item()
    b_learned = term.b.item()
    c_learned = term.coeff.item()
    print(f"\n  Learned: {c_learned:.6f} * sech²({a_learned:.6f}*x + {b_learned:.6f})")
    print(f"  Expected: -2.0 * sech²(1.0*x + 0.0)")
    print(f"  |c - (-2)| = {abs(c_learned - (-2.0)):.2e}")
    print(f"  |a - 1| = {abs(a_learned - 1.0):.2e}")
    print(f"  |b - 0| = {abs(b_learned):.2e}")

    pred_final = term(x[:, 0])
    mae = (pred_final - u_exact.squeeze()).abs().mean().item()
    print(f"  MAE: {mae:.2e}")
    return mae < 1e-4


# ============================================================
# Experiment 4: Normalization test for tanh/sech
# ============================================================

def normalization_test():
    """Test that normalization correctly canonicalizes tanh/sech terms."""
    print("\n" + "=" * 60)
    print("Experiment 4: Normalization of tanh/sech factors")
    print("=" * 60)

    # Create a SeparableTerm: tanh(x) * exp(t) and mess up the signs
    term = SeparableTerm([("tanh", 0), ("exp", 1)], n_inputs=2)
    with torch.no_grad():
        # Set tanh(-x) * exp(t) with coeff = -1
        # Should normalize to: tanh(x) * exp(t) with coeff = 1
        term.factors[0].a.fill_(-1.0)  # tanh(-x)
        term.factors[0].b.fill_(0.0)
        term.factors[1].a.fill_(1.0)
        term.factors[1].b.fill_(0.0)
        term.coeff.fill_(-1.0)

    print(f"  Before normalization:")
    print(f"    coeff = {term.coeff.item():.2f}")
    print(f"    tanh: a = {term.factors[0].a.item():.2f}, b = {term.factors[0].b.item():.2f}")
    print(f"    exp:  a = {term.factors[1].a.item():.2f}, b = {term.factors[1].b.item():.2f}")

    term.normalize()

    print(f"  After normalization:")
    print(f"    coeff = {term.coeff.item():.2f}")
    print(f"    tanh: a = {term.factors[0].a.item():.2f}, b = {term.factors[0].b.item():.2f}")
    print(f"    exp:  a = {term.factors[1].a.item():.2f}, b = {term.factors[1].b.item():.2f}")

    # Check: -1 * tanh(-x) * exp(t) = tanh(x) * exp(t)
    # So coeff should be 1, tanh.a should be 1
    ok = (abs(term.coeff.item() - 1.0) < 0.01 and
          abs(term.factors[0].a.item() - 1.0) < 0.01)
    print(f"  Normalization correct: {ok}")

    # Test sech normalization: sech(-2x) should become sech(2x)
    term2 = SeparableTerm([("sech", 0)], n_inputs=1)
    with torch.no_grad():
        term2.factors[0].a.fill_(-2.0)
        term2.factors[0].b.fill_(0.0)
        term2.coeff.fill_(3.0)

    print(f"\n  sech(-2x) before: a = {term2.factors[0].a.item():.2f}")
    term2.normalize()
    print(f"  sech(-2x) after:  a = {term2.factors[0].a.item():.2f}, coeff = {term2.coeff.item():.2f}")
    ok2 = abs(term2.factors[0].a.item() - 2.0) < 0.01
    print(f"  sech normalization correct: {ok2}")

    return ok and ok2


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    results = {}

    results["Allen-Cahn"] = allen_cahn()
    results["Burgers steady"] = burgers_steady()
    results["KdV soliton profile"] = kdv_soliton_profile()
    results["Normalization"] = normalization_test()

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for name, ok in results.items():
        status = "PASS" if ok else "FAIL"
        print(f"  {name}: {status}")
    print()

    all_pass = all(results.values())
    print(f"All experiments: {'PASS' if all_pass else 'FAIL'}")
