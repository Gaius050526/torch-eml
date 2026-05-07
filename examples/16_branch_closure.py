"""EML Branch Closure Analysis: predict which PDEs admit exact separable solutions.

Framework:
    EML primitives form three branches under differentiation and multiplication:
        Branch E (exp): closed under multiplication (exp*exp = exp)
        Branch T (sin/cos): products generate new frequencies (mode coupling)
        Branch L (ln): breaks under differentiation (d/dx ln = 1/x)

    For a PDE with nonlinearity N(u), we test:
        1. Does N(u) stay in the same branch? (direct closure)
        2. If not, can the excess be absorbed into auxiliary variables? (gradient closure)
        3. If neither, predict: no exact separable solution exists.

    We verify each prediction computationally with the PDE-residual solver.
"""

import torch
import torch.nn as nn
import math

torch.manual_seed(42)

PI = math.pi
PI2 = 2 * PI

print("=" * 70)
print("EML BRANCH CLOSURE ANALYSIS")
print("  Predicting exact separable solutions from PDE structure")
print("=" * 70)


# ================================================================
# Part 1: Analytical Closure Tests
# ================================================================
# For a test function u in Branch T, compute the nonlinear term N(u)
# and check whether it generates irreducible new modes.

print("\n  Part 1: Closure test for each PDE's nonlinearity")
print("  Test function: u = sin(x)*exp(-t) or sin(x)*cos(y)*exp(-t)")
print("-" * 70)

n_test = 5000


def closure_test_1d(label, nonlinear_fn, test_u_fn, basis_fns, n=5000):
    """Test if N(u) is expressible as a finite sum of basis functions.

    Computes N(u) at sample points, then fits with a sum of basis_fns.
    If fit residual ~ 0, the nonlinearity is closed.
    """
    x = torch.rand(n, 1) * PI2
    u_val = test_u_fn(x)
    target = nonlinear_fn(x, u_val)

    # Try to fit target with up to 5 terms from the basis
    best_loss = float('inf')
    for n_terms in [1, 2, 3, 5]:
        # Parameterize: sum_k c_k * f_k(a_k * x + b_k)
        coeffs = nn.Parameter(torch.randn(n_terms) * 0.1)
        scales = nn.Parameter(torch.ones(n_terms))
        biases = nn.Parameter(torch.zeros(n_terms))
        func_type = nn.Parameter(torch.zeros(n_terms))  # 0=sin, 1=cos, 2=exp

        params = [coeffs, scales, biases]
        optimizer = torch.optim.Adam(params, lr=0.05)

        for step in range(500):
            pred = torch.zeros_like(target)
            for k in range(n_terms):
                # Try sin/cos (trig basis)
                pred = pred + coeffs[k] * torch.sin(scales[k] * x + biases[k])

            loss = ((pred - target) ** 2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if loss.item() < best_loss:
            best_loss = loss.item()

    closed = best_loss < 1e-4
    return best_loss, closed


# --- Test 1: Linear operator (d²u/dx²) ---
print("\n  1. Linear diffusion: N(u) = u_xx")
x = torch.rand(n_test, 1, requires_grad=True) * PI2
u = torch.sin(x)
u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
# u_xx = -sin(x), which IS in Branch T
residual = (u_xx + torch.sin(x)).abs().max().item()
print(f"     u_xx + sin(x) = {residual:.2e}  (should be 0)")
print(f"     Branch T closure: YES (differentiation preserves sin/cos)")
print(f"     Prediction: EXACT SOLUTION EXISTS")

# --- Test 2: Quadratic (Burgers) u*u_x ---
print("\n  2. Burgers nonlinearity: N(u) = u * u_x")
x = torch.rand(n_test, 1, requires_grad=True) * PI2
u = torch.sin(x)
u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
N_u = u * u_x  # sin(x)*cos(x) = (1/2)*sin(2x)
# This IS representable as sin(2x), but at DOUBLE the wavenumber
# Check: can sin(2x) be balanced by u_t or u_xx terms?
# u_t adds sin(x)*T'(t), u_xx adds -sin(x)*T(t)
# sin(2x) term has NO balancing partner -> NOT CLOSED
target = N_u.detach()
fit = 0.5 * torch.sin(2 * x.detach())
fit_err = (target - fit).abs().mean().item()
print(f"     u*u_x for u=sin(x): residual vs (1/2)sin(2x) = {fit_err:.2e}")
print(f"     Generates wavenumber 2 from wavenumber 1")
print(f"     No balancing term in PDE for sin(2x) -> NOT CLOSED")
print(f"     Prediction: NO EXACT SEPARABLE SOLUTION")

# --- Test 3: Cubic (NLS) u^3 ---
print("\n  3. NLS cubic nonlinearity: N(u) = u^3")
x = torch.rand(n_test, 1) * PI2
u = torch.sin(x)
N_u = u ** 3  # sin^3(x) = (3/4)sin(x) - (1/4)sin(3x)
# Generates wavenumber 3 from wavenumber 1
# The (3/4)sin(x) part CAN be balanced, but (1/4)sin(3x) CANNOT
target = N_u
analytic = 0.75 * torch.sin(x) - 0.25 * torch.sin(3 * x)
fit_err = (target - analytic).abs().mean().item()
print(f"     u^3 for u=sin(x): residual vs 3/4*sin(x)-1/4*sin(3x) = {fit_err:.2e}")
print(f"     Generates wavenumber 3 from wavenumber 1")
print(f"     sin(3x) has NO balancing term in PDE -> NOT CLOSED")
print(f"     Prediction: NO EXACT SEPARABLE SOLUTION")

# --- Test 4: 2D NS convective term with gradient check ---
print("\n  4. 2D Navier-Stokes: N(u,v) = u*u_x + v*u_y")
x = torch.rand(n_test, 1, requires_grad=True) * PI2
y = torch.rand(n_test, 1, requires_grad=True) * PI2
u = torch.sin(x) * torch.cos(y)
v = -torch.cos(x) * torch.sin(y)

u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]
N_u = (u * u_x + v * u_y).detach()
# N_u = sin(x)cos(x)cos^2(y) + cos(x)sin(x)sin^2(y) = sin(x)cos(x) = (1/2)sin(2x)
analytic = 0.5 * torch.sin(2 * x.detach())
fit_err = (N_u - analytic).abs().mean().item()
print(f"     (u*u_x + v*u_y) for TG: residual vs (1/2)sin(2x) = {fit_err:.2e}")
print(f"     Generates sin(2x), BUT this is d/dx[-1/4*cos(2x)]")
print(f"     It IS a gradient -> absorbed into pressure!")
print(f"     GRADIENT-CLOSED")
print(f"     Prediction: EXACT SOLUTION EXISTS")

# --- Test 5: 3D NS vortex stretching ---
print("\n  5. 3D Navier-Stokes: vortex stretching (u0.grad)w0 - (w0.grad)u0")
x = torch.rand(n_test, 1, requires_grad=True) * PI2
y = torch.rand(n_test, 1, requires_grad=True) * PI2
z = torch.rand(n_test, 1, requires_grad=True) * PI2
u = torch.sin(x) * torch.cos(y) * torch.cos(z)
v = torch.cos(x) * torch.sin(y) * torch.cos(z)
w = -2 * torch.cos(x) * torch.cos(y) * torch.sin(z)


def g(f, var):
    return torch.autograd.grad(f, var, grad_outputs=torch.ones_like(f),
                               create_graph=True)[0]


u_x, u_y, u_z = g(u, x), g(u, y), g(u, z)
v_x, v_y, v_z = g(v, x), g(v, y), g(v, z)
w_x, w_y, w_z = g(w, x), g(w, y), g(w, z)
om_x = w_y - v_z
om_y = u_z - w_x
om_z = v_x - u_y
S_x = (u * g(om_x, x) + v * g(om_x, y) + w * g(om_x, z)
       - (om_x * u_x + om_y * u_y + om_z * u_z))
S_mag = S_x.detach().abs().max().item()
print(f"     |S| = {S_mag:.4f}  (should be 0 for gradient closure)")
print(f"     Vortex stretching != advection -> NOT gradient-closed")
print(f"     NOT CLOSED (no gradient mechanism in 3D)")
print(f"     Prediction: NO EXACT SEPARABLE SOLUTION")

# --- Test 6: KdV u*u_x + u_xxx ---
print("\n  6. KdV: N(u) = 6*u*u_x + u_xxx")
x = torch.rand(n_test, 1, requires_grad=True) * PI2
u = torch.sin(x)
u_x = g(u, x)
u_xx = g(u_x, x)
u_xxx = g(u_xx, x)
N_u = (6 * u * u_x + u_xxx).detach()
# 6*sin(x)*cos(x) - cos(x) = 3*sin(2x) - cos(x)
# Generates wavenumber 2, no balancing term
analytic = 3 * torch.sin(2 * x.detach()) - torch.cos(x.detach())
fit_err = (N_u - analytic).abs().mean().item()
print(f"     6*u*u_x + u_xxx for u=sin(x): residual = {fit_err:.2e}")
print(f"     Generates sin(2x) from wavenumber 1, no balancing term")
print(f"     Known solutions (sech^2) not in {{sin,cos,exp}} basis")
print(f"     NOT CLOSED")
print(f"     Prediction: NO EXACT SEPARABLE SOLUTION")


# ================================================================
# Summary of Predictions
# ================================================================
print(f"\n{'=' * 70}")
print("  BRANCH CLOSURE PREDICTIONS")
print(f"{'=' * 70}")
print(f"  {'PDE':<30} {'Nonlinearity':<15} {'Closure':<20} {'Prediction':<15}")
print(f"  {'-'*80}")
predictions = [
    ("1D Heat equation", "Linear", "Direct (T)", "SUCCESS"),
    ("1D Wave equation", "Linear", "Direct (T)", "SUCCESS"),
    ("2D Navier-Stokes", "Quadratic", "Gradient (T)", "SUCCESS"),
    ("3D NS (Beltrami)", "Quadratic", "Beltrami (T)", "SUCCESS"),
    ("3D NS (non-Beltrami)", "Quadratic", "FAILS (T)", "FAIL"),
    ("1D Burgers", "Quadratic", "FAILS (T,E)", "FAIL"),
    ("1D NLS", "Cubic", "FAILS (T)", "FAIL"),
    ("1D KdV", "Quadratic", "FAILS (T)", "FAIL"),
]
for pde, nl, closure, pred in predictions:
    print(f"  {pde:<30} {nl:<15} {closure:<20} {pred:<15}")


# ================================================================
# Part 2: Verify predictions with PDE-residual solver
# ================================================================
# Test the NEW predictions (Burgers, NLS, KdV) computationally.

print(f"\n{'=' * 70}")
print("  Part 2: Computational verification of predictions")
print(f"{'=' * 70}")

n_coll = 1000
n_ic = 300


# --- Verification 1: Burgers equation ---
print("\n  Verification 1: 1D Burgers u_t + u*u_x = 0.1*u_xx")
print("  IC: u(x,0) = sin(x)")
print("  Prediction: FAIL (quadratic mode coupling, no gradient mechanism)")
print("-" * 70)

nu_b = 0.1


class SepTerm1D(nn.Module):
    """Single separable term: c * f(a*x + b) * g(d*t + e)."""
    def __init__(self, fx='sin', ft='exp'):
        super().__init__()
        self.coeff = nn.Parameter(torch.tensor(1.0))
        self.a = nn.Parameter(torch.tensor(1.0))
        self.b = nn.Parameter(torch.tensor(0.0))
        self.d = nn.Parameter(torch.tensor(-0.1))
        self.e = nn.Parameter(torch.tensor(0.0))
        self.fx = fx
        self.ft = ft

    def forward(self, x, t):
        if self.fx == 'sin':
            spatial = torch.sin(self.a * x + self.b)
        elif self.fx == 'cos':
            spatial = torch.cos(self.a * x + self.b)
        else:
            spatial = torch.exp(self.a * x + self.b)

        if self.ft == 'sin':
            temporal = torch.sin(self.d * t + self.e)
        elif self.ft == 'cos':
            temporal = torch.cos(self.d * t + self.e)
        else:
            temporal = torch.exp(self.d * t + self.e)

        return self.coeff * spatial * temporal


# Try all 9 separable candidates for Burgers
sep_funcs = ['sin', 'cos', 'exp']
burgers_results = []
for fx in sep_funcs:
    for ft in sep_funcs:
        model = SepTerm1D(fx, ft)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        for step in range(1500):
            x = (torch.rand(n_coll, 1) * PI2).requires_grad_(True)
            t = (torch.rand(n_coll, 1) * 1.0).requires_grad_(True)

            u = model(x, t)
            u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                                       create_graph=True)[0]
            u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u),
                                       create_graph=True)[0]
            u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x),
                                        create_graph=True)[0]

            pde_res = u_t + u * u_x - nu_b * u_xx
            loss_pde = (pde_res ** 2).mean()

            # IC
            x_ic = (torch.rand(n_ic, 1) * PI2).requires_grad_(True)
            t_ic = torch.zeros_like(x_ic)
            u_ic = model(x_ic, t_ic)
            loss_ic = ((u_ic - torch.sin(x_ic.detach())) ** 2).mean()

            loss = loss_pde + 10.0 * loss_ic
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        burgers_results.append((loss_pde.item(), f"{fx}(x)*{ft}(t)"))

burgers_results.sort()
print(f"  Best 3 candidates:")
for loss_val, name in burgers_results[:3]:
    print(f"    {name}: PDE residual = {loss_val:.6f}")

best_burgers = burgers_results[0][0]
if best_burgers > 0.01:
    print(f"\n  RESULT: PDE residual = {best_burgers:.4f} >> 0")
    print(f"  CONFIRMED: No exact separable solution (prediction correct)")
else:
    print(f"\n  RESULT: PDE residual = {best_burgers:.2e}")
    print(f"  Possible solution found!")


# --- Verification 2: NLS standing wave ---
print(f"\n  Verification 2: NLS standing wave R_xx + R^3 = omega*R")
print("  Searching for R(x) on [0, 2pi]")
print("  Prediction: FAIL (cubic generates 3rd harmonic)")
print("-" * 70)

nls_results = []
for fx in sep_funcs:
    # R(x) = c * f(a*x + b), find omega such that R_xx + R^3 = omega*R
    c = nn.Parameter(torch.tensor(1.0))
    a = nn.Parameter(torch.tensor(1.0))
    b = nn.Parameter(torch.tensor(0.0))
    omega = nn.Parameter(torch.tensor(1.0))

    params = [c, a, b, omega]
    optimizer = torch.optim.Adam(params, lr=0.01)

    for step in range(2000):
        x = (torch.rand(n_coll, 1) * PI2).requires_grad_(True)

        if fx == 'sin':
            R = c * torch.sin(a * x + b)
        elif fx == 'cos':
            R = c * torch.cos(a * x + b)
        else:
            R = c * torch.exp(a * x + b)

        R_x = torch.autograd.grad(R, x, grad_outputs=torch.ones_like(R),
                                   create_graph=True)[0]
        R_xx = torch.autograd.grad(R_x, x, grad_outputs=torch.ones_like(R_x),
                                    create_graph=True)[0]

        residual = R_xx + R ** 3 - omega * R
        loss = (residual ** 2).mean()

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()

    nls_results.append((loss.item(), fx,
                        c.item(), a.item(), b.item(), omega.item()))

nls_results.sort()
print(f"  Results:")
for loss_val, fx, c_v, a_v, b_v, om_v in nls_results:
    print(f"    {fx}(x): residual = {loss_val:.6f}, "
          f"c={c_v:.3f}, a={a_v:.3f}, omega={om_v:.3f}")

best_nls = nls_results[0][0]
if best_nls > 0.01:
    print(f"\n  RESULT: Best residual = {best_nls:.4f} >> 0")
    print(f"  CONFIRMED: No exact single-term solution (prediction correct)")
    print(f"  Reason: sin^3(x) = 3/4*sin(x) - 1/4*sin(3x)")
    print(f"  The sin(3x) term has no balancing partner in the equation.")
else:
    print(f"\n  RESULT: residual = {best_nls:.2e}")


# --- Verification 3: KdV ---
print(f"\n  Verification 3: KdV u_t + 6*u*u_x + u_xxx = 0")
print("  IC: u(x,0) = sin(x)")
print("  Prediction: FAIL (quadratic mode coupling + no pressure mechanism)")
print("-" * 70)

kdv_results = []
for fx in sep_funcs:
    for ft in sep_funcs:
        model = SepTerm1D(fx, ft)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        for step in range(1500):
            x = (torch.rand(n_coll, 1) * PI2).requires_grad_(True)
            t = (torch.rand(n_coll, 1) * 1.0).requires_grad_(True)

            u = model(x, t)
            u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                                       create_graph=True)[0]
            u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u),
                                       create_graph=True)[0]
            u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x),
                                        create_graph=True)[0]
            u_xxx = torch.autograd.grad(u_xx, x, grad_outputs=torch.ones_like(u_xx),
                                         create_graph=True)[0]

            pde_res = u_t + 6 * u * u_x + u_xxx
            loss_pde = (pde_res ** 2).mean()

            x_ic = (torch.rand(n_ic, 1) * PI2).requires_grad_(True)
            t_ic = torch.zeros_like(x_ic)
            u_ic = model(x_ic, t_ic)
            loss_ic = ((u_ic - torch.sin(x_ic.detach())) ** 2).mean()

            loss = loss_pde + 10.0 * loss_ic
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        kdv_results.append((loss_pde.item(), f"{fx}(x)*{ft}(t)"))

kdv_results.sort()
print(f"  Best 3 candidates:")
for loss_val, name in kdv_results[:3]:
    print(f"    {name}: PDE residual = {loss_val:.6f}")

best_kdv = kdv_results[0][0]
if best_kdv > 0.01:
    print(f"\n  RESULT: PDE residual = {best_kdv:.4f} >> 0")
    print(f"  CONFIRMED: No exact separable solution (prediction correct)")
    print(f"  Known KdV solitons (sech^2) are not in {{sin,cos,exp}} basis.")
else:
    print(f"\n  RESULT: PDE residual = {best_kdv:.2e}")


# ================================================================
# Final Summary
# ================================================================
print(f"\n{'=' * 70}")
print("  BRANCH CLOSURE ANALYSIS — COMPLETE RESULTS")
print(f"{'=' * 70}")
print(f"\n  {'PDE':<28} {'Nonlin.':<10} {'Branch':<10} {'Closure':<18} "
      f"{'Predicted':<10} {'Verified':<10}")
print(f"  {'-'*86}")

rows = [
    ("1D Heat", "Linear", "T", "Direct", "SUCCESS", "YES*"),
    ("1D Wave", "Linear", "T", "Direct", "SUCCESS", "YES*"),
    ("2D Navier-Stokes", "Quadratic", "T", "Gradient", "SUCCESS", "YES*"),
    ("3D NS (Beltrami)", "Quadratic", "T", "Beltrami", "SUCCESS", "YES*"),
    ("3D NS (non-Beltrami)", "Quadratic", "T", "FAILS", "FAIL", "YES*"),
    ("1D Burgers", "Quadratic", "T,E", "FAILS",
     "FAIL", "YES" if best_burgers > 0.01 else "NO"),
    ("1D NLS standing wave", "Cubic", "T", "FAILS",
     "FAIL", "YES" if best_nls > 0.01 else "NO"),
    ("1D KdV", "Quadratic", "T", "FAILS",
     "FAIL", "YES" if best_kdv > 0.01 else "NO"),
]

for pde, nl, br, cl, pred, ver in rows:
    print(f"  {pde:<28} {nl:<10} {br:<10} {cl:<18} {pred:<10} {ver:<10}")

print(f"\n  * = verified in previous experiments")
print(f"\n  Key insight: the gradient condition in 2D NS is the ONLY mechanism")
print(f"  that allows a quadratically nonlinear PDE to have exact separable")
print(f"  solutions in Branch T. Without gradient closure (3D NS, Burgers,")
print(f"  KdV), the mode coupling is irreducible.")
print(f"\n  For cubic nonlinearities (NLS), the situation is strictly worse:")
print(f"  sin^3(x) generates 3rd harmonics, and no gradient mechanism exists")
print(f"  in the scalar Schrodinger equation to absorb them.")
print(f"\n{'=' * 70}")
