"""PDE-Residual-Only Solver: discover exact solutions from equations alone.

No training data from a known solution. The loss is:
    L = L_PDE (residual of governing equations)
      + L_IC  (initial condition penalty)
      + L_BC  (boundary condition penalty, if not periodic)

This is a "symbolic PINN" — the ComposeHead parameterizes the solution,
and autograd computes the PDE residuals for the loss.

Target: 2D incompressible Navier-Stokes (vorticity form to eliminate pressure)
    ω_t + u·∇ω = ν·∇²ω
    u_x + v_y = 0  (continuity)
    ω = v_x - u_y  (vorticity definition)

Approach:
    1. Brute-force search: try all separable triple-term candidates for u
    2. Derive v from continuity: v_y = -u_x → integrate
    3. Minimize PDE residual (vorticity eq) + IC penalty
    4. Periodic BCs assumed on [0, 2π]²

Validation: recover Taylor-Green vortex from equations alone.
Then: try novel initial conditions.
"""

import torch
import torch.nn as nn
import copy
import math
from itertools import product as cart_product
from torch_eml.compose import ComposeHead, SeparableTerm

torch.manual_seed(42)

PI = math.pi
PI2 = 2 * PI
nu = 0.1  # kinematic viscosity


def compute_derivatives(head, x, y, t):
    """Compute u and all needed spatial/temporal derivatives via autograd."""
    inputs = torch.cat([x, y, t], dim=1)
    u = head(inputs)  # [n, 1]

    # First derivatives
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                               create_graph=True)[0]
    u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u),
                               create_graph=True)[0]
    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u),
                               create_graph=True)[0]

    # Second derivatives
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x),
                                create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y),
                                create_graph=True)[0]

    return u, u_x, u_y, u_t, u_xx, u_yy


def pde_residual_loss(head_u, head_v, x, y, t, nu=0.1):
    """Compute Navier-Stokes residual loss (no pressure, using vorticity).

    Returns: total_loss, dict of component losses
    """
    u, u_x, u_y, u_t, u_xx, u_yy = compute_derivatives(head_u, x, y, t)
    v, v_x, v_y, v_t, v_xx, v_yy = compute_derivatives(head_v, x, y, t)

    # Continuity: u_x + v_y = 0
    continuity = u_x + v_y

    # Vorticity: ω = v_x - u_y
    omega = v_x - u_y

    # Vorticity derivatives
    omega_x = torch.autograd.grad(omega, x, grad_outputs=torch.ones_like(omega),
                                   create_graph=True)[0]
    omega_y = torch.autograd.grad(omega, y, grad_outputs=torch.ones_like(omega),
                                   create_graph=True)[0]
    omega_t = torch.autograd.grad(omega, t, grad_outputs=torch.ones_like(omega),
                                   create_graph=True)[0]
    omega_xx = torch.autograd.grad(omega_x, x, grad_outputs=torch.ones_like(omega_x),
                                    create_graph=True)[0]
    omega_yy = torch.autograd.grad(omega_y, y, grad_outputs=torch.ones_like(omega_y),
                                    create_graph=True)[0]

    # Vorticity equation: ω_t + u·ω_x + v·ω_y = ν·(ω_xx + ω_yy)
    vorticity_res = omega_t + u * omega_x + v * omega_y - nu * (omega_xx + omega_yy)

    loss_cont = (continuity ** 2).mean()
    loss_vort = (vorticity_res ** 2).mean()

    return loss_cont + loss_vort, {"continuity": loss_cont.item(), "vorticity": loss_vort.item()}


def ic_loss(head_u, head_v, x_ic, y_ic, u_ic, v_ic):
    """Initial condition penalty at t=0."""
    t_zero = torch.zeros_like(x_ic)
    inputs = torch.cat([x_ic, y_ic, t_zero], dim=1)
    u_pred = head_u(inputs)
    v_pred = head_v(inputs)
    return ((u_pred - u_ic) ** 2).mean() + ((v_pred - v_ic) ** 2).mean()


# ================================================================
# EXPERIMENT 1: Validate on Taylor-Green (known solution)
# ================================================================
print("=" * 70)
print("PDE-RESIDUAL-ONLY SOLVER — 2D Navier-Stokes")
print("  No solution data. Only equations + initial conditions.")
print("=" * 70)

# Collocation points (interior)
n_coll = 1500
n_ic = 500

sep_funcs = ["sin", "cos", "exp"]
candidates = list(cart_product(sep_funcs, repeat=3))  # 27 candidates

# Initial conditions: Taylor-Green
# u(x,y,0) = sin(x)*cos(y), v(x,y,0) = -cos(x)*sin(y)
print("\n  Initial conditions: u₀ = sin(x)cos(y), v₀ = -cos(x)sin(y)")
print(f"  Viscosity: ν = {nu}")
print(f"  Domain: [0, 2π]² × [0, 1]")
print(f"  Collocation points: {n_coll}, IC points: {n_ic}")

# Since u and v are coupled through the PDE, we search over (u,v) pairs.
# Key insight: for separable solutions, if u = f1(x)*f2(y)*f3(t),
# then continuity u_x + v_y = 0 constrains v.
# We search u candidates and derive v constraints from continuity.

# For efficiency, search u candidates by fitting IC + continuity only first,
# then verify PDE residual for the best candidates.

print(f"\n  Phase 1: Search {len(candidates)} candidates (IC + continuity, 1000 steps)")
print("-" * 70)

results = []
for funcs_u in candidates:
    # Create u head
    head_u = ComposeHead(n_inputs=3, primitives=[], repeat=0,
                         products=False, separable=False)
    term_u = SeparableTerm(list(zip(funcs_u, (0, 1, 2))), 3)
    head_u.terms = nn.ModuleList([term_u])

    # For v, try the "paired" candidate: if u ~ sin*cos*exp, try v ~ cos*sin*exp
    # This is motivated by continuity: d/dx[sin(ax)] ~ cos(ax), d/dy[cos(by)] ~ -sin(by)
    swap_map = {"sin": "cos", "cos": "sin", "exp": "exp"}
    funcs_v = (swap_map[funcs_u[0]], swap_map[funcs_u[1]], funcs_u[2])

    head_v = ComposeHead(n_inputs=3, primitives=[], repeat=0,
                         products=False, separable=False)
    term_v = SeparableTerm(list(zip(funcs_v, (0, 1, 2))), 3)
    head_v.terms = nn.ModuleList([term_v])

    # Train on IC + continuity (cheaper than full PDE)
    all_params = list(head_u.parameters()) + list(head_v.parameters())
    optimizer = torch.optim.Adam(all_params, lr=0.01)

    for step in range(1000):
        # IC points
        x_ic = torch.rand(n_ic, 1, requires_grad=True) * PI2
        y_ic = torch.rand(n_ic, 1, requires_grad=True) * PI2
        u_ic_target = torch.sin(x_ic.detach()) * torch.cos(y_ic.detach())
        v_ic_target = -torch.cos(x_ic.detach()) * torch.sin(y_ic.detach())
        loss_ic = ic_loss(head_u, head_v, x_ic, y_ic, u_ic_target, v_ic_target)

        # Continuity at collocation points
        x_c = torch.rand(n_coll, 1, requires_grad=True) * PI2
        y_c = torch.rand(n_coll, 1, requires_grad=True) * PI2
        t_c = torch.rand(n_coll, 1, requires_grad=True)
        inputs_c = torch.cat([x_c, y_c, t_c], dim=1)

        u_c = head_u(inputs_c)
        v_c = head_v(inputs_c)
        u_x = torch.autograd.grad(u_c, x_c, grad_outputs=torch.ones_like(u_c),
                                   create_graph=True)[0]
        v_y = torch.autograd.grad(v_c, y_c, grad_outputs=torch.ones_like(v_c),
                                   create_graph=True)[0]
        loss_cont = ((u_x + v_y) ** 2).mean()

        loss = loss_ic + 0.1 * loss_cont
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(all_params, 1.0)
        optimizer.step()

    final_loss = loss.item()
    fname_u = "*".join(f"{f}({['x','y','t'][d]})" for f, d in zip(funcs_u, (0,1,2)))
    fname_v = "*".join(f"{f}({['x','y','t'][d]})" for f, d in zip(funcs_v, (0,1,2)))
    results.append((final_loss, fname_u, fname_v, head_u, head_v))

results.sort()
print(f"\n  Top 5 (u, v) candidates:")
for loss_val, fname_u, fname_v, _, _ in results[:5]:
    print(f"    u={fname_u}, v={fname_v}: loss={loss_val:.6f}")

# ================================================================
# Phase 2: Fine-tune best candidate with FULL PDE residual
# ================================================================
print(f"\n  Phase 2: Fine-tune best with full PDE residual (3000 steps)")
print("-" * 70)

best_head_u = results[0][3]
best_head_v = results[0][4]

all_params = list(best_head_u.parameters()) + list(best_head_v.parameters())
optimizer = torch.optim.Adam(all_params, lr=0.005)

for step in range(3000):
    # Collocation points (fresh each step)
    x_c = torch.rand(n_coll, 1, requires_grad=True) * PI2
    y_c = torch.rand(n_coll, 1, requires_grad=True) * PI2
    t_c = torch.rand(n_coll, 1, requires_grad=True)

    # PDE residual
    loss_pde, components = pde_residual_loss(best_head_u, best_head_v, x_c, y_c, t_c, nu)

    # IC penalty
    x_ic = torch.rand(n_ic, 1, requires_grad=True) * PI2
    y_ic = torch.rand(n_ic, 1, requires_grad=True) * PI2
    u_ic_target = torch.sin(x_ic.detach()) * torch.cos(y_ic.detach())
    v_ic_target = -torch.cos(x_ic.detach()) * torch.sin(y_ic.detach())
    loss_ic_val = ic_loss(best_head_u, best_head_v, x_ic, y_ic, u_ic_target, v_ic_target)

    loss = loss_pde + 10.0 * loss_ic_val
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(all_params, 1.0)
    optimizer.step()

    if (step + 1) % 500 == 0:
        print(f"    step {step+1}: PDE={loss_pde.item():.8f} "
              f"(cont={components['continuity']:.8f}, vort={components['vorticity']:.8f}), "
              f"IC={loss_ic_val.item():.8f}")

# ================================================================
# Phase 3: Normalize, snap, and verify
# ================================================================
print(f"\n  Phase 3: Normalize + snap")
print("-" * 70)

# Print raw parameters
print("\n  Raw parameters:")
for label, head in [("u", best_head_u), ("v", best_head_v)]:
    for t in head.terms:
        print(f"    {label}: coeff={t.coeff.item():.6f}")
        for f in t.factors:
            print(f"      {f.func_name}[{f.input_dim}]: a={f.a.item():.6f}, b={f.b.item():.6f}")

# Normalize and snap
head_u_snap = copy.deepcopy(best_head_u)
head_v_snap = copy.deepcopy(best_head_v)
for head in [head_u_snap, head_v_snap]:
    for t in head.terms:
        if isinstance(t, SeparableTerm):
            t.normalize()

head_u_snap.snap_coefficients(tolerance=0.05)
head_v_snap.snap_coefficients(tolerance=0.05)

expr_u = head_u_snap.to_symbolic(input_names=["x", "y", "t"])
expr_v = head_v_snap.to_symbolic(input_names=["x", "y", "t"])

print(f"\n  DISCOVERED SOLUTION (from PDE residual only):")
print(f"    u(x,y,t) = {expr_u.string}")
print(f"    v(x,y,t) = {expr_v.string}")
print(f"\n  Known Taylor-Green solution:")
print(f"    u = sin(x)cos(y)exp(-0.2t)")
print(f"    v = -cos(x)sin(y)exp(-0.2t)")

# Verify against known solution
with torch.no_grad():
    n_test = 5000
    xt = torch.rand(n_test, 1) * PI2
    yt = torch.rand(n_test, 1) * PI2
    tt = torch.rand(n_test, 1)
    test_in = torch.cat([xt, yt, tt], dim=1)

    u_true = torch.sin(xt) * torch.cos(yt) * torch.exp(-2 * nu * tt)
    v_true = -torch.cos(xt) * torch.sin(yt) * torch.exp(-2 * nu * tt)

    u_err = (head_u_snap(test_in) - u_true).abs().mean().item()
    v_err = (head_v_snap(test_in) - v_true).abs().mean().item()

    print(f"\n  Verification against known solution:")
    print(f"    u MAE: {u_err:.8f}")
    print(f"    v MAE: {v_err:.8f}")

print(f"\n{'=' * 70}")
print("  Solution discovered from PDE + initial conditions alone.")
print("  No solution data was used during training.")
print(f"{'=' * 70}")
