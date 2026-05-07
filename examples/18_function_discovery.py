"""Example 18: Function Discovery via Unconstrained EML.

The fundamental question: what if the function that solves 3D Navier-Stokes
hasn't been named yet?

History shows that new function families are discovered when PDEs demand them:
  - Bessel functions (1817): cylindrical wave equation
  - Airy functions (1838): optics
  - Weierstrass ℘ (1860s): elliptic PDEs

All of these are EML-constructible (by universality) at sufficient depth.
The ComposeHead constrains search to NAMED primitives (sin, cos, exp, tanh, ...).
The raw EMLHead searches over ALL compositions of exp and ln — the full
universal space. If it converges on a PDE residual, the resulting tree IS
a closed-form solution, even if the function has no name.

This example explores:
  1. Validation: Can raw EML recover tanh(x) without being told "tanh"?
  2. Validation: Can raw EML recover sech²(x) without being told "sech"?
  3. Exploration: Train raw EML on Allen-Cahn PDE residual directly.
  4. Exploration: Train raw EML on 3D NS PDE residual for a non-Beltrami IC.

If experiment 4 converges, the tree encodes a function that solves 3D NS
but may not correspond to any named function — a genuine discovery.
"""

import math
import torch
import torch.nn as nn

from torch_eml.head import EMLHead
from torch_eml.tree import EMLTree
from torch_eml.node import EMLNode


# ============================================================
# Experiment 1: Can raw EML rediscover tanh(x)?
# ============================================================

def discover_tanh():
    """Train raw EML tree to fit tanh(x) without naming it."""
    print("=" * 60)
    print("Experiment 1: Rediscover tanh(x) from raw EML")
    print("  EML knows only exp(x) and ln(x).")
    print("  Can it learn tanh(x) = (exp(x)-exp(-x))/(exp(x)+exp(-x))?")
    print("=" * 60)

    x = torch.linspace(-3, 3, 500).unsqueeze(1)
    y = torch.tanh(x)

    # Depth 4 = 15 EML nodes, 16 leaves
    # tanh needs ~depth 3-4 in theory (exp, subtraction, division)
    head = EMLHead(n_inputs=1, depth=4)
    optimizer = torch.optim.Adam(head.parameters(), lr=0.005)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5000)

    best_loss = float('inf')
    for step in range(5000):
        pred = head(x)
        loss = ((pred - y) ** 2).mean()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if loss.item() < best_loss:
            best_loss = loss.item()

        if (step + 1) % 1000 == 0:
            print(f"  Step {step+1:5d}: loss = {loss.item():.2e} (best: {best_loss:.2e})")

    # Evaluate
    with torch.no_grad():
        pred = head(x)
        mae = (pred - y).abs().mean().item()
        max_err = (pred - y).abs().max().item()

    print(f"\n  Final MAE: {mae:.2e}")
    print(f"  Final max error: {max_err:.2e}")
    print(f"  Tree has {sum(p.numel() for p in head.parameters())} parameters")
    print(f"  Depth {head.tree.depth}: {len(head.tree.nodes)} EML nodes")

    success = mae < 0.01
    print(f"  Recovery: {'SUCCESS' if success else 'PARTIAL'}")

    # Examine tree structure
    print(f"\n  Tree node weights (looking for exp/ln patterns):")
    for i, node in enumerate(head.tree.nodes[:7]):  # Show first 7
        print(f"    Node {i}: w_l={node.w_left.item():+.4f} b_l={node.bias_left.item():+.4f} "
              f"w_r={node.w_right.item():+.4f} b_r={node.bias_right.item():+.4f}")

    return success


# ============================================================
# Experiment 2: Can raw EML rediscover sech²(x)?
# ============================================================

def discover_sech_squared():
    """Train raw EML tree to fit sech²(x) = 1/cosh²(x)."""
    print("\n" + "=" * 60)
    print("Experiment 2: Rediscover sech²(x) from raw EML")
    print("  KdV soliton profile: u = -2·sech²(x)")
    print("=" * 60)

    x = torch.linspace(-4, 4, 500).unsqueeze(1)
    y = 1.0 / torch.cosh(x) ** 2

    head = EMLHead(n_inputs=1, depth=5)  # More depth for sech²
    optimizer = torch.optim.Adam(head.parameters(), lr=0.003)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=8000)

    best_loss = float('inf')
    for step in range(8000):
        pred = head(x)
        loss = ((pred - y) ** 2).mean()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if loss.item() < best_loss:
            best_loss = loss.item()

        if (step + 1) % 2000 == 0:
            print(f"  Step {step+1:5d}: loss = {loss.item():.2e} (best: {best_loss:.2e})")

    with torch.no_grad():
        pred = head(x)
        mae = (pred - y).abs().mean().item()

    print(f"\n  Final MAE: {mae:.2e}")
    print(f"  Tree has {len(head.tree.nodes)} EML nodes at depth {head.tree.depth}")

    success = mae < 0.01
    print(f"  Recovery: {'SUCCESS' if success else 'PARTIAL'}")
    return success


# ============================================================
# Experiment 3: Allen-Cahn from PDE residual via raw EML
# ============================================================

def allen_cahn_raw_eml():
    """Discover Allen-Cahn solution from PDE residual using raw EML.

    PDE: u_xx + u - u³ = 0
    Exact: u = tanh(x/√2)
    But we don't tell the model about tanh!
    """
    print("\n" + "=" * 60)
    print("Experiment 3: Allen-Cahn PDE-residual-only via raw EML")
    print("  PDE: u_xx + u - u³ = 0")
    print("  BC: u(-5) ≈ -1, u(+5) ≈ +1")
    print("  Can raw EML discover u = tanh(x/√2) from the equation alone?")
    print("=" * 60)

    head = EMLHead(n_inputs=1, depth=5)
    optimizer = torch.optim.Adam(head.parameters(), lr=0.003)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10000)

    best_loss = float('inf')
    for step in range(10000):
        # Collocation points
        x = torch.linspace(-5, 5, 300, requires_grad=True).unsqueeze(1)

        u = head(x)

        # Compute u_xx via autograd
        u_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0]

        # PDE residual: u_xx + u - u³ = 0
        pde_residual = u_xx + u - u ** 3
        loss_pde = (pde_residual ** 2).mean()

        # Boundary conditions: u(-5) ≈ -1, u(5) ≈ 1
        x_bc = torch.tensor([[-5.0], [5.0]])
        u_bc = head(x_bc)
        loss_bc = ((u_bc - torch.tensor([[-1.0], [1.0]])) ** 2).sum()

        # Monotonicity hint: u(0) ≈ 0 (odd symmetry)
        x_origin = torch.tensor([[0.0]])
        loss_sym = head(x_origin) ** 2

        loss = loss_pde + 10.0 * loss_bc + loss_sym
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if loss.item() < best_loss:
            best_loss = loss.item()

        if (step + 1) % 2000 == 0:
            print(f"  Step {step+1:5d}: PDE={loss_pde.item():.2e} "
                  f"BC={loss_bc.item():.2e} total={loss.item():.2e}")

    # Compare to exact solution
    with torch.no_grad():
        x_test = torch.linspace(-4, 4, 500).unsqueeze(1)
        u_pred = head(x_test)
        u_exact = torch.tanh(x_test / math.sqrt(2))
        mae = (u_pred - u_exact).abs().mean().item()
        max_err = (u_pred - u_exact).abs().max().item()

    print(f"\n  MAE vs tanh(x/√2): {mae:.2e}")
    print(f"  Max error: {max_err:.2e}")

    success = mae < 0.05
    print(f"  Discovery: {'SUCCESS' if success else 'PARTIAL'}")

    if success:
        print("  → Raw EML discovered tanh(x/√2) from PDE alone,")
        print("    without knowing tanh exists!")

    return success


# ============================================================
# Experiment 4: 3D NS non-Beltrami — hunting for unnamed functions
# ============================================================

def ns_3d_raw_eml():
    """Attempt to discover a 3D NS solution via unconstrained EML.

    This is the frontier experiment. We know:
    - Non-Beltrami 3D NS has no separable sin/cos/exp solution
    - Vortex stretching prevents mode closure in Branch T
    - But the PDE may admit solutions in UNNAMED function families

    If the EML tree converges, its structure encodes whatever function
    solves the equation — named or not.

    We use a simple non-Beltrami IC: u₀ = (sin(y), 0, 0)
    which has vorticity ω₀ = (0, 0, cos(y)) and is known to develop
    complex 3D structure.
    """
    print("\n" + "=" * 60)
    print("Experiment 4: 3D Navier-Stokes — hunting unnamed functions")
    print("  IC: u₀ = (sin(y), 0, 0)")
    print("  This is non-Beltrami: no sin/cos/exp solution exists.")
    print("  Training raw EML (depth 6) on PDE residual...")
    print("  If it converges, the tree IS the solution.")
    print("=" * 60)

    nu = 0.1

    # Three EML heads for u, v, w velocity components
    # Plus one for pressure p
    head_u = EMLHead(n_inputs=4, depth=5)  # inputs: x, y, z, t
    head_v = EMLHead(n_inputs=4, depth=5)
    head_w = EMLHead(n_inputs=4, depth=5)
    head_p = EMLHead(n_inputs=4, depth=4)

    all_params = (list(head_u.parameters()) + list(head_v.parameters()) +
                  list(head_w.parameters()) + list(head_p.parameters()))
    optimizer = torch.optim.Adam(all_params, lr=0.002)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5000)

    best_loss = float('inf')
    for step in range(5000):
        # Collocation points in [0, 2π]³ × [0, 0.5]
        N = 200
        coords = torch.rand(N, 4, requires_grad=True)
        coords_scaled = coords.clone()
        coords_scaled[:, :3] = coords[:, :3] * 2 * math.pi
        coords_scaled[:, 3] = coords[:, 3] * 0.5

        # Need grad w.r.t. physical coordinates
        x_phys = coords_scaled

        u = head_u(x_phys)
        v = head_v(x_phys)
        w = head_w(x_phys)
        p = head_p(x_phys)

        # Compute gradients
        ones = torch.ones_like(u)

        def grad(f, x):
            return torch.autograd.grad(f, x, ones, create_graph=True)[0]

        grads_u = grad(u, x_phys)
        grads_v = grad(v, x_phys)
        grads_w = grad(w, x_phys)
        grads_p = grad(p, x_phys)

        u_x, u_y, u_z, u_t = grads_u[:, 0:1], grads_u[:, 1:2], grads_u[:, 2:3], grads_u[:, 3:4]
        v_x, v_y, v_z, v_t = grads_v[:, 0:1], grads_v[:, 1:2], grads_v[:, 2:3], grads_v[:, 3:4]
        w_x, w_y, w_z, w_t = grads_w[:, 0:1], grads_w[:, 1:2], grads_w[:, 2:3], grads_w[:, 3:4]
        p_x, p_y, p_z = grads_p[:, 0:1], grads_p[:, 1:2], grads_p[:, 2:3]

        # Second derivatives for Laplacian (would need second autograd pass)
        # For now, approximate Laplacian via finite differences
        eps = 0.01
        def laplacian_fd(head, x_phys):
            """Finite-difference Laplacian in x, y, z."""
            lap = torch.zeros(x_phys.shape[0], 1)
            for d in range(3):
                xp = x_phys.clone().detach()
                xm = x_phys.clone().detach()
                xp[:, d] += eps
                xm[:, d] -= eps
                with torch.no_grad():
                    fp = head(xp)
                    fm = head(xm)
                    f0 = head(x_phys.detach())
                lap += (fp + fm - 2 * f0) / eps ** 2
            return lap

        lap_u = laplacian_fd(head_u, x_phys)
        lap_v = laplacian_fd(head_v, x_phys)
        lap_w = laplacian_fd(head_w, x_phys)

        # NS momentum: u_t + (u·∇)u = -∇p + ν·Δu
        res_u = u_t + u * u_x + v * u_y + w * u_z + p_x - nu * lap_u
        res_v = v_t + u * v_x + v * v_y + w * v_z + p_y - nu * lap_v
        res_w = w_t + u * w_x + v * w_y + w * w_z + p_z - nu * lap_w

        # Continuity: ∇·u = 0
        div = u_x + v_y + w_z

        loss_pde = (res_u ** 2 + res_v ** 2 + res_w ** 2).mean()
        loss_div = (div ** 2).mean()

        # Initial condition: u(x,y,z,0) = (sin(y), 0, 0)
        N_ic = 100
        ic_coords = torch.zeros(N_ic, 4, requires_grad=True)
        ic_coords_val = ic_coords.clone().detach()
        ic_coords_val[:, 0] = torch.rand(N_ic) * 2 * math.pi
        ic_coords_val[:, 1] = torch.rand(N_ic) * 2 * math.pi
        ic_coords_val[:, 2] = torch.rand(N_ic) * 2 * math.pi
        ic_coords_val[:, 3] = 0.0

        with torch.no_grad():
            u_ic = head_u(ic_coords_val)
            v_ic = head_v(ic_coords_val)
            w_ic = head_w(ic_coords_val)
            u_target = torch.sin(ic_coords_val[:, 1:2])

        loss_ic = ((u_ic - u_target) ** 2 + v_ic ** 2 + w_ic ** 2).mean()

        loss = loss_pde + 10 * loss_div + 10 * loss_ic
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(all_params, 1.0)
        optimizer.step()
        scheduler.step()

        if loss.item() < best_loss:
            best_loss = loss.item()

        if (step + 1) % 1000 == 0:
            print(f"  Step {step+1:5d}: PDE={loss_pde.item():.2e} "
                  f"div={loss_div.item():.2e} IC={loss_ic.item():.2e} "
                  f"total={loss.item():.2e}")

    print(f"\n  Best total loss: {best_loss:.2e}")
    print(f"  Total parameters: {sum(p.numel() for p in all_params)}")
    print(f"  Each velocity head: depth {head_u.tree.depth}, "
          f"{len(head_u.tree.nodes)} EML nodes")

    if best_loss < 0.01:
        print("\n  *** CONVERGENCE DETECTED ***")
        print("  The EML tree has found a representation of the solution.")
        print("  This may encode a function with no classical name.")
        print("  Next step: analyze tree structure for recurring motifs.")
        return True
    else:
        print("\n  Did not converge (expected for this hard problem).")
        print("  The solution may require deeper trees or better optimization.")
        print("  This is the frontier — the function may exist but be hard to find.")
        return False


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    results = {}

    results["Rediscover tanh"] = discover_tanh()
    results["Rediscover sech²"] = discover_sech_squared()
    results["Allen-Cahn PDE-only"] = allen_cahn_raw_eml()
    results["3D NS exploration"] = ns_3d_raw_eml()

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for name, ok in results.items():
        status = "CONVERGED" if ok else "DID NOT CONVERGE"
        print(f"  {name}: {status}")

    print("\n" + "=" * 60)
    print("Interpretation")
    print("=" * 60)
    print("""
  The raw EML tree is an unconstrained function discoverer.
  Unlike ComposeHead (which searches over named primitives),
  EMLHead searches over ALL compositions of exp() and ln().

  If experiment 3 converges, it means raw EML independently
  discovered tanh — a function that wasn't in its vocabulary.

  If experiment 4 converges, the tree encodes a function that
  solves 3D NS. That function may have no name in mathematics.
  It would be defined by its EML composition — a new discovery.

  This is EML's deepest promise: not just recovering known
  solutions, but discovering functions that mathematics hasn't
  catalogued yet.
    """)
