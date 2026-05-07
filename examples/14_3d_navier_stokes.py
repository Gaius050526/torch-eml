"""3D Navier-Stokes: search for new exact solutions.

Mathematical framework:
    For u = u0(x,y,z) * exp(-nu*k^2*t) to be an exact NS solution, three
    conditions must hold:
        1. div(u0) = 0                          (incompressible)
        2. Laplacian(u0) = -k^2 * u0            (eigenfunction)
        3. (u0.grad)omega0 = (omega0.grad)u0     (gradient condition)

    Condition 3 means the nonlinear vortex stretching equals advection,
    so the nonlinear term is a pure gradient (absorbed into pressure).

    Phase 1 screens ICs by testing condition 3 computationally.
    Phase 2 verifies candidates with the full PDE-residual solver.
"""

import torch
import torch.nn as nn
import copy
import math
from torch_eml.compose import ComposeHead, SeparableTerm

torch.manual_seed(42)

PI = math.pi
PI2 = 2 * PI
nu = 0.1

print("=" * 70)
print("3D NAVIER-STOKES — Search for New Exact Solutions")
print("  Domain: [0, 2pi]^3, periodic BCs")
print(f"  Viscosity: nu = {nu}")
print("=" * 70)

# ================================================================
# Phase 1: Screen ICs via gradient condition
# ================================================================
# For each IC, compute S = (u0.grad)omega0 - (omega0.grad)u0
# If |S| ~ 0, the IC admits an exact exponentially-decaying solution.

print("\n  Phase 1: Screen 3D initial conditions")
print("  Condition: (u0.grad)w0 = (w0.grad)u0  (gradient condition)")
print("-" * 70)


def screen_ic(u_func, v_func, w_func, n=5000):
    """Test gradient condition at n random spatial points.

    Returns (max|S|, mean|S|, max|div|).
    """
    x = (torch.rand(n, 1) * PI2).requires_grad_(True)
    y = (torch.rand(n, 1) * PI2).requires_grad_(True)
    z = (torch.rand(n, 1) * PI2).requires_grad_(True)

    u = u_func(x, y, z)
    v = v_func(x, y, z)
    w = w_func(x, y, z)

    def grad(f, var):
        return torch.autograd.grad(f, var, grad_outputs=torch.ones_like(f),
                                   create_graph=True)[0]

    # Velocity gradients
    u_x, u_y, u_z = grad(u, x), grad(u, y), grad(u, z)
    v_x, v_y, v_z = grad(v, x), grad(v, y), grad(v, z)
    w_x, w_y, w_z = grad(w, x), grad(w, y), grad(w, z)

    # Continuity
    div = u_x + v_y + w_z

    # Vorticity
    om_x = w_y - v_z
    om_y = u_z - w_x
    om_z = v_x - u_y

    # Vorticity gradients
    om_x_x, om_x_y, om_x_z = grad(om_x, x), grad(om_x, y), grad(om_x, z)
    om_y_x, om_y_y, om_y_z = grad(om_y, x), grad(om_y, y), grad(om_y, z)
    om_z_x, om_z_y, om_z_z = grad(om_z, x), grad(om_z, y), grad(om_z, z)

    # S = (u.grad)omega - (omega.grad)u
    S_x = (u * om_x_x + v * om_x_y + w * om_x_z
           - (om_x * u_x + om_y * u_y + om_z * u_z))
    S_y = (u * om_y_x + v * om_y_y + w * om_y_z
           - (om_x * v_x + om_y * v_y + om_z * v_z))
    S_z = (u * om_z_x + v * om_z_y + w * om_z_z
           - (om_x * w_x + om_y * w_y + om_z * w_z))

    S_mag = (S_x ** 2 + S_y ** 2 + S_z ** 2).sqrt()

    return S_mag.max().item(), S_mag.mean().item(), div.abs().max().item()


# Define IC families
# Family A: u = A*sin(x)cos(y)cos(z), v = B*cos(x)sin(y)cos(z),
#           w = C*cos(x)cos(y)sin(z), with A + B + C = 0
# k^2 = 3 (wavenumber 1 in each direction)
family_A = [
    (1, -1, 0, "A: (1,-1,0) 2D TG embedded [known]"),
    (1, 1, -2, "A: (1,1,-2) fully 3D"),
    (2, -1, -1, "A: (2,-1,-1) asymmetric 3D"),
    (1, 0, -1, "A: (1,0,-1) xz-plane"),
    (3, -1, -2, "A: (3,-1,-2)"),
    (1, -2, 1, "A: (1,-2,1)"),
    (5, -3, -2, "A: (5,-3,-2)"),
    (1, 3, -4, "A: (1,3,-4)"),
]

# Family B: higher wavenumber sin(2x)cos(y)cos(z)
# continuity: 2A + B + C = 0, k^2 = 4+1+1 = 6
family_B = [
    (1, -1, -1, "B: sin(2x) (1,-1,-1)"),
    (1, 1, -3, "B: sin(2x) (1,1,-3)"),
]

# Family C: mixed wavenumber sin(x)cos(y)cos(2z)
# continuity: A + B + 2C = 0, k^2 = 1+1+4 = 6
family_C = [
    (1, 1, -1, "C: cos(2z) (1,1,-1)"),
]

# Family D: higher wavenumber sin(2x)cos(2y)cos(2z)
# continuity: 2A + 2B + 2C = 0 => A+B+C=0, k^2 = 12
family_D_high = [
    (1, 1, -2, "D: k=2 all dirs (1,1,-2)"),
    (1, -1, 0, "D: k=2 all dirs (1,-1,0)"),
]

ics = []

for A, B, C, desc in family_A:
    def make(A=A, B=B, C=C):
        return (lambda x, y, z: A * torch.sin(x) * torch.cos(y) * torch.cos(z),
                lambda x, y, z: B * torch.cos(x) * torch.sin(y) * torch.cos(z),
                lambda x, y, z: C * torch.cos(x) * torch.cos(y) * torch.sin(z))
    uf, vf, wf = make()
    ics.append((uf, vf, wf, desc, 3))

for A, B, C, desc in family_B:
    def make(A=A, B=B, C=C):
        return (lambda x, y, z: A * torch.sin(2*x) * torch.cos(y) * torch.cos(z),
                lambda x, y, z: B * torch.cos(2*x) * torch.sin(y) * torch.cos(z),
                lambda x, y, z: C * torch.cos(2*x) * torch.cos(y) * torch.sin(z))
    uf, vf, wf = make()
    ics.append((uf, vf, wf, desc, 6))

for A, B, C, desc in family_C:
    def make(A=A, B=B, C=C):
        return (lambda x, y, z: A * torch.sin(x) * torch.cos(y) * torch.cos(2*z),
                lambda x, y, z: B * torch.cos(x) * torch.sin(y) * torch.cos(2*z),
                lambda x, y, z: C * torch.cos(x) * torch.cos(y) * torch.sin(2*z))
    uf, vf, wf = make()
    ics.append((uf, vf, wf, desc, 6))

for A, B, C, desc in family_D_high:
    def make(A=A, B=B, C=C):
        return (lambda x, y, z: A * torch.sin(2*x) * torch.cos(2*y) * torch.cos(2*z),
                lambda x, y, z: B * torch.cos(2*x) * torch.sin(2*y) * torch.cos(2*z),
                lambda x, y, z: C * torch.cos(2*x) * torch.cos(2*y) * torch.sin(2*z))
    uf, vf, wf = make()
    ics.append((uf, vf, wf, desc, 12))

# Screen all ICs
candidates = []
for uf, vf, wf, desc, k2 in ics:
    max_s, mean_s, div_check = screen_ic(uf, vf, wf)
    decay = -nu * k2
    if max_s < 1e-3:
        status = "PASS"
        candidates.append((uf, vf, wf, desc, k2, decay))
    else:
        status = "FAIL"
    print(f"    {status}  {desc}")
    print(f"           max|S|={max_s:.6f}  mean|S|={mean_s:.6f}  "
          f"|div|={div_check:.6f}  k2={k2}  decay={decay:.2f}")

print(f"\n  Candidates passing gradient condition: {len(candidates)}")


# ================================================================
# Phase 2: Verify candidates with full PDE-residual solver
# ================================================================
print(f"\n  Phase 2: PDE-residual verification")
print("-" * 70)


def compute_derivs_3d(head, x, y, z, t):
    """Compute value and all needed derivatives for one velocity component."""
    inputs = torch.cat([x, y, z, t], dim=1)
    u = head(inputs)

    def g(f, var):
        return torch.autograd.grad(f, var, grad_outputs=torch.ones_like(f),
                                   create_graph=True)[0]

    u_x, u_y, u_z, u_t = g(u, x), g(u, y), g(u, z), g(u, t)
    u_xx, u_yy, u_zz = g(u_x, x), g(u_y, y), g(u_z, z)

    return u, u_x, u_y, u_z, u_t, u_xx, u_yy, u_zz


def vorticity_residual_3d(head_u, head_v, head_w, x, y, z, t, nu=0.1):
    """3D NS vorticity equation residual + continuity."""
    u, u_x, u_y, u_z, u_t, u_xx, u_yy, u_zz = compute_derivs_3d(head_u, x, y, z, t)
    v, v_x, v_y, v_z, v_t, v_xx, v_yy, v_zz = compute_derivs_3d(head_v, x, y, z, t)
    w, w_x, w_y, w_z, w_t, w_xx, w_yy, w_zz = compute_derivs_3d(head_w, x, y, z, t)

    # Continuity
    cont = u_x + v_y + w_z

    def g(f, var):
        return torch.autograd.grad(f, var, grad_outputs=torch.ones_like(f),
                                   create_graph=True)[0]

    # Vorticity
    om_x = w_y - v_z
    om_y = u_z - w_x
    om_z = v_x - u_y

    # Vorticity time derivatives
    om_x_t, om_y_t, om_z_t = g(om_x, t), g(om_y, t), g(om_z, t)

    # Vorticity spatial derivatives
    om_x_x, om_x_y, om_x_z = g(om_x, x), g(om_x, y), g(om_x, z)
    om_y_x, om_y_y, om_y_z = g(om_y, x), g(om_y, y), g(om_y, z)
    om_z_x, om_z_y, om_z_z = g(om_z, x), g(om_z, y), g(om_z, z)

    # Vorticity second derivatives (for diffusion)
    om_x_xx, om_x_yy, om_x_zz = g(om_x_x, x), g(om_x_y, y), g(om_x_z, z)
    om_y_xx, om_y_yy, om_y_zz = g(om_y_x, x), g(om_y_y, y), g(om_y_z, z)
    om_z_xx, om_z_yy, om_z_zz = g(om_z_x, x), g(om_z_y, y), g(om_z_z, z)

    # Vorticity equation: omega_t + (u.grad)omega - (omega.grad)u = nu * laplacian(omega)
    res_x = (om_x_t
             + u * om_x_x + v * om_x_y + w * om_x_z
             - (om_x * u_x + om_y * u_y + om_z * u_z)
             - nu * (om_x_xx + om_x_yy + om_x_zz))
    res_y = (om_y_t
             + u * om_y_x + v * om_y_y + w * om_y_z
             - (om_x * v_x + om_y * v_y + om_z * v_z)
             - nu * (om_y_xx + om_y_yy + om_y_zz))
    res_z = (om_z_t
             + u * om_z_x + v * om_z_y + w * om_z_z
             - (om_x * w_x + om_y * w_y + om_z * w_z)
             - nu * (om_z_xx + om_z_yy + om_z_zz))

    loss_cont = (cont ** 2).mean()
    loss_vort = (res_x ** 2).mean() + (res_y ** 2).mean() + (res_z ** 2).mean()

    return loss_cont + loss_vort, {
        "continuity": loss_cont.item(),
        "vorticity": loss_vort.item()
    }


def ic_loss_3d(head_u, head_v, head_w, x, y, z, u_tgt, v_tgt, w_tgt):
    """IC penalty at t=0."""
    t0 = torch.zeros_like(x)
    inp = torch.cat([x, y, z, t0], dim=1)
    return (((head_u(inp) - u_tgt) ** 2).mean()
            + ((head_v(inp) - v_tgt) ** 2).mean()
            + ((head_w(inp) - w_tgt) ** 2).mean())


n_coll = 800   # collocation points (smaller for 4D)
n_ic = 400

for uf, vf, wf, desc, k2, decay in candidates:
    print(f"\n  Verifying: {desc}")
    print(f"    Expected decay: exp({decay:.4f}*t)")

    # Determine spatial function types from IC
    # For Family A: sin*cos*cos / cos*sin*cos / cos*cos*sin
    # We add exp as the time factor
    # Map function names based on the IC structure

    # For now, we use (sin,cos,cos,exp) for u, etc.
    # This works for Family A. For other families, we'd need different assignments.
    # We detect the family from desc.

    if desc.startswith("A:"):
        # Extract A, B, C from desc
        parts = desc.split("(")[1].split(")")[0].split(",")
        A_val, B_val, C_val = float(parts[0]), float(parts[1]), float(parts[2])
        funcs_u = ("sin", "cos", "cos", "exp")
        funcs_v = ("cos", "sin", "cos", "exp")
        funcs_w = ("cos", "cos", "sin", "exp")
    elif desc.startswith("B:"):
        parts = desc.split("(")[1].split(")")[0].split(",")
        A_val, B_val, C_val = float(parts[0]), float(parts[1]), float(parts[2])
        funcs_u = ("sin", "cos", "cos", "exp")
        funcs_v = ("cos", "sin", "cos", "exp")
        funcs_w = ("cos", "cos", "sin", "exp")
    elif desc.startswith("C:"):
        parts = desc.split("(")[1].split(")")[0].split(",")
        A_val, B_val, C_val = float(parts[0]), float(parts[1]), float(parts[2])
        funcs_u = ("sin", "cos", "cos", "exp")
        funcs_v = ("cos", "sin", "cos", "exp")
        funcs_w = ("cos", "cos", "sin", "exp")
    elif desc.startswith("D:"):
        parts = desc.split("(")[1].split(")")[0].split(",")
        A_val, B_val, C_val = float(parts[0]), float(parts[1]), float(parts[2])
        funcs_u = ("sin", "cos", "cos", "exp")
        funcs_v = ("cos", "sin", "cos", "exp")
        funcs_w = ("cos", "cos", "sin", "exp")
    else:
        continue

    # Create ComposeHeads for u, v, w
    heads = {}
    for label, funcs, coeff_init in [("u", funcs_u, A_val),
                                      ("v", funcs_v, B_val),
                                      ("w", funcs_w, C_val)]:
        head = ComposeHead(n_inputs=4, primitives=[], repeat=0,
                           products=False, separable=False)
        term = SeparableTerm(list(zip(funcs, (0, 1, 2, 3))), 4)
        head.terms = nn.ModuleList([term])
        # Initialize coefficient near expected value
        with torch.no_grad():
            term.coeff.fill_(1.0)  # will learn the right value
        heads[label] = head

    head_u, head_v, head_w = heads["u"], heads["v"], heads["w"]
    all_params = (list(head_u.parameters())
                  + list(head_v.parameters())
                  + list(head_w.parameters()))
    optimizer = torch.optim.Adam(all_params, lr=0.01)

    # Phase 2a: Train on IC + continuity (1000 steps)
    for step in range(1000):
        x_ic = torch.rand(n_ic, 1, requires_grad=True) * PI2
        y_ic = torch.rand(n_ic, 1, requires_grad=True) * PI2
        z_ic = torch.rand(n_ic, 1, requires_grad=True) * PI2

        u_tgt = uf(x_ic.detach(), y_ic.detach(), z_ic.detach())
        v_tgt = vf(x_ic.detach(), y_ic.detach(), z_ic.detach())
        w_tgt = wf(x_ic.detach(), y_ic.detach(), z_ic.detach())

        loss_ic = ic_loss_3d(head_u, head_v, head_w,
                             x_ic, y_ic, z_ic, u_tgt, v_tgt, w_tgt)

        # Continuity at collocation points
        x_c = torch.rand(n_coll, 1, requires_grad=True) * PI2
        y_c = torch.rand(n_coll, 1, requires_grad=True) * PI2
        z_c = torch.rand(n_coll, 1, requires_grad=True) * PI2
        t_c = torch.rand(n_coll, 1, requires_grad=True)
        inp_c = torch.cat([x_c, y_c, z_c, t_c], dim=1)

        uc = head_u(inp_c)
        vc = head_v(inp_c)
        wc = head_w(inp_c)
        uc_x = torch.autograd.grad(uc, x_c, grad_outputs=torch.ones_like(uc),
                                    create_graph=True)[0]
        vc_y = torch.autograd.grad(vc, y_c, grad_outputs=torch.ones_like(vc),
                                    create_graph=True)[0]
        wc_z = torch.autograd.grad(wc, z_c, grad_outputs=torch.ones_like(wc),
                                    create_graph=True)[0]
        loss_cont = ((uc_x + vc_y + wc_z) ** 2).mean()

        loss = loss_ic + 0.1 * loss_cont
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(all_params, 1.0)
        optimizer.step()

    print(f"    After IC+cont training: IC={loss_ic.item():.8f}, "
          f"cont={loss_cont.item():.8f}")

    # Phase 2b: Full vorticity equation (2000 steps)
    optimizer = torch.optim.Adam(all_params, lr=0.005)

    for step in range(2000):
        x_c = torch.rand(n_coll, 1, requires_grad=True) * PI2
        y_c = torch.rand(n_coll, 1, requires_grad=True) * PI2
        z_c = torch.rand(n_coll, 1, requires_grad=True) * PI2
        t_c = torch.rand(n_coll, 1, requires_grad=True)

        loss_pde, comps = vorticity_residual_3d(
            head_u, head_v, head_w, x_c, y_c, z_c, t_c, nu)

        x_ic = torch.rand(n_ic, 1, requires_grad=True) * PI2
        y_ic = torch.rand(n_ic, 1, requires_grad=True) * PI2
        z_ic = torch.rand(n_ic, 1, requires_grad=True) * PI2
        u_tgt = uf(x_ic.detach(), y_ic.detach(), z_ic.detach())
        v_tgt = vf(x_ic.detach(), y_ic.detach(), z_ic.detach())
        w_tgt = wf(x_ic.detach(), y_ic.detach(), z_ic.detach())
        loss_ic = ic_loss_3d(head_u, head_v, head_w,
                             x_ic, y_ic, z_ic, u_tgt, v_tgt, w_tgt)

        loss = loss_pde + 10.0 * loss_ic
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(all_params, 1.0)
        optimizer.step()

        if (step + 1) % 500 == 0:
            print(f"    step {step+1}: PDE={loss_pde.item():.8f} "
                  f"(cont={comps['continuity']:.8f}, "
                  f"vort={comps['vorticity']:.8f}), "
                  f"IC={loss_ic.item():.8f}")

    # Phase 2c: Extract solution
    print(f"\n    Raw parameters:")
    for label, head in [("u", head_u), ("v", head_v), ("w", head_w)]:
        for t in head.terms:
            print(f"      {label}: coeff={t.coeff.item():.6f}")
            for f in t.factors:
                print(f"        {f.func_name}[{f.input_dim}]: "
                      f"a={f.a.item():.6f}, b={f.b.item():.6f}")

    # Normalize and snap
    head_u_snap = copy.deepcopy(head_u)
    head_v_snap = copy.deepcopy(head_v)
    head_w_snap = copy.deepcopy(head_w)
    for head in [head_u_snap, head_v_snap, head_w_snap]:
        for t in head.terms:
            if isinstance(t, SeparableTerm):
                t.normalize()

    head_u_snap.snap_coefficients(tolerance=0.05)
    head_v_snap.snap_coefficients(tolerance=0.05)
    head_w_snap.snap_coefficients(tolerance=0.05)

    names = ["x", "y", "z", "t"]
    expr_u = head_u_snap.to_symbolic(input_names=names)
    expr_v = head_v_snap.to_symbolic(input_names=names)
    expr_w = head_w_snap.to_symbolic(input_names=names)

    print(f"\n    DISCOVERED SOLUTION:")
    print(f"      u(x,y,z,t) = {expr_u.string}")
    print(f"      v(x,y,z,t) = {expr_v.string}")
    print(f"      w(x,y,z,t) = {expr_w.string}")

    # Check if this is known or new
    is_known = "2D TG" in desc or "known" in desc.lower()
    if is_known:
        print(f"\n    STATUS: Known solution (baseline verification)")
    else:
        print(f"\n    STATUS: *** POTENTIALLY NEW 3D NS SOLUTION ***")
        print(f"    Verify: substitute back into NS analytically")

    # Verify: evaluate at test points vs IC * exp(decay*t)
    with torch.no_grad():
        n_test = 5000
        xt = torch.rand(n_test, 1) * PI2
        yt = torch.rand(n_test, 1) * PI2
        zt = torch.rand(n_test, 1) * PI2
        tt = torch.rand(n_test, 1)
        test_in = torch.cat([xt, yt, zt, tt], dim=1)

        exp_decay = torch.exp(decay * tt)
        u_expected = uf(xt, yt, zt) * exp_decay
        v_expected = vf(xt, yt, zt) * exp_decay
        w_expected = wf(xt, yt, zt) * exp_decay

        u_err = (head_u_snap(test_in) - u_expected).abs().mean().item()
        v_err = (head_v_snap(test_in) - v_expected).abs().mean().item()
        w_err = (head_w_snap(test_in) - w_expected).abs().mean().item()

        print(f"\n    Verification vs u0*exp({decay:.2f}*t):")
        print(f"      u MAE: {u_err:.8f}")
        print(f"      v MAE: {v_err:.8f}")
        print(f"      w MAE: {w_err:.8f}")

print(f"\n{'=' * 70}")
print("  3D Navier-Stokes search complete.")
print(f"{'=' * 70}")
