"""3D Navier-Stokes: multi-term search for exact solutions.

Phase 1 (example 14) showed that NO single-product 3D IC admits a simple
exponentially-decaying solution — vortex stretching prevents it.

This script tries a different approach:
    1. Multi-term ansatz: u = sum_k c_k * f_k(x,y,z,t)
    2. The nonlinear NS interactions can generate new spatial modes
    3. If the solution closes within a finite set of modes, we can find it

We also test Beltrami flows (omega = lambda*u), which ARE known to work.

Strategy: try the PDE-residual solver directly with multi-term ComposeHeads.
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
print("3D NAVIER-STOKES — Multi-Term Search")
print("=" * 70)


def compute_derivs(head, x, y, z, t):
    """Compute u and first/second spatial + time derivatives."""
    inputs = torch.cat([x, y, z, t], dim=1)
    u = head(inputs)
    ones = torch.ones_like(u)

    def g(f, var):
        return torch.autograd.grad(f, var, grad_outputs=torch.ones_like(f),
                                   create_graph=True)[0]

    u_x, u_y, u_z, u_t = g(u, x), g(u, y), g(u, z), g(u, t)
    u_xx, u_yy, u_zz = g(u_x, x), g(u_y, y), g(u_z, z)
    return u, u_x, u_y, u_z, u_t, u_xx, u_yy, u_zz


def ns_vorticity_loss(head_u, head_v, head_w, x, y, z, t, nu=0.1):
    """3D vorticity equation residual."""
    u, u_x, u_y, u_z, u_t, u_xx, u_yy, u_zz = compute_derivs(head_u, x, y, z, t)
    v, v_x, v_y, v_z, v_t, v_xx, v_yy, v_zz = compute_derivs(head_v, x, y, z, t)
    w, w_x, w_y, w_z, w_t, w_xx, w_yy, w_zz = compute_derivs(head_w, x, y, z, t)

    cont = u_x + v_y + w_z

    def g(f, var):
        return torch.autograd.grad(f, var, grad_outputs=torch.ones_like(f),
                                   create_graph=True)[0]

    om_x = w_y - v_z
    om_y = u_z - w_x
    om_z = v_x - u_y

    om_x_t = g(om_x, t)
    om_y_t = g(om_y, t)
    om_z_t = g(om_z, t)

    om_x_x, om_x_y, om_x_z = g(om_x, x), g(om_x, y), g(om_x, z)
    om_y_x, om_y_y, om_y_z = g(om_y, x), g(om_y, y), g(om_y, z)
    om_z_x, om_z_y, om_z_z = g(om_z, x), g(om_z, y), g(om_z, z)

    om_x_xx, om_x_yy, om_x_zz = g(om_x_x, x), g(om_x_y, y), g(om_x_z, z)
    om_y_xx, om_y_yy, om_y_zz = g(om_y_x, x), g(om_y_y, y), g(om_y_z, z)
    om_z_xx, om_z_yy, om_z_zz = g(om_z_x, x), g(om_z_y, y), g(om_z_z, z)

    # omega_t + (u.grad)omega - (omega.grad)u = nu * laplacian(omega)
    rx = (om_x_t + u*om_x_x + v*om_x_y + w*om_x_z
          - (om_x*u_x + om_y*u_y + om_z*u_z)
          - nu * (om_x_xx + om_x_yy + om_x_zz))
    ry = (om_y_t + u*om_y_x + v*om_y_y + w*om_y_z
          - (om_x*v_x + om_y*v_y + om_z*v_z)
          - nu * (om_y_xx + om_y_yy + om_y_zz))
    rz = (om_z_t + u*om_z_x + v*om_z_y + w*om_z_z
          - (om_x*w_x + om_y*w_y + om_z*w_z)
          - nu * (om_z_xx + om_z_yy + om_z_zz))

    loss_cont = (cont ** 2).mean()
    loss_vort = (rx ** 2).mean() + (ry ** 2).mean() + (rz ** 2).mean()
    return loss_cont + loss_vort, {"continuity": loss_cont.item(),
                                   "vorticity": loss_vort.item()}


def ic_loss_3d(head_u, head_v, head_w, x, y, z, u0, v0, w0):
    t0 = torch.zeros_like(x)
    inp = torch.cat([x, y, z, t0], dim=1)
    return (((head_u(inp) - u0) ** 2).mean()
            + ((head_v(inp) - v0) ** 2).mean()
            + ((head_w(inp) - w0) ** 2).mean())


n_coll = 600
n_ic = 300

# ================================================================
# Experiment 1: Verify Beltrami flow (known, baseline)
# ================================================================
# ABC flow with A=B=C=1: omega = u, solution = u0*exp(-nu*t)
# u0 = (sin(z)+cos(y), sin(x)+cos(z), sin(y)+cos(x))
# This is multi-term: each component is a sum of 2 separable terms

print("\n  Experiment 1: Beltrami ABC flow A=B=C=1 (known solution)")
print("  IC: u=(sin(z)+cos(y)), v=(sin(x)+cos(z)), w=(sin(y)+cos(x))")
print("  Expected: u0 * exp(-0.1*t)")
print("-" * 70)

# For ABC flow, each velocity component needs 2 terms.
# Each term is a 1D function * exp(t), extended to 4D by using trivial factors.
# Problem: SeparableTerm needs factors on ALL dimensions.
# Solution: use id function for unused dims? No — id(x) = x, not constant.
#
# Alternative: use sin(0*x + pi/2) = cos(0) = 1 as a constant factor?
# Actually: sin(0*x + b) = sin(b) = constant. If b = pi/2, that's 1.
# But the optimizer would need to learn a=0, b=pi/2 for the unused dims.
#
# Simpler: just parameterize directly without ComposeHead for this test.

# Direct parameterization for Beltrami verification
class BeltramiHead(nn.Module):
    """u = (A*sin(z) + B*cos(y)) * exp(alpha*t)"""
    def __init__(self, func1, dim1, func2, dim2):
        super().__init__()
        self.c1 = nn.Parameter(torch.tensor(1.0))
        self.c2 = nn.Parameter(torch.tensor(1.0))
        self.alpha = nn.Parameter(torch.tensor(-0.1))
        self.func1 = func1  # 'sin' or 'cos'
        self.dim1 = dim1
        self.func2 = func2
        self.dim2 = dim2

    def forward(self, inputs):
        # inputs: [n, 4] = (x, y, z, t)
        v1 = inputs[:, self.dim1:self.dim1+1]
        v2 = inputs[:, self.dim2:self.dim2+1]
        t = inputs[:, 3:4]
        f1 = torch.sin(v1) if self.func1 == 'sin' else torch.cos(v1)
        f2 = torch.sin(v2) if self.func2 == 'sin' else torch.cos(v2)
        return (self.c1 * f1 + self.c2 * f2) * torch.exp(self.alpha * t)


# ABC with A=B=C=1:
# u = sin(z) + cos(y), v = sin(x) + cos(z), w = sin(y) + cos(x)
head_u = BeltramiHead('sin', 2, 'cos', 1)  # sin(z) + cos(y)
head_v = BeltramiHead('sin', 0, 'cos', 2)  # sin(x) + cos(z)
head_w = BeltramiHead('sin', 1, 'cos', 0)  # sin(y) + cos(x)

all_params = list(head_u.parameters()) + list(head_v.parameters()) + list(head_w.parameters())
optimizer = torch.optim.Adam(all_params, lr=0.01)

def abc_ic(x, y, z):
    return (torch.sin(z) + torch.cos(y),
            torch.sin(x) + torch.cos(z),
            torch.sin(y) + torch.cos(x))

for step in range(2000):
    x_c = torch.rand(n_coll, 1, requires_grad=True) * PI2
    y_c = torch.rand(n_coll, 1, requires_grad=True) * PI2
    z_c = torch.rand(n_coll, 1, requires_grad=True) * PI2
    t_c = torch.rand(n_coll, 1, requires_grad=True)

    loss_pde, comps = ns_vorticity_loss(head_u, head_v, head_w,
                                         x_c, y_c, z_c, t_c, nu)

    x_ic = torch.rand(n_ic, 1, requires_grad=True) * PI2
    y_ic = torch.rand(n_ic, 1, requires_grad=True) * PI2
    z_ic = torch.rand(n_ic, 1, requires_grad=True) * PI2
    u0, v0, w0 = abc_ic(x_ic.detach(), y_ic.detach(), z_ic.detach())
    loss_ic = ic_loss_3d(head_u, head_v, head_w, x_ic, y_ic, z_ic, u0, v0, w0)

    loss = loss_pde + 10.0 * loss_ic
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(all_params, 1.0)
    optimizer.step()

    if (step + 1) % 500 == 0:
        print(f"    step {step+1}: PDE={loss_pde.item():.8f} "
              f"(cont={comps['continuity']:.8f}, vort={comps['vorticity']:.8f}), "
              f"IC={loss_ic.item():.8f}")

print(f"\n    Learned parameters:")
for label, head in [("u", head_u), ("v", head_v), ("w", head_w)]:
    print(f"      {label}: c1={head.c1.item():.6f}, c2={head.c2.item():.6f}, "
          f"alpha={head.alpha.item():.6f}")

print(f"\n    Expected: c1=1.0, c2=1.0, alpha=-0.100 (Beltrami decay)")

# Verify
with torch.no_grad():
    n_test = 5000
    xt = torch.rand(n_test, 1) * PI2
    yt = torch.rand(n_test, 1) * PI2
    zt = torch.rand(n_test, 1) * PI2
    tt = torch.rand(n_test, 1)
    test_in = torch.cat([xt, yt, zt, tt], dim=1)

    decay = torch.exp(-nu * tt)
    u0, v0, w0 = abc_ic(xt, yt, zt)
    u_true = u0 * decay
    v_true = v0 * decay
    w_true = w0 * decay

    u_err = (head_u(test_in) - u_true).abs().mean().item()
    v_err = (head_v(test_in) - v_true).abs().mean().item()
    w_err = (head_w(test_in) - w_true).abs().mean().item()

    print(f"\n    Verification vs known solution u0*exp(-{nu}*t):")
    print(f"      u MAE: {u_err:.8f}")
    print(f"      v MAE: {v_err:.8f}")
    print(f"      w MAE: {w_err:.8f}")


# ================================================================
# Experiment 2: Non-Beltrami 3D flow with multi-term ansatz
# ================================================================
# IC: u = sin(x)cos(y)cos(z), v = cos(x)sin(y)cos(z), w = -2cos(x)cos(y)sin(z)
# This is NOT Beltrami. Phase 1 showed the gradient condition fails.
# Nonlinear interactions generate wavenumber-2 modes.
#
# Ansatz: each component = k=1 mode * exp(a1*t) + k=2 mode * exp(a2*t)
# The k=2 modes come from products of k=1 trig functions:
#   sin(x)cos(x) = (1/2)sin(2x), cos^2(x) = (1+cos(2x))/2, etc.

print(f"\n{'=' * 70}")
print("  Experiment 2: Non-Beltrami 3D flow (1,1,-2)")
print("  IC: u=sin(x)cos(y)cos(z), v=cos(x)sin(y)cos(z), w=-2cos(x)cos(y)sin(z)")
print("  Testing if multi-term ansatz can find exact solution...")
print("-" * 70)


class MultiTermHead3D(nn.Module):
    """Multi-term separable ansatz for 3D NS.

    Each term is c * f(x) * g(y) * h(z) * exp(alpha * t)
    where f, g, h are sin or cos with learnable wavenumber.
    """
    def __init__(self, n_terms=4):
        super().__init__()
        self.n_terms = n_terms
        self.coeffs = nn.Parameter(torch.randn(n_terms) * 0.1)
        self.alphas = nn.Parameter(torch.ones(n_terms) * -0.3)
        # Wavenumbers for x, y, z (integer-ish)
        self.kx = nn.Parameter(torch.ones(n_terms))
        self.ky = nn.Parameter(torch.ones(n_terms))
        self.kz = nn.Parameter(torch.ones(n_terms))
        # Phase: 0 = sin, pi/2 = cos (learnable)
        self.px = nn.Parameter(torch.zeros(n_terms))
        self.py = nn.Parameter(torch.zeros(n_terms))
        self.pz = nn.Parameter(torch.zeros(n_terms))

    def forward(self, inputs):
        x, y, z, t = inputs[:, 0:1], inputs[:, 1:2], inputs[:, 2:3], inputs[:, 3:4]
        result = torch.zeros(x.shape[0], 1)
        for i in range(self.n_terms):
            fx = torch.sin(self.kx[i] * x + self.px[i])
            fy = torch.sin(self.ky[i] * y + self.py[i])
            fz = torch.sin(self.kz[i] * z + self.pz[i])
            ft = torch.exp(self.alphas[i] * t)
            result = result + self.coeffs[i] * fx * fy * fz * ft
        return result


# IC: A=1, B=1, C=-2
def ic_112(x, y, z):
    return (torch.sin(x) * torch.cos(y) * torch.cos(z),
            torch.cos(x) * torch.sin(y) * torch.cos(z),
            -2 * torch.cos(x) * torch.cos(y) * torch.sin(z))

n_terms = 4  # allow up to 4 modes per component
head_u = MultiTermHead3D(n_terms)
head_v = MultiTermHead3D(n_terms)
head_w = MultiTermHead3D(n_terms)

# Initialize first term near the IC
with torch.no_grad():
    # u = sin(x)cos(y)cos(z)*exp(at): sin = sin(x+0), cos = sin(y+pi/2), cos = sin(z+pi/2)
    head_u.coeffs[0] = 1.0
    head_u.kx[0], head_u.ky[0], head_u.kz[0] = 1.0, 1.0, 1.0
    head_u.px[0], head_u.py[0], head_u.pz[0] = 0.0, PI/2, PI/2

    head_v.coeffs[0] = 1.0
    head_v.kx[0], head_v.ky[0], head_v.kz[0] = 1.0, 1.0, 1.0
    head_v.px[0], head_v.py[0], head_v.pz[0] = PI/2, 0.0, PI/2

    head_w.coeffs[0] = -2.0
    head_w.kx[0], head_w.ky[0], head_w.kz[0] = 1.0, 1.0, 1.0
    head_w.px[0], head_w.py[0], head_w.pz[0] = PI/2, PI/2, 0.0

    # Initialize other terms small
    for h in [head_u, head_v, head_w]:
        for i in range(1, n_terms):
            h.coeffs[i] = 0.01

all_params = list(head_u.parameters()) + list(head_v.parameters()) + list(head_w.parameters())

# Phase 1: IC + continuity (1000 steps)
optimizer = torch.optim.Adam(all_params, lr=0.01)
for step in range(1000):
    x_ic = torch.rand(n_ic, 1, requires_grad=True) * PI2
    y_ic = torch.rand(n_ic, 1, requires_grad=True) * PI2
    z_ic = torch.rand(n_ic, 1, requires_grad=True) * PI2
    u0, v0, w0 = ic_112(x_ic.detach(), y_ic.detach(), z_ic.detach())
    loss_ic = ic_loss_3d(head_u, head_v, head_w, x_ic, y_ic, z_ic, u0, v0, w0)

    x_c = torch.rand(n_coll, 1, requires_grad=True) * PI2
    y_c = torch.rand(n_coll, 1, requires_grad=True) * PI2
    z_c = torch.rand(n_coll, 1, requires_grad=True) * PI2
    t_c = torch.rand(n_coll, 1, requires_grad=True)
    inp = torch.cat([x_c, y_c, z_c, t_c], dim=1)

    uc = head_u(inp)
    vc = head_v(inp)
    wc = head_w(inp)
    uc_x = torch.autograd.grad(uc, x_c, grad_outputs=torch.ones_like(uc), create_graph=True)[0]
    vc_y = torch.autograd.grad(vc, y_c, grad_outputs=torch.ones_like(vc), create_graph=True)[0]
    wc_z = torch.autograd.grad(wc, z_c, grad_outputs=torch.ones_like(wc), create_graph=True)[0]
    loss_cont = ((uc_x + vc_y + wc_z) ** 2).mean()

    loss = loss_ic + 0.1 * loss_cont
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(all_params, 1.0)
    optimizer.step()

print(f"    After IC+cont: IC={loss_ic.item():.8f}, cont={loss_cont.item():.8f}")

# Phase 2: Full vorticity equation (3000 steps)
optimizer = torch.optim.Adam(all_params, lr=0.005)
for step in range(3000):
    x_c = torch.rand(n_coll, 1, requires_grad=True) * PI2
    y_c = torch.rand(n_coll, 1, requires_grad=True) * PI2
    z_c = torch.rand(n_coll, 1, requires_grad=True) * PI2
    t_c = torch.rand(n_coll, 1, requires_grad=True) * 0.5  # shorter time horizon

    loss_pde, comps = ns_vorticity_loss(head_u, head_v, head_w,
                                         x_c, y_c, z_c, t_c, nu)

    x_ic = torch.rand(n_ic, 1, requires_grad=True) * PI2
    y_ic = torch.rand(n_ic, 1, requires_grad=True) * PI2
    z_ic = torch.rand(n_ic, 1, requires_grad=True) * PI2
    u0, v0, w0 = ic_112(x_ic.detach(), y_ic.detach(), z_ic.detach())
    loss_ic = ic_loss_3d(head_u, head_v, head_w, x_ic, y_ic, z_ic, u0, v0, w0)

    loss = loss_pde + 10.0 * loss_ic
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(all_params, 1.0)
    optimizer.step()

    if (step + 1) % 500 == 0:
        print(f"    step {step+1}: PDE={loss_pde.item():.8f} "
              f"(cont={comps['continuity']:.8f}, vort={comps['vorticity']:.8f}), "
              f"IC={loss_ic.item():.8f}")

# Report learned parameters
print(f"\n    Learned multi-term parameters:")
for label, head in [("u", head_u), ("v", head_v), ("w", head_w)]:
    print(f"    {label}:")
    for i in range(n_terms):
        c = head.coeffs[i].item()
        if abs(c) < 0.001:
            continue  # skip negligible terms
        kx = head.kx[i].item()
        ky = head.ky[i].item()
        kz = head.kz[i].item()
        px = head.px[i].item()
        py = head.py[i].item()
        pz = head.pz[i].item()
        a = head.alphas[i].item()
        print(f"      term {i}: c={c:.4f}, "
              f"k=({kx:.2f},{ky:.2f},{kz:.2f}), "
              f"phase=({px:.2f},{py:.2f},{pz:.2f}), "
              f"alpha={a:.4f}")

final_pde = loss_pde.item()
if final_pde < 1e-4:
    print(f"\n    *** PDE residual {final_pde:.2e} — POTENTIAL EXACT SOLUTION! ***")
elif final_pde < 1e-2:
    print(f"\n    PDE residual {final_pde:.2e} — approximate but not exact")
else:
    print(f"\n    PDE residual {final_pde:.2e} — no exact solution found in this ansatz")
    print(f"    (Expected: 3D vortex stretching prevents simple multi-mode closure)")


# ================================================================
# Experiment 3: Asymmetric Beltrami (non-equal ABC coefficients)
# ================================================================
# ABC with A=1, B=sqrt(2), C=sqrt(3)
# Still Beltrami? omega = curl(u):
# For ABC: u = (A*sin(z)+C*cos(y), B*sin(x)+A*cos(z), C*sin(y)+B*cos(x))
# omega = (C*cos(y)+... let me compute)
# Actually for general ABC: omega = (C*cos(y)-(-A*sin(z)), ...)
# omega_x = d_y(w) - d_z(v) = d_y(C*sin(y)+B*cos(x)) - d_z(B*sin(x)+A*cos(z))
#         = C*cos(y) - (-A*sin(z)) = C*cos(y) + A*sin(z) = u_x component!
# So omega = u for ANY A,B,C! All ABC flows are Beltrami with lambda=1.

print(f"\n{'=' * 70}")
print("  Experiment 3: Asymmetric ABC (A=1, B=sqrt(2), C=sqrt(3))")
print("  Still Beltrami (omega=u for all ABC). Expected: u0*exp(-0.1t)")
print("-" * 70)

A3, B3, C3 = 1.0, math.sqrt(2), math.sqrt(3)

head_u3 = BeltramiHead('sin', 2, 'cos', 1)
head_v3 = BeltramiHead('sin', 0, 'cos', 2)
head_w3 = BeltramiHead('sin', 1, 'cos', 0)

# Initialize near expected
with torch.no_grad():
    head_u3.c1.fill_(A3)
    head_u3.c2.fill_(C3)
    head_v3.c1.fill_(B3)
    head_v3.c2.fill_(A3)
    head_w3.c1.fill_(C3)
    head_w3.c2.fill_(B3)

def abc_ic_asym(x, y, z):
    return (A3 * torch.sin(z) + C3 * torch.cos(y),
            B3 * torch.sin(x) + A3 * torch.cos(z),
            C3 * torch.sin(y) + B3 * torch.cos(x))

all_params3 = list(head_u3.parameters()) + list(head_v3.parameters()) + list(head_w3.parameters())
optimizer = torch.optim.Adam(all_params3, lr=0.01)

for step in range(2000):
    x_c = torch.rand(n_coll, 1, requires_grad=True) * PI2
    y_c = torch.rand(n_coll, 1, requires_grad=True) * PI2
    z_c = torch.rand(n_coll, 1, requires_grad=True) * PI2
    t_c = torch.rand(n_coll, 1, requires_grad=True)

    loss_pde, comps = ns_vorticity_loss(head_u3, head_v3, head_w3,
                                         x_c, y_c, z_c, t_c, nu)

    x_ic = torch.rand(n_ic, 1, requires_grad=True) * PI2
    y_ic = torch.rand(n_ic, 1, requires_grad=True) * PI2
    z_ic = torch.rand(n_ic, 1, requires_grad=True) * PI2
    u0, v0, w0 = abc_ic_asym(x_ic.detach(), y_ic.detach(), z_ic.detach())
    loss_ic = ic_loss_3d(head_u3, head_v3, head_w3, x_ic, y_ic, z_ic, u0, v0, w0)

    loss = loss_pde + 10.0 * loss_ic
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(all_params3, 1.0)
    optimizer.step()

    if (step + 1) % 500 == 0:
        print(f"    step {step+1}: PDE={loss_pde.item():.8f} "
              f"(cont={comps['continuity']:.8f}, vort={comps['vorticity']:.8f}), "
              f"IC={loss_ic.item():.8f}")

print(f"\n    Learned parameters:")
for label, head in [("u", head_u3), ("v", head_v3), ("w", head_w3)]:
    print(f"      {label}: c1={head.c1.item():.6f}, c2={head.c2.item():.6f}, "
          f"alpha={head.alpha.item():.6f}")

print(f"    Expected: alpha=-0.100 for all components")

# Verify
with torch.no_grad():
    xt = torch.rand(5000, 1) * PI2
    yt = torch.rand(5000, 1) * PI2
    zt = torch.rand(5000, 1) * PI2
    tt = torch.rand(5000, 1)
    test_in = torch.cat([xt, yt, zt, tt], dim=1)

    decay = torch.exp(-nu * tt)
    u0, v0, w0 = abc_ic_asym(xt, yt, zt)

    u_err = (head_u3(test_in) - u0 * decay).abs().mean().item()
    v_err = (head_v3(test_in) - v0 * decay).abs().mean().item()
    w_err = (head_w3(test_in) - w0 * decay).abs().mean().item()

    print(f"\n    Verification:")
    print(f"      u MAE: {u_err:.8f}")
    print(f"      v MAE: {v_err:.8f}")
    print(f"      w MAE: {w_err:.8f}")

    if max(u_err, v_err, w_err) < 1e-5:
        print(f"\n    Confirmed: ABC(1, sqrt(2), sqrt(3)) * exp(-0.1t)")
        print(f"    is an exact 3D Navier-Stokes solution.")

print(f"\n{'=' * 70}")
print("  Summary:")
print("    - Single-product 3D ICs: no exact exponential solutions (vortex stretching)")
print("    - Beltrami ABC flows: exact solutions confirmed (any A,B,C)")
print("    - Non-Beltrami multi-term: results above")
print(f"{'=' * 70}")
