"""Physics-Informed EML: Discover closed-form solutions to Navier-Stokes.

Instead of fitting data, we train EML trees to satisfy the Navier-Stokes
equations directly. The loss is the PDE residual — if it hits zero, the
symbolic expression IS a solution.

We validate on known exact solutions, then push further.

Incompressible Navier-Stokes:
    du/dt + (u . nabla)u = -nabla(p)/rho + nu * laplacian(u)
    div(u) = 0
"""

import logging
import torch
import torch.nn as nn

from torch_eml import EMLHead, save_html

logging.basicConfig(level=logging.INFO)
torch.manual_seed(42)


def train_pinn(heads, loss_fn, epochs=3000, lr=0.005):
    """Train one or more EML heads with PDE loss and gradient clipping."""
    params = []
    for h in heads if isinstance(heads, (list, tuple)) else [heads]:
        params.extend(h.parameters())
    optimizer = torch.optim.Adam(params, lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for step in range(epochs):
        loss_dict = loss_fn(step)
        total = sum(loss_dict.values())

        optimizer.zero_grad()
        total.backward()
        nn.utils.clip_grad_norm_(params, max_norm=1.0)
        optimizer.step()
        scheduler.step()

        if (step + 1) % 500 == 0:
            parts = ", ".join(f"{k}={v.item():.8f}" for k, v in loss_dict.items())
            print(f"  step {step+1}: {parts}")


# ============================================================
# 1. COUETTE FLOW
#    Two parallel plates, top moving at velocity U.
#    Known solution: u(y) = U * y/H (linear profile)
#    Steady 1D: d²u/dy² = 0, u(0)=0, u(H)=U
# ============================================================
print("=" * 60)
print("1. COUETTE FLOW: u(y) = U * y / H")
print("   PDE: d²u/dy² = 0,  BC: u(0)=0, u(1)=1")
print("=" * 60)

y_col = torch.linspace(0, 1, 200, requires_grad=True).unsqueeze(1)
head_c = EMLHead(n_inputs=1, depth=2)


def couette_loss(step):
    u = head_c(y_col)
    du = torch.autograd.grad(u, y_col, torch.ones_like(u), create_graph=True)[0]
    d2u = torch.autograd.grad(du, y_col, torch.ones_like(du), create_graph=True)[0]
    pde = (d2u ** 2).mean()
    bc = (head_c(torch.tensor([[0.0]])) ** 2).squeeze() + \
         ((head_c(torch.tensor([[1.0]])) - 1.0) ** 2).squeeze()
    return {"pde": pde, "bc": 20.0 * bc}


train_pinn(head_c, couette_loss, epochs=3000, lr=0.005)

y_val = torch.linspace(0, 1, 200).unsqueeze(1)
head_c.prune(threshold=0.1, calibration_data=y_val)
expr = head_c.snap(tolerance=0.15, validation_data=(y_val, y_val))

print(f"\n  Discovered: u(y) = {expr.string}")
print(f"  Expected:   u(y) = y")

with torch.no_grad():
    error = (head_c(y_val) - y_val).abs().max().item()
    print(f"  Max error: {error:.6f}")

save_html(head_c, "couette_tree.html",
          title="Couette Flow", equation=expr.string)


# ============================================================
# 2. POISEUILLE FLOW (pressure-driven channel flow)
#    Known solution: u(y) = G/2 * y * (H - y)
#    PDE: d²u/dy² = -G, u(0)=0, u(H)=0
# ============================================================
print(f"\n{'=' * 60}")
print("2. POISEUILLE FLOW: u(y) = y * (1 - y)")
print("   PDE: d²u/dy² = -2,  BC: u(0)=0, u(1)=0")
print("=" * 60)

G = 2.0
y_col2 = torch.linspace(0, 1, 200, requires_grad=True).unsqueeze(1)
head_p = EMLHead(n_inputs=1, depth=3)


def poiseuille_loss(step):
    u = head_p(y_col2)
    du = torch.autograd.grad(u, y_col2, torch.ones_like(u), create_graph=True)[0]
    d2u = torch.autograd.grad(du, y_col2, torch.ones_like(du), create_graph=True)[0]
    pde = ((d2u + G) ** 2).mean()
    bc = (head_p(torch.tensor([[0.0]])) ** 2).squeeze() + \
         (head_p(torch.tensor([[1.0]])) ** 2).squeeze()
    return {"pde": pde, "bc": 20.0 * bc}


train_pinn(head_p, poiseuille_loss, epochs=4000, lr=0.005)

y_val = torch.linspace(0, 1, 200).unsqueeze(1)
u_exact = y_val * (1 - y_val)
head_p.prune(threshold=0.1, calibration_data=y_val)
expr = head_p.snap(tolerance=0.15, validation_data=(y_val, u_exact))

print(f"\n  Discovered: u(y) = {expr.string}")
print(f"  Expected:   u(y) = y*(1-y)")

with torch.no_grad():
    error = (head_p(y_val) - u_exact).abs().max().item()
    print(f"  Max error: {error:.6f}")

save_html(head_p, "poiseuille_tree.html",
          title="Poiseuille Flow", equation=expr.string)


# ============================================================
# 3. TAYLOR-GREEN VORTEX (2D, decaying)
#    u(x,y,t) =  sin(x)*cos(y)*exp(-2νt)
#    v(x,y,t) = -cos(x)*sin(y)*exp(-2νt)
#
#    Known exact solution to 2D incompressible NS.
# ============================================================
print(f"\n{'=' * 60}")
print("3. TAYLOR-GREEN VORTEX (2D decaying)")
print("   u = sin(x)cos(y)exp(-2vt)")
print("=" * 60)

nu = 0.1

n_pts = 400
x_c = (torch.rand(n_pts, 1) * 2 * 3.14159).requires_grad_(True)
y_c = (torch.rand(n_pts, 1) * 2 * 3.14159).requires_grad_(True)
t_c = (torch.rand(n_pts, 1) * 1.0).requires_grad_(True)

head_u = EMLHead(n_inputs=3, depth=3)
head_v = EMLHead(n_inputs=3, depth=3)


def taylor_green_loss(step):
    xyt = torch.cat([x_c, y_c, t_c], dim=1)
    u = head_u(xyt)
    v = head_v(xyt)

    u_x = torch.autograd.grad(u, x_c, torch.ones_like(u), create_graph=True)[0]
    u_y = torch.autograd.grad(u, y_c, torch.ones_like(u), create_graph=True)[0]
    u_t = torch.autograd.grad(u, t_c, torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x_c, torch.ones_like(u_x), create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y_c, torch.ones_like(u_y), create_graph=True)[0]

    v_x = torch.autograd.grad(v, x_c, torch.ones_like(v), create_graph=True)[0]
    v_y = torch.autograd.grad(v, y_c, torch.ones_like(v), create_graph=True)[0]
    v_t = torch.autograd.grad(v, t_c, torch.ones_like(v), create_graph=True)[0]
    v_xx = torch.autograd.grad(v_x, x_c, torch.ones_like(v_x), create_graph=True)[0]
    v_yy = torch.autograd.grad(v_y, y_c, torch.ones_like(v_y), create_graph=True)[0]

    # Continuity
    cont = (u_x + v_y) ** 2

    # Momentum (pressure-free form using vorticity approach)
    mom_x = (u_t + u * u_x + v * u_y - nu * (u_xx + u_yy)) ** 2
    mom_y = (v_t + u * v_x + v * v_y - nu * (v_xx + v_yy)) ** 2

    pde = cont.mean() + mom_x.mean() + mom_y.mean()

    # Initial condition
    n_ic = 200
    xi = torch.rand(n_ic, 1) * 2 * 3.14159
    yi = torch.rand(n_ic, 1) * 2 * 3.14159
    ti = torch.zeros(n_ic, 1)
    ic_in = torch.cat([xi, yi, ti], dim=1)

    ic = ((head_u(ic_in) - torch.sin(xi) * torch.cos(yi)) ** 2).mean() + \
         ((head_v(ic_in) + torch.cos(xi) * torch.sin(yi)) ** 2).mean()

    return {"pde": pde, "ic": 10.0 * ic}


train_pinn([head_u, head_v], taylor_green_loss, epochs=5000, lr=0.003)

# Extract
xyt_val = torch.cat([x_c.detach(), y_c.detach(), t_c.detach()], dim=1)
u_exact = (torch.sin(x_c) * torch.cos(y_c) * torch.exp(-2 * nu * t_c)).detach()
v_exact = (-torch.cos(x_c) * torch.sin(y_c) * torch.exp(-2 * nu * t_c)).detach()

head_u.prune(threshold=0.1, calibration_data=xyt_val)
head_v.prune(threshold=0.1, calibration_data=xyt_val)

expr_u = head_u.snap(tolerance=0.15, validation_data=(xyt_val, u_exact))
expr_v = head_v.snap(tolerance=0.15, validation_data=(xyt_val, v_exact))

print(f"\n  u(x,y,t) = {expr_u.string}")
print(f"  v(x,y,t) = {expr_v.string}")
print(f"\n  Expected u = sin(x)*cos(y)*exp(-0.2t)")
print(f"  Expected v = -cos(x)*sin(y)*exp(-0.2t)")

with torch.no_grad():
    xt = torch.rand(200, 1) * 2 * 3.14159
    yt = torch.rand(200, 1) * 2 * 3.14159
    tt = torch.rand(200, 1)
    test_in = torch.cat([xt, yt, tt], dim=1)

    u_err = (head_u(test_in) - torch.sin(xt) * torch.cos(yt) * torch.exp(-2 * nu * tt)).abs().mean().item()
    v_err = (head_v(test_in) + torch.cos(xt) * torch.sin(yt) * torch.exp(-2 * nu * tt)).abs().mean().item()
    print(f"\n  Mean |u error|: {u_err:.6f}")
    print(f"  Mean |v error|: {v_err:.6f}")

save_html(head_u, "taylor_green_u.html",
          title="Taylor-Green u(x,y,t)", equation=expr_u.string)
save_html(head_v, "taylor_green_v.html",
          title="Taylor-Green v(x,y,t)", equation=expr_v.string)

print(f"\n{'=' * 60}")
print("STATUS:")
print("  These are KNOWN solutions — validation that the method works.")
print("  Next: train on regimes with NO known closed-form solution.")
print("  If PDE residual -> 0 and the equation is new -> discovery.")
print("=" * 60)
