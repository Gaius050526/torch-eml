"""Physics-Informed EML: Discover closed-form solutions to Navier-Stokes.

Uses curriculum learning:
  Phase 1: Data-driven — learn sin/cos structure from known solution samples
  Phase 2: Mixed — data loss + gradually increasing PDE residual
  Phase 3: PDE-dominant — L-BFGS refinement with data as regularizer
"""

import logging
import torch
import torch.nn as nn

from torch_eml import EMLHead, save_html

logging.basicConfig(level=logging.INFO)
torch.manual_seed(42)


# ============================================================
# 1. COUETTE FLOW: u(y) = y
# ============================================================
print("=" * 60)
print("1. COUETTE FLOW: u(y) = y")
print("=" * 60)

y_col = torch.linspace(0, 1, 200, requires_grad=True).unsqueeze(1)
head_c = EMLHead(n_inputs=1, depth=2)
optimizer = torch.optim.Adam(head_c.parameters(), lr=0.005)

for step in range(3000):
    u = head_c(y_col)
    du = torch.autograd.grad(u, y_col, torch.ones_like(u), create_graph=True)[0]
    d2u = torch.autograd.grad(du, y_col, torch.ones_like(du), create_graph=True)[0]
    loss = (d2u ** 2).mean() + 20.0 * (
        head_c(torch.tensor([[0.0]])).squeeze() ** 2 +
        (head_c(torch.tensor([[1.0]])).squeeze() - 1.0) ** 2
    )
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(head_c.parameters(), 1.0)
    optimizer.step()

y_val = torch.linspace(0, 1, 200).unsqueeze(1)
head_c.prune(threshold=0.1, calibration_data=y_val)
expr = head_c.snap(tolerance=0.15, validation_data=(y_val, y_val))
with torch.no_grad():
    error = (head_c(y_val) - y_val).abs().max().item()
print(f"  Discovered: u(y) = {expr.string}")
print(f"  Max error: {error:.6f}")


# ============================================================
# 2. POISEUILLE FLOW: u(y) = y*(1-y)
# ============================================================
print(f"\n{'=' * 60}")
print("2. POISEUILLE FLOW: u(y) = y*(1-y)")
print("=" * 60)

y_col2 = torch.linspace(0, 1, 200, requires_grad=True).unsqueeze(1)
head_p = EMLHead(n_inputs=1, depth=3)
optimizer = torch.optim.Adam(head_p.parameters(), lr=0.005)

for step in range(4000):
    u = head_p(y_col2)
    du = torch.autograd.grad(u, y_col2, torch.ones_like(u), create_graph=True)[0]
    d2u = torch.autograd.grad(du, y_col2, torch.ones_like(du), create_graph=True)[0]
    loss = ((d2u + 2.0) ** 2).mean() + 20.0 * (
        head_p(torch.tensor([[0.0]])).squeeze() ** 2 +
        head_p(torch.tensor([[1.0]])).squeeze() ** 2
    )
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(head_p.parameters(), 1.0)
    optimizer.step()

y_val = torch.linspace(0, 1, 200).unsqueeze(1)
u_exact = y_val * (1 - y_val)
head_p.prune(threshold=0.1, calibration_data=y_val)
expr = head_p.snap(tolerance=0.15, validation_data=(y_val, u_exact))
with torch.no_grad():
    error = (head_p(y_val) - u_exact).abs().max().item()
print(f"  Discovered: u(y) = {expr.string}")
print(f"  Max error: {error:.6f}")


# ============================================================
# 3. TAYLOR-GREEN VORTEX — CURRICULUM LEARNING
# ============================================================
print(f"\n{'=' * 60}")
print("3. TAYLOR-GREEN VORTEX")
print("=" * 60)

nu = 0.1
PI2 = 2 * 3.14159265

head_u = EMLHead(n_inputs=3, depth=4)
head_v = EMLHead(n_inputs=3, depth=4)
params = list(head_u.parameters()) + list(head_v.parameters())

# Training data from known solution
n_data = 1000
x_d = torch.rand(n_data, 1) * PI2
y_d = torch.rand(n_data, 1) * PI2
t_d = torch.rand(n_data, 1) * 1.0
data_in = torch.cat([x_d, y_d, t_d], dim=1)
u_target = torch.sin(x_d) * torch.cos(y_d) * torch.exp(-2 * nu * t_d)
v_target = -torch.cos(x_d) * torch.sin(y_d) * torch.exp(-2 * nu * t_d)

# Collocation points for PDE
x_c = (torch.rand(500, 1) * PI2).requires_grad_(True)
y_c = (torch.rand(500, 1) * PI2).requires_grad_(True)
t_c = (torch.rand(500, 1) * 1.0).requires_grad_(True)


def compute_pde_residual():
    """Compute NS residual at collocation points."""
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

    cont = (u_x + v_y) ** 2
    mom_x = (u_t + u * u_x + v * u_y - nu * (u_xx + u_yy)) ** 2
    mom_y = (v_t + u * v_x + v * v_y - nu * (v_xx + v_yy)) ** 2
    return cont.mean() + mom_x.mean() + mom_y.mean()


# --- Phase 1: Pure data fitting (10k steps) ---
print("\n  Phase 1: Data-driven (learning sin*cos structure)...")
optimizer = torch.optim.Adam(params, lr=0.005)

best_data_loss = float("inf")
for step in range(10000):
    u_pred = head_u(data_in)
    v_pred = head_v(data_in)
    loss = nn.functional.mse_loss(u_pred, u_target) + \
           nn.functional.mse_loss(v_pred, v_target)

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(params, 1.0)
    optimizer.step()

    if loss.item() < best_data_loss:
        best_data_loss = loss.item()

    if (step + 1) % 2000 == 0:
        print(f"    step {step+1}: data_loss={loss.item():.6f}")

print(f"    Best data loss: {best_data_loss:.6f}")

# --- Phase 2: Mixed data + PDE (8k steps, PDE ramps up) ---
print("\n  Phase 2: Mixed data + PDE (ramping)...")
optimizer = torch.optim.Adam(params, lr=0.002)

for step in range(8000):
    pde_weight = min(1.0, step / 3000.0)

    u_pred = head_u(data_in)
    v_pred = head_v(data_in)
    data_loss = nn.functional.mse_loss(u_pred, u_target) + \
                nn.functional.mse_loss(v_pred, v_target)

    pde_loss = compute_pde_residual()

    # Guard against NaN
    if not torch.isfinite(pde_loss):
        continue

    loss = data_loss + pde_weight * pde_loss

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(params, 1.0)
    optimizer.step()

    if (step + 1) % 2000 == 0:
        print(f"    step {step+1}: data={data_loss.item():.6f}, "
              f"pde={pde_loss.item():.6f}, w={pde_weight:.2f}")

# --- Phase 3: L-BFGS refinement (data as regularizer) ---
print("\n  Phase 3: L-BFGS refinement...")
lbfgs = torch.optim.LBFGS(params, lr=0.5, max_iter=20, history_size=50,
                           line_search_fn="strong_wolfe")
lbfgs_steps = 0


def lbfgs_closure():
    global lbfgs_steps
    lbfgs.zero_grad()

    u_pred = head_u(data_in)
    v_pred = head_v(data_in)
    data_loss = nn.functional.mse_loss(u_pred, u_target) + \
                nn.functional.mse_loss(v_pred, v_target)

    pde_loss = compute_pde_residual()

    if not torch.isfinite(pde_loss):
        return data_loss

    loss = 0.1 * data_loss + pde_loss
    loss.backward()

    nn.utils.clip_grad_norm_(params, 1.0)
    lbfgs_steps += 1
    return loss


for i in range(50):
    loss = lbfgs.step(lbfgs_closure)
    if (i + 1) % 10 == 0:
        loss_val = loss.item() if torch.is_tensor(loss) else loss
        print(f"    L-BFGS iter {i+1}: loss={loss_val:.6f}")


# --- Extract ---
print("\n  Extracting symbolic solutions...")
xyt_val = torch.cat([x_c.detach(), y_c.detach(), t_c.detach()], dim=1)
u_exact = (torch.sin(x_c) * torch.cos(y_c) * torch.exp(-2 * nu * t_c)).detach()
v_exact = (-torch.cos(x_c) * torch.sin(y_c) * torch.exp(-2 * nu * t_c)).detach()

head_u.prune(threshold=0.05, calibration_data=xyt_val)
head_v.prune(threshold=0.05, calibration_data=xyt_val)

expr_u = head_u.snap(tolerance=0.1, validation_data=(xyt_val, u_exact))
expr_v = head_v.snap(tolerance=0.1, validation_data=(xyt_val, v_exact))

print(f"\n  u(x,y,t) = {expr_u.string}")
print(f"  v(x,y,t) = {expr_v.string}")
print(f"\n  Expected: u = sin(x)*cos(y)*exp(-0.2t)")

with torch.no_grad():
    xt = torch.rand(500, 1) * PI2
    yt = torch.rand(500, 1) * PI2
    tt = torch.rand(500, 1)
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
print("NEXT: Apply to regimes with NO known closed-form solution.")
print("=" * 60)
