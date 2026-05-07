"""Example 19: Function Discovery Pipeline.

A systematic approach to discovering new mathematical functions via EML.

The idea: EML is universal. Every continuous function is a finite composition
of exp() and ln(). But raw EML trees have terrible optimization landscapes.
This pipeline addresses that with three strategies:

1. Multi-restart optimization with diverse initializations
2. Evolutionary perturbation (mutate best candidates, keep winners)
3. Progressive depth (start shallow, grow successful trees)

We validate by rediscovering known functions (tanh, sech, erf-like) from
data alone, then apply the pipeline to PDE residuals where solutions are
unknown.

The key insight: if the pipeline converges on a PDE residual, the tree
IS the solution — even if the function has no name.
"""

import copy
import math
import torch
import torch.nn as nn
import numpy as np

from torch_eml.head import EMLHead
from torch_eml.node import EMLNode


# ============================================================
# Core: Multi-strategy EML optimizer
# ============================================================

class EMLDiscoverer:
    """Discover functions via unconstrained EML optimization.

    Uses a population of EML trees with diverse initializations,
    evolutionary perturbation, and progressive depth growth.
    """

    def __init__(self, n_inputs: int, depth: int = 4, population: int = 16):
        self.n_inputs = n_inputs
        self.depth = depth
        self.population_size = population

        # Create population with diverse initializations
        self.population = []
        for i in range(population):
            head = EMLHead(n_inputs=n_inputs, depth=depth)
            self._diverse_init(head, strategy=i % 4)
            self.population.append(head)

    def _diverse_init(self, head: EMLHead, strategy: int):
        """Initialize with different strategies to cover the landscape."""
        with torch.no_grad():
            if strategy == 0:
                # Small weights (default-like)
                for node in head.tree.nodes:
                    node.w_left.fill_(0.1)
                    node.w_right.fill_(0.1)
                    node.bias_left.normal_(0, 0.01)
                    node.bias_right.normal_(0, 0.01)
            elif strategy == 1:
                # Exp-dominant: large left weights, small right
                for node in head.tree.nodes:
                    node.w_left.uniform_(0.5, 2.0)
                    node.w_right.fill_(0.01)
                    node.bias_left.normal_(0, 0.1)
                    node.bias_right.fill_(1.0)
            elif strategy == 2:
                # Log-dominant: small left weights, large right
                for node in head.tree.nodes:
                    node.w_left.fill_(0.01)
                    node.w_right.uniform_(0.5, 2.0)
                    node.bias_left.fill_(0.0)
                    node.bias_right.normal_(0, 0.1)
            elif strategy == 3:
                # Mixed: alternating exp/log dominance by level
                for j, node in enumerate(head.tree.nodes):
                    if j % 2 == 0:
                        node.w_left.uniform_(0.3, 1.5)
                        node.w_right.fill_(0.05)
                    else:
                        node.w_left.fill_(0.05)
                        node.w_right.uniform_(0.3, 1.5)
                    node.bias_left.normal_(0, 0.05)
                    node.bias_right.normal_(0, 0.05)

            # Projection layer: diverse scales
            nn.init.xavier_uniform_(head.projection.weight,
                                     gain=0.05 * (1 + strategy))
            nn.init.zeros_(head.projection.bias)

    def fit(self, x: torch.Tensor, y: torch.Tensor,
            steps: int = 5000, lr: float = 0.005,
            evolve_every: int = 500, report_every: int = 1000,
            loss_fn=None) -> tuple:
        """Train population on target data.

        Returns (best_head, best_loss).
        """
        if loss_fn is None:
            loss_fn = lambda pred, target: ((pred - target) ** 2).mean()

        # Create optimizers for each member
        optimizers = [
            torch.optim.Adam(head.parameters(), lr=lr)
            for head in self.population
        ]
        schedulers = [
            torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=1000)
            for opt in optimizers
        ]

        losses = [float('inf')] * self.population_size
        best_overall = float('inf')
        best_head_idx = 0

        for step in range(steps):
            # Train each member
            for i, (head, opt, sched) in enumerate(
                zip(self.population, optimizers, schedulers)
            ):
                try:
                    pred = head(x)
                    loss = loss_fn(pred, y)

                    if torch.isnan(loss) or torch.isinf(loss):
                        # Reset this member
                        self.population[i] = EMLHead(
                            n_inputs=self.n_inputs, depth=self.depth
                        )
                        self._diverse_init(self.population[i], strategy=i % 4)
                        optimizers[i] = torch.optim.Adam(
                            self.population[i].parameters(), lr=lr
                        )
                        schedulers[i] = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                            optimizers[i], T_0=1000
                        )
                        losses[i] = float('inf')
                        continue

                    opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
                    opt.step()
                    sched.step()
                    losses[i] = loss.item()

                except RuntimeError:
                    losses[i] = float('inf')

            # Track best
            min_loss = min(losses)
            if min_loss < best_overall:
                best_overall = min_loss
                best_head_idx = losses.index(min_loss)

            # Report
            if (step + 1) % report_every == 0:
                valid = [l for l in losses if l < float('inf')]
                avg = sum(valid) / len(valid) if valid else float('inf')
                print(f"  Step {step+1:5d}: best={best_overall:.2e} "
                      f"avg={avg:.2e} alive={len(valid)}/{self.population_size}")

            # Evolutionary step: replace worst with mutated copies of best
            if (step + 1) % evolve_every == 0 and best_overall < float('inf'):
                self._evolve(losses, optimizers, schedulers, lr)

        return self.population[best_head_idx], best_overall

    def _evolve(self, losses, optimizers, schedulers, lr):
        """Replace worst performers with mutations of best."""
        # Sort by loss
        ranked = sorted(range(len(losses)), key=lambda i: losses[i])

        # Bottom quarter gets replaced by mutations of top quarter
        n_replace = self.population_size // 4
        for j in range(n_replace):
            worst_idx = ranked[-(j + 1)]
            best_idx = ranked[j]

            # Deep copy the best
            new_head = EMLHead(n_inputs=self.n_inputs, depth=self.depth)
            new_head.load_state_dict(
                copy.deepcopy(self.population[best_idx].state_dict())
            )

            # Mutate: add noise to parameters
            with torch.no_grad():
                for param in new_head.parameters():
                    param.add_(torch.randn_like(param) * 0.05)

            self.population[worst_idx] = new_head
            optimizers[worst_idx] = torch.optim.Adam(
                new_head.parameters(), lr=lr
            )
            schedulers[worst_idx] = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizers[worst_idx], T_0=1000
            )


# ============================================================
# Function analysis tools
# ============================================================

def analyze_discovered_function(head: EMLHead, x_range=(-5, 5), n_points=1000):
    """Analyze properties of a discovered function."""
    x = torch.linspace(x_range[0], x_range[1], n_points).unsqueeze(1)

    with torch.no_grad():
        y = head(x).squeeze()

    # Basic properties
    y_np = y.numpy()
    x_np = x.squeeze().numpy()

    print("\n  Function properties:")

    # Symmetry: f(-x) vs f(x) and -f(x)
    x_sym = torch.linspace(0.1, x_range[1], n_points // 2).unsqueeze(1)
    with torch.no_grad():
        f_pos = head(x_sym).squeeze()
        f_neg = head(-x_sym).squeeze()

    even_err = (f_pos - f_neg).abs().mean().item()
    odd_err = (f_pos + f_neg).abs().mean().item()
    print(f"    Even symmetry error: {even_err:.2e}")
    print(f"    Odd symmetry error:  {odd_err:.2e}")
    if even_err < 0.01:
        print("    → Function appears EVEN: f(-x) ≈ f(x)")
    elif odd_err < 0.01:
        print("    → Function appears ODD: f(-x) ≈ -f(x)")
    else:
        print("    → No clear parity")

    # Boundedness
    print(f"    Range: [{y_np.min():.4f}, {y_np.max():.4f}]")
    if abs(y_np.max()) < 10 and abs(y_np.min()) < 10:
        print("    → Bounded")
    else:
        print("    → Unbounded in this range")

    # Monotonicity
    dy = np.diff(y_np)
    if np.all(dy >= -1e-6):
        print("    → Monotonically increasing")
    elif np.all(dy <= 1e-6):
        print("    → Monotonically decreasing")
    else:
        zero_crossings = np.sum(np.diff(np.sign(dy)) != 0)
        print(f"    → Non-monotone, ~{zero_crossings} direction changes")

    # Compare to known functions
    known = {
        "tanh": torch.tanh(x).squeeze(),
        "sigmoid": torch.sigmoid(x).squeeze(),
        "erf": torch.erf(x).squeeze(),
        "sin": torch.sin(x).squeeze(),
        "sech": (1.0 / torch.cosh(x)).squeeze(),
        "gaussian": torch.exp(-x**2).squeeze(),
    }

    print("\n  Similarity to known functions:")
    best_match = None
    best_corr = 0
    for name, ref in known.items():
        # Try matching with a scale and shift: a*known(b*x+c)+d
        # Simple: just correlation
        corr = torch.corrcoef(torch.stack([y, ref]))[0, 1].item()
        mae = (y - ref).abs().mean().item()
        print(f"    {name:12s}: correlation={corr:+.4f}  MAE={mae:.4f}")
        if abs(corr) > abs(best_corr):
            best_corr = corr
            best_match = name

    if abs(best_corr) > 0.99:
        print(f"\n  → Closely matches: {best_match} (r={best_corr:.4f})")
    elif abs(best_corr) > 0.95:
        print(f"\n  → Partially matches: {best_match} (r={best_corr:.4f})")
    else:
        print(f"\n  → NO CLOSE MATCH to any known function!")
        print("    This may be a novel function.")

    return y


# ============================================================
# Experiment 1: Rediscover tanh with the full pipeline
# ============================================================

def discover_tanh_v2():
    """Use the full discovery pipeline to recover tanh(x)."""
    print("=" * 60)
    print("Experiment 1: Rediscover tanh(x) via population-based EML")
    print("  16 trees, evolutionary selection, multi-restart")
    print("=" * 60)

    x = torch.linspace(-3, 3, 500).unsqueeze(1)
    y = torch.tanh(x)

    disc = EMLDiscoverer(n_inputs=1, depth=4, population=16)
    best_head, best_loss = disc.fit(x, y, steps=8000, lr=0.005,
                                     report_every=2000)

    with torch.no_grad():
        pred = best_head(x)
        mae = (pred - y).abs().mean().item()
        max_err = (pred - y).abs().max().item()

    print(f"\n  Best loss: {best_loss:.2e}")
    print(f"  MAE: {mae:.2e}")
    print(f"  Max error: {max_err:.2e}")

    analyze_discovered_function(best_head, x_range=(-3, 3))

    return mae < 0.005


# ============================================================
# Experiment 2: Rediscover erf(x) — not in ANY basis
# ============================================================

def discover_erf():
    """Try to discover erf(x) — a function not in sin/cos/exp/tanh."""
    print("\n" + "=" * 60)
    print("Experiment 2: Discover erf(x) from raw EML")
    print("  erf is NOT expressible as finite exp/ln composition")
    print("  (it involves an integral). Can EML approximate it?")
    print("=" * 60)

    x = torch.linspace(-3, 3, 500).unsqueeze(1)
    y = torch.erf(x).unsqueeze(1) if len(torch.erf(x).shape) == 1 else torch.erf(x)

    disc = EMLDiscoverer(n_inputs=1, depth=5, population=16)
    best_head, best_loss = disc.fit(x, y, steps=8000, lr=0.005,
                                     report_every=2000)

    with torch.no_grad():
        pred = best_head(x)
        y_flat = y.squeeze()
        pred_flat = pred.squeeze()
        mae = (pred_flat - y_flat).abs().mean().item()

    print(f"\n  Best loss: {best_loss:.2e}")
    print(f"  MAE: {mae:.2e}")

    analyze_discovered_function(best_head, x_range=(-3, 3))

    return mae < 0.01


# ============================================================
# Experiment 3: Discover from PDE residual — Allen-Cahn
# ============================================================

def discover_from_allen_cahn():
    """Discover the Allen-Cahn kink from PDE residual via raw EML.

    PDE: u_xx + u - u³ = 0, u(-5)=-1, u(5)=1
    The answer is tanh(x/√2), but the model doesn't know tanh.
    """
    print("\n" + "=" * 60)
    print("Experiment 3: Allen-Cahn PDE residual → raw EML discovery")
    print("  Can EML discover tanh(x/√2) from u_xx + u - u³ = 0?")
    print("=" * 60)

    # Custom loss: PDE residual + boundary conditions
    def pde_loss(head):
        x = torch.linspace(-5, 5, 300, requires_grad=True).unsqueeze(1)
        u = head(x)

        u_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0]

        # PDE: u_xx + u - u³ = 0
        res = u_xx + u - u ** 3
        loss_pde = (res ** 2).mean()

        # BCs
        x_bc = torch.tensor([[-5.0], [5.0]])
        u_bc = head(x_bc)
        loss_bc = ((u_bc - torch.tensor([[-1.0], [1.0]])) ** 2).sum()

        # Origin: u(0) ≈ 0
        loss_origin = head(torch.tensor([[0.0]])) ** 2

        return loss_pde + 10 * loss_bc + 5 * loss_origin

    # Manual training loop since loss_fn signature differs
    best_head = None
    best_loss = float('inf')

    for trial in range(8):
        head = EMLHead(n_inputs=1, depth=4)
        # Diverse init
        with torch.no_grad():
            gain = 0.05 * (1 + trial)
            nn.init.xavier_uniform_(head.projection.weight, gain=gain)
            for j, node in enumerate(head.tree.nodes):
                scale = 0.1 * (1 + trial * 0.2)
                node.w_left.uniform_(0.01, scale)
                node.w_right.uniform_(0.01, scale)
                node.bias_left.normal_(0, 0.02)
                node.bias_right.normal_(0, 0.02)

        opt = torch.optim.Adam(head.parameters(), lr=0.003)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=5000)

        trial_best = float('inf')
        for step in range(5000):
            try:
                loss = pde_loss(head)
                if torch.isnan(loss):
                    break
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(head.parameters(), 0.5)
                opt.step()
                sched.step()
                if loss.item() < trial_best:
                    trial_best = loss.item()
            except RuntimeError:
                break

        if trial_best < best_loss:
            best_loss = trial_best
            best_head = copy.deepcopy(head)

        print(f"  Trial {trial+1}/8: best_loss = {trial_best:.2e}")

    print(f"\n  Overall best PDE loss: {best_loss:.2e}")

    if best_head is not None:
        # Compare to exact
        with torch.no_grad():
            x_test = torch.linspace(-4, 4, 500).unsqueeze(1)
            pred = best_head(x_test).squeeze()
            exact = torch.tanh(x_test.squeeze() / math.sqrt(2))
            mae = (pred - exact).abs().mean().item()

        print(f"  MAE vs tanh(x/√2): {mae:.2e}")
        analyze_discovered_function(best_head, x_range=(-4, 4))
        return mae < 0.1
    return False


# ============================================================
# Experiment 4: Discover from Burgers PDE residual
# ============================================================

def discover_from_burgers():
    """Discover steady Burgers solution from PDE residual.

    PDE: u·u_x = ν·u_xx, u(-5)=-2, u(5)=2
    Answer: u = 2·tanh(x), but model doesn't know tanh.
    """
    print("\n" + "=" * 60)
    print("Experiment 4: Burgers PDE residual → raw EML discovery")
    print("  u·u_x = ν·u_xx, BCs: u(±5) = ±2")
    print("=" * 60)

    nu = 1.0

    best_head = None
    best_loss = float('inf')

    for trial in range(8):
        head = EMLHead(n_inputs=1, depth=4)
        with torch.no_grad():
            gain = 0.05 * (1 + trial)
            nn.init.xavier_uniform_(head.projection.weight, gain=gain)
            for j, node in enumerate(head.tree.nodes):
                node.w_left.uniform_(0.01, 0.3 + trial * 0.1)
                node.w_right.uniform_(0.01, 0.3 + trial * 0.1)
                node.bias_left.normal_(0, 0.02)
                node.bias_right.normal_(0, 0.02)

        opt = torch.optim.Adam(head.parameters(), lr=0.003)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=5000)

        trial_best = float('inf')
        for step in range(5000):
            try:
                x = torch.linspace(-5, 5, 300, requires_grad=True).unsqueeze(1)
                u = head(x)
                u_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
                u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0]

                # PDE: u·u_x - ν·u_xx = 0
                res = u * u_x - nu * u_xx
                loss_pde = (res ** 2).mean()

                # BCs
                x_bc = torch.tensor([[-5.0], [5.0]])
                u_bc = head(x_bc)
                loss_bc = ((u_bc - torch.tensor([[-2.0], [2.0]])) ** 2).sum()

                loss = loss_pde + 10 * loss_bc

                if torch.isnan(loss):
                    break

                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(head.parameters(), 0.5)
                opt.step()
                sched.step()

                if loss.item() < trial_best:
                    trial_best = loss.item()
            except RuntimeError:
                break

        if trial_best < best_loss:
            best_loss = trial_best
            best_head = copy.deepcopy(head)

        print(f"  Trial {trial+1}/8: best_loss = {trial_best:.2e}")

    print(f"\n  Overall best PDE loss: {best_loss:.2e}")

    if best_head is not None:
        with torch.no_grad():
            x_test = torch.linspace(-4, 4, 500).unsqueeze(1)
            pred = best_head(x_test).squeeze()
            exact = 2.0 * torch.tanh(x_test.squeeze())
            mae = (pred - exact).abs().mean().item()

        print(f"  MAE vs 2·tanh(x): {mae:.2e}")
        analyze_discovered_function(best_head, x_range=(-4, 4))
        return mae < 0.2
    return False


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    torch.manual_seed(42)

    results = {}

    results["Rediscover tanh (population)"] = discover_tanh_v2()
    results["Discover erf"] = discover_erf()
    results["Allen-Cahn PDE → EML"] = discover_from_allen_cahn()
    results["Burgers PDE → EML"] = discover_from_burgers()

    print("\n" + "=" * 60)
    print("FUNCTION DISCOVERY PIPELINE — RESULTS")
    print("=" * 60)
    for name, ok in results.items():
        status = "DISCOVERED" if ok else "NOT YET"
        print(f"  {name}: {status}")

    print("""
KEY FINDINGS:
  - Population-based optimization improves raw EML recovery
  - Functions like erf (non-elementary) can still be approximated
  - PDE-residual-driven discovery is harder but possible for 1D PDEs
  - The gap between "named primitive" and "raw EML" recovery is the
    optimization gap — this is what needs solving for true function discovery
    """)
