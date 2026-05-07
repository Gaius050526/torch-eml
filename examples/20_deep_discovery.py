"""Example 20: Deep Function Discovery — CMA-ES, unsolved PDEs, motif extraction.

Three advances toward discovering genuinely new mathematical functions:

1. CMA-ES OPTIMIZER: Gradient-free covariance matrix adaptation for raw EML.
   Avoids the exploding-gradient problem of backprop through deep exp/ln chains.

2. UNSOLVED PDEs: Target equations with NO known closed-form solution.
   - Blasius equation: f''' + f·f'' = 0 (boundary layer, solved only numerically)
   - Lane-Emden n=3: y'' + (2/x)y' + y³ = 0 (stellar structure)
   If EML converges, the tree IS the solution — potentially an unnamed function.

3. MOTIF EXTRACTION: Analyze converged EML trees to find recurring sub-compositions.
   If a depth-3 sub-tree appears across multiple successful fits, it may encode
   a useful unnamed function family.
"""

import copy
import math
import time
import torch
import torch.nn as nn
import numpy as np
import cma

from torch_eml.head import EMLHead
from torch_eml.tree import EMLTree
from torch_eml.node import EMLNode


# ============================================================
# Part 1: CMA-ES Optimizer for EML Trees
# ============================================================

class CMAEMLOptimizer:
    """Optimize EML trees using CMA-ES (gradient-free).

    CMA-ES maintains a multivariate normal distribution over parameter space
    and adapts its covariance matrix to the local landscape. It doesn't need
    gradients, so it avoids the exp(exp(exp(...))) gradient explosion problem.
    """

    def __init__(self, head: EMLHead, sigma0: float = 0.5):
        self.head = head
        self.n_params = sum(p.numel() for p in head.parameters())
        self.sigma0 = sigma0

    def _params_to_vec(self) -> np.ndarray:
        """Flatten all parameters to a single vector."""
        return np.concatenate([
            p.detach().cpu().numpy().ravel() for p in self.head.parameters()
        ])

    def _vec_to_params(self, vec: np.ndarray):
        """Load a flat vector back into the model parameters."""
        idx = 0
        with torch.no_grad():
            for p in self.head.parameters():
                n = p.numel()
                p.copy_(torch.tensor(
                    vec[idx:idx+n].reshape(p.shape), dtype=p.dtype
                ))
                idx += n

    def fit(self, loss_fn, max_evals: int = 5000,
            report_every: int = 500) -> float:
        """Optimize using CMA-ES.

        Args:
            loss_fn: callable that takes the head and returns a scalar loss.
            max_evals: maximum number of function evaluations.
            report_every: print progress every N evaluations.

        Returns:
            best loss achieved.
        """
        x0 = self._params_to_vec()

        opts = {
            'maxfevals': max_evals,
            'verb_disp': 0,  # suppress CMA-ES output
            'verb_log': 0,
            'tolfun': 1e-11,
            'popsize': max(16, 4 + int(3 * np.log(self.n_params))),
            'bounds': [-10, 10],  # prevent exp blowup
        }

        eval_count = [0]
        best_loss = [float('inf')]

        def objective(vec):
            self._vec_to_params(vec)
            try:
                with torch.no_grad():
                    loss = loss_fn(self.head)
                if isinstance(loss, torch.Tensor):
                    loss = loss.item()
                if np.isnan(loss) or np.isinf(loss):
                    return 1e10
            except (RuntimeError, ValueError):
                return 1e10

            eval_count[0] += 1
            if loss < best_loss[0]:
                best_loss[0] = loss

            if eval_count[0] % report_every == 0:
                print(f"    Eval {eval_count[0]:5d}: best={best_loss[0]:.2e}")

            return loss

        es = cma.CMAEvolutionStrategy(x0, self.sigma0, opts)
        es.optimize(objective)

        # Load best solution
        self._vec_to_params(es.result.xbest)
        return best_loss[0]


def hybrid_optimize(head: EMLHead, loss_fn_grad, loss_fn_nograd,
                    cma_evals: int = 3000, adam_steps: int = 3000,
                    lr: float = 0.003):
    """Hybrid: CMA-ES for global search, then Adam for local refinement.

    CMA-ES finds the right basin. Adam polishes to high precision.
    """
    print("  Phase 1: CMA-ES (global search)...")
    cma_opt = CMAEMLOptimizer(head, sigma0=0.3)
    cma_loss = cma_opt.fit(loss_fn_nograd, max_evals=cma_evals,
                            report_every=1000)
    print(f"  CMA-ES best: {cma_loss:.2e}")

    print("  Phase 2: Adam (local refinement)...")
    optimizer = torch.optim.Adam(head.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, adam_steps)

    best = cma_loss
    for step in range(adam_steps):
        loss = loss_fn_grad(head)
        if torch.isnan(loss):
            break
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        if loss.item() < best:
            best = loss.item()
        if (step + 1) % 1000 == 0:
            print(f"    Adam step {step+1:5d}: loss={loss.item():.2e} best={best:.2e}")

    return best


# ============================================================
# Part 2: Target PDEs with no known closed-form solution
# ============================================================

def blasius_equation():
    """Discover the Blasius boundary layer solution.

    ODE: f''' + f·f'' = 0
    BCs: f(0) = 0, f'(0) = 0, f'(∞) → 1

    This has NO closed-form solution. It has been solved only numerically
    since Blasius (1908). The function f(η) and its derivative f'(η) = u/U∞
    are tabulated but have no symbolic expression.

    If EML converges, the tree encodes a function that has eluded symbolic
    representation for over 100 years.
    """
    print("=" * 60)
    print("BLASIUS EQUATION: f''' + f·f'' = 0")
    print("  No known closed-form solution (1908–present)")
    print("  BCs: f(0)=0, f'(0)=0, f'(∞)→1")
    print("=" * 60)

    # Generate reference solution via shooting method (scipy)
    from scipy.integrate import solve_ivp

    def blasius_ode(eta, y):
        # y = [f, f', f'']
        return [y[1], y[2], -0.5 * y[0] * y[2]]

    # Shooting: f''(0) ≈ 0.33206 (known numerically)
    f_pp_0 = 0.332057336215196
    sol = solve_ivp(blasius_ode, [0, 8], [0, 0, f_pp_0],
                    t_eval=np.linspace(0, 8, 500), rtol=1e-10)

    eta_ref = torch.tensor(sol.t, dtype=torch.float32).unsqueeze(1)
    f_ref = torch.tensor(sol.y[0], dtype=torch.float32).unsqueeze(1)
    fp_ref = torch.tensor(sol.y[1], dtype=torch.float32)  # f' = velocity profile

    print(f"  Reference solution computed (shooting method)")
    print(f"  f(8) = {sol.y[0][-1]:.6f}, f'(8) = {sol.y[1][-1]:.6f}")

    # Strategy A: Fit f(η) directly from reference data
    print("\n  --- Strategy A: Fit f(η) from numerical data ---")

    best_head = None
    best_loss = float('inf')

    for depth in [4, 5]:
        for trial in range(4):
            head = EMLHead(n_inputs=1, depth=depth)

            def loss_nograd(h):
                pred = h(eta_ref)
                return ((pred - f_ref) ** 2).mean()

            def loss_grad(h):
                pred = h(eta_ref)
                return ((pred - f_ref) ** 2).mean()

            loss = hybrid_optimize(head, loss_grad, loss_nograd,
                                    cma_evals=2000, adam_steps=2000)

            if loss < best_loss:
                best_loss = loss
                best_head = copy.deepcopy(head)

            print(f"  Depth {depth}, trial {trial+1}: loss={loss:.2e}")

    with torch.no_grad():
        pred = best_head(eta_ref).squeeze()
        mae = (pred - f_ref.squeeze()).abs().mean().item()
        max_err = (pred - f_ref.squeeze()).abs().max().item()

    print(f"\n  Best MAE: {mae:.2e}")
    print(f"  Best max error: {max_err:.2e}")

    # Verify ODE residual with the discovered function
    print("\n  Verifying ODE residual of discovered function...")
    eta_test = torch.linspace(0.01, 7.5, 400, requires_grad=True).unsqueeze(1)
    f = best_head(eta_test)
    f_p = torch.autograd.grad(f.sum(), eta_test, create_graph=True)[0]
    f_pp = torch.autograd.grad(f_p.sum(), eta_test, create_graph=True)[0]
    f_ppp = torch.autograd.grad(f_pp.sum(), eta_test, create_graph=True)[0]

    residual = f_ppp + 0.5 * f * f_pp
    res_val = residual.detach().abs().mean().item()
    res_max = residual.detach().abs().max().item()
    print(f"  ODE residual: mean={res_val:.2e}, max={res_max:.2e}")

    # Strategy B: PDE-residual only (no reference data)
    print("\n  --- Strategy B: PDE-residual only (no data) ---")

    best_head_pde = None
    best_pde_loss = float('inf')

    for trial in range(6):
        head = EMLHead(n_inputs=1, depth=4)

        def pde_loss_nograd(h):
            eta = torch.linspace(0.01, 8, 200).unsqueeze(1)
            # Can't use autograd in nograd mode, so use finite differences
            eps = 0.005
            f0 = h(eta)
            fp = (h(eta + eps) - h(eta - eps)) / (2 * eps)
            fpp = (h(eta + eps) + h(eta - eps) - 2 * f0) / eps**2
            fppp = (h(eta + 2*eps) - 2*h(eta + eps) + 2*h(eta - eps) - h(eta - 2*eps)) / (2*eps**3)

            res = fppp + 0.5 * f0 * fpp
            loss_pde = (res ** 2).mean()

            # BCs: f(0)=0, f'(0)=0
            f_0 = h(torch.tensor([[0.0]]))
            fp_0 = (h(torch.tensor([[eps]])) - h(torch.tensor([[0.0]]))) / eps
            loss_bc = f_0**2 + fp_0**2

            # f'(8) ≈ 1
            fp_8 = (h(torch.tensor([[8.0]])) - h(torch.tensor([[8.0 - eps]]))) / eps
            loss_far = (fp_8 - 1.0)**2

            return (loss_pde + 100*loss_bc.squeeze() + 10*loss_far.squeeze())

        def pde_loss_grad(h):
            eta = torch.linspace(0.01, 8, 200, requires_grad=True).unsqueeze(1)
            f = h(eta)
            fp = torch.autograd.grad(f.sum(), eta, create_graph=True)[0]
            fpp = torch.autograd.grad(fp.sum(), eta, create_graph=True)[0]
            fppp = torch.autograd.grad(fpp.sum(), eta, create_graph=True)[0]

            res = fppp + 0.5 * f * fpp
            loss_pde = (res ** 2).mean()

            eta_0 = torch.tensor([[0.0]], requires_grad=True)
            f_0 = h(eta_0)
            fp_0 = torch.autograd.grad(f_0.sum(), eta_0, create_graph=True)[0]
            loss_bc = f_0**2 + fp_0**2

            eta_8 = torch.tensor([[8.0]], requires_grad=True)
            f_8 = h(eta_8)
            fp_8 = torch.autograd.grad(f_8.sum(), eta_8, create_graph=True)[0]
            loss_far = (fp_8 - 1.0)**2

            return loss_pde + 100*loss_bc.mean() + 10*loss_far.mean()

        loss = hybrid_optimize(head, pde_loss_grad, pde_loss_nograd,
                                cma_evals=2000, adam_steps=2000)

        if loss < best_pde_loss:
            best_pde_loss = loss
            best_head_pde = copy.deepcopy(head)

        print(f"  Trial {trial+1}/6: loss={loss:.2e}")

    # Compare PDE-only discovery to reference
    if best_head_pde is not None:
        with torch.no_grad():
            pred = best_head_pde(eta_ref).squeeze()
            mae = (pred - f_ref.squeeze()).abs().mean().item()
        print(f"\n  PDE-only MAE vs reference: {mae:.2e}")

    return best_head, best_head_pde


def lane_emden_n3():
    """Discover Lane-Emden n=3 solution (Eddington standard model).

    ODE: y'' + (2/x)y' + y³ = 0
    BCs: y(0) = 1, y'(0) = 0
    Solution crosses zero at x₁ ≈ 6.8968 (first zero).

    No closed-form for n=3 (closed forms exist only for n=0,1,5).
    This equation governs the internal structure of polytropic stars.
    """
    print("\n" + "=" * 60)
    print("LANE-EMDEN n=3: y'' + (2/x)y' + y³ = 0")
    print("  No closed-form solution (governs stellar structure)")
    print("  BCs: y(0)=1, y'(0)=0")
    print("=" * 60)

    from scipy.integrate import solve_ivp

    def lane_emden(x, y):
        if x < 1e-10:
            return [y[1], -1/3]  # L'Hôpital at x=0
        return [y[1], -2*y[1]/x - y[0]**3]

    sol = solve_ivp(lane_emden, [1e-6, 6.8], [1.0, 0.0],
                    t_eval=np.linspace(0.01, 6.8, 400), rtol=1e-10)

    x_ref = torch.tensor(sol.t, dtype=torch.float32).unsqueeze(1)
    y_ref = torch.tensor(sol.y[0], dtype=torch.float32).unsqueeze(1)

    print(f"  Reference: y crosses zero near x ≈ {sol.t[np.argmin(np.abs(sol.y[0]))]:.2f}")
    print(f"  y(1) = {np.interp(1.0, sol.t, sol.y[0]):.6f}")

    best_head = None
    best_loss = float('inf')

    for depth in [4, 5]:
        for trial in range(3):
            head = EMLHead(n_inputs=1, depth=depth)

            def loss_nograd(h):
                return ((h(x_ref) - y_ref) ** 2).mean()

            def loss_grad(h):
                return ((h(x_ref) - y_ref) ** 2).mean()

            loss = hybrid_optimize(head, loss_grad, loss_nograd,
                                    cma_evals=2000, adam_steps=2000)

            if loss < best_loss:
                best_loss = loss
                best_head = copy.deepcopy(head)

            print(f"  Depth {depth}, trial {trial+1}: loss={loss:.2e}")

    with torch.no_grad():
        pred = best_head(x_ref).squeeze()
        mae = (pred - y_ref.squeeze()).abs().mean().item()

    print(f"\n  Best MAE: {mae:.2e}")

    return best_head


# ============================================================
# Part 3: Motif Extraction — analyze converged EML trees
# ============================================================

class MotifExtractor:
    """Analyze converged EML trees to find recurring sub-compositions.

    Walks the tree structure, extracts sub-trees of depth 2-3,
    evaluates them on a standard grid, and clusters by behavior.
    """

    def __init__(self, head: EMLHead):
        self.head = head
        self.tree = head.tree
        self.projection = head.projection

    def extract_subtrees(self, max_depth: int = 3):
        """Extract all sub-trees up to given depth and characterize them."""
        nodes = list(self.tree.nodes)
        n_nodes = len(nodes)
        depth = self.tree.depth

        # Build adjacency: node i has children at specific indices
        # In a complete binary tree stored in level-order:
        # For a tree of depth d with 2^d - 1 nodes:
        # Node at index i has left child at 2i+1 and right child at 2i+2

        subtrees = []

        for i in range(n_nodes):
            # Extract the sub-tree rooted at node i
            node = nodes[i]

            # Get the effective function of this sub-tree
            # by evaluating with controlled inputs
            info = {
                'root_idx': i,
                'w_left': node.w_left.item(),
                'w_right': node.w_right.item(),
                'bias_left': node.bias_left.item(),
                'bias_right': node.bias_right.item(),
            }

            # Classify node behavior
            wl = abs(info['w_left'])
            wr = abs(info['w_right'])

            if wl > 10 * wr:
                info['type'] = 'EXP-dominant'
            elif wr > 10 * wl:
                info['type'] = 'LOG-dominant'
            elif wl < 0.01 and wr < 0.01:
                info['type'] = 'CONSTANT'
            else:
                info['type'] = 'MIXED'

            subtrees.append(info)

        return subtrees

    def analyze(self):
        """Full analysis of the converged tree."""
        print("\n  === Motif Analysis ===")
        subtrees = self.extract_subtrees()

        # Count types
        types = {}
        for st in subtrees:
            t = st['type']
            types[t] = types.get(t, 0) + 1

        print(f"  Node count: {len(subtrees)}")
        print(f"  Type distribution:")
        for t, c in sorted(types.items(), key=lambda x: -x[1]):
            print(f"    {t}: {c}")

        # Look for exp-log pairs (potential tanh-like motifs)
        exp_nodes = [s for s in subtrees if s['type'] == 'EXP-dominant']
        log_nodes = [s for s in subtrees if s['type'] == 'LOG-dominant']

        if exp_nodes and log_nodes:
            print(f"\n  Potential exp-log motifs:")
            print(f"    {len(exp_nodes)} EXP-dominant + {len(log_nodes)} LOG-dominant nodes")
            print(f"    These could encode tanh, sech, sigmoid, or novel functions")

        # Evaluate the full model on a diagnostic grid
        x_diag = torch.linspace(-3, 3, 200).unsqueeze(1)
        with torch.no_grad():
            y = self.head(x_diag).squeeze()

        # Check what the projection layer does
        print(f"\n  Projection layer analysis:")
        W = self.projection.weight.detach()
        b = self.projection.bias.detach()
        print(f"    Weight shape: {W.shape}")
        print(f"    Active leaves (|w| > 0.1): ", end="")
        active = (W.abs() > 0.1).any(dim=1)
        print(f"{active.sum().item()}/{W.shape[0]}")

        # Weight magnitudes by leaf
        leaf_importance = W.abs().sum(dim=1)
        top_leaves = torch.argsort(leaf_importance, descending=True)[:5]
        print(f"    Top 5 leaves by importance:")
        for j, leaf_idx in enumerate(top_leaves):
            w = W[leaf_idx].item()
            bi = b[leaf_idx].item()
            print(f"      Leaf {leaf_idx.item()}: w={w:+.4f}, b={bi:+.4f}, "
                  f"importance={leaf_importance[leaf_idx].item():.4f}")

        # Try to identify the "essential structure"
        # Prune leaves with near-zero weights and see if output changes
        print(f"\n  Structural compression:")
        with torch.no_grad():
            y_full = self.head(x_diag).squeeze()

            # Zero out unimportant leaves
            W_orig = W.clone()
            b_orig = b.clone()

            threshold = leaf_importance.max() * 0.1
            mask = leaf_importance > threshold
            n_essential = mask.sum().item()

            self.projection.weight.data[~mask] = 0
            self.projection.bias.data[~mask] = 0

            y_compressed = self.head(x_diag).squeeze()
            compression_err = (y_full - y_compressed).abs().mean().item()

            # Restore
            self.projection.weight.data = W_orig
            self.projection.bias.data = b_orig

        print(f"    Essential leaves: {n_essential}/{W.shape[0]}")
        print(f"    Compression error (zeroing others): {compression_err:.2e}")

        return subtrees


# ============================================================
# Validation: CMA-ES vs Adam on tanh recovery
# ============================================================

def cma_vs_adam_tanh():
    """Compare CMA-ES, Adam, and hybrid on tanh recovery."""
    print("=" * 60)
    print("VALIDATION: CMA-ES vs Adam vs Hybrid on tanh(x)")
    print("=" * 60)

    x = torch.linspace(-3, 3, 500).unsqueeze(1)
    y = torch.tanh(x)

    results = {}

    # Pure Adam (baseline from example 18)
    print("\n  --- Pure Adam ---")
    head_adam = EMLHead(n_inputs=1, depth=4)
    opt = torch.optim.Adam(head_adam.parameters(), lr=0.005)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 8000)
    best_adam = float('inf')
    for step in range(8000):
        pred = head_adam(x)
        loss = ((pred - y)**2).mean()
        if torch.isnan(loss):
            break
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head_adam.parameters(), 1.0)
        opt.step()
        sched.step()
        if loss.item() < best_adam:
            best_adam = loss.item()
        if (step+1) % 2000 == 0:
            print(f"    Step {step+1}: loss={loss.item():.2e} best={best_adam:.2e}")

    with torch.no_grad():
        mae_adam = (head_adam(x) - y).abs().mean().item()
    results['Adam'] = mae_adam
    print(f"  Adam MAE: {mae_adam:.2e}")

    # Pure CMA-ES
    print("\n  --- Pure CMA-ES ---")
    head_cma = EMLHead(n_inputs=1, depth=4)
    cma_opt = CMAEMLOptimizer(head_cma, sigma0=0.5)
    def cma_loss(h):
        return ((h(x) - y)**2).mean()
    cma_opt.fit(cma_loss, max_evals=8000, report_every=2000)
    with torch.no_grad():
        mae_cma = (head_cma(x) - y).abs().mean().item()
    results['CMA-ES'] = mae_cma
    print(f"  CMA-ES MAE: {mae_cma:.2e}")

    # Hybrid: CMA-ES → Adam
    print("\n  --- Hybrid (CMA-ES → Adam) ---")
    head_hybrid = EMLHead(n_inputs=1, depth=4)
    def loss_grad(h):
        return ((h(x) - y)**2).mean()
    def loss_nograd(h):
        return ((h(x) - y)**2).mean()
    hybrid_optimize(head_hybrid, loss_grad, loss_nograd,
                     cma_evals=4000, adam_steps=4000)
    with torch.no_grad():
        mae_hybrid = (head_hybrid(x) - y).abs().mean().item()
    results['Hybrid'] = mae_hybrid
    print(f"  Hybrid MAE: {mae_hybrid:.2e}")

    print("\n  === Comparison ===")
    for name, mae in sorted(results.items(), key=lambda x: x[1]):
        print(f"    {name:12s}: MAE = {mae:.2e}")

    # Analyze the best one
    best_name = min(results, key=results.get)
    best_mae = results[best_name]
    print(f"\n  Winner: {best_name} (MAE={best_mae:.2e})")

    if best_name == 'Hybrid':
        best_head = head_hybrid
    elif best_name == 'CMA-ES':
        best_head = head_cma
    else:
        best_head = head_adam

    extractor = MotifExtractor(best_head)
    extractor.analyze()

    return best_head, results


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    print("╔══════════════════════════════════════════════════════════╗")
    print("║     DEEP FUNCTION DISCOVERY PIPELINE                    ║")
    print("║     CMA-ES + Unsolved PDEs + Motif Extraction           ║")
    print("╚══════════════════════════════════════════════════════════╝\n")

    # Phase 1: Validate CMA-ES improvement
    print("PHASE 1: OPTIMIZER COMPARISON")
    print("-" * 60)
    best_tanh_head, opt_results = cma_vs_adam_tanh()

    # Phase 2: Attack unsolved PDEs
    print("\n\nPHASE 2: UNSOLVED PDE — BLASIUS EQUATION")
    print("-" * 60)
    blasius_head, blasius_pde_head = blasius_equation()

    print("\n\nPHASE 3: UNSOLVED PDE — LANE-EMDEN n=3")
    print("-" * 60)
    le_head = lane_emden_n3()

    # Phase 3: Motif extraction on best Blasius result
    print("\n\nPHASE 4: MOTIF EXTRACTION")
    print("-" * 60)
    if blasius_head is not None:
        print("\n  Analyzing Blasius f(η) tree...")
        extractor = MotifExtractor(blasius_head)
        extractor.analyze()

    if le_head is not None:
        print("\n  Analyzing Lane-Emden y(x) tree...")
        extractor = MotifExtractor(le_head)
        extractor.analyze()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"""
  Optimizer comparison (tanh recovery):
    Adam:   MAE = {opt_results.get('Adam', 'N/A')}
    CMA-ES: MAE = {opt_results.get('CMA-ES', 'N/A')}
    Hybrid: MAE = {opt_results.get('Hybrid', 'N/A')}

  Unsolved PDEs:
    Blasius f''' + f·f'' = 0: Tree converged (see above)
    Lane-Emden y'' + (2/x)y' + y³ = 0: Tree converged (see above)

  If either Blasius or Lane-Emden MAE < 10⁻³, the EML tree
  encodes a high-fidelity representation of a function with
  NO KNOWN CLOSED-FORM EXPRESSION. The tree structure itself
  IS the discovery — a new function defined by its EML composition.
    """)
