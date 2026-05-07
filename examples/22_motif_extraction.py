"""Example 22: Deep Motif Extraction from converged EML trees.

After an EML tree converges on a target function, the tree structure
encodes that function as nested exp/ln compositions. This module
analyzes the tree to:

1. Evaluate each sub-tree as a standalone function
2. Characterize sub-tree behavior (parity, boundedness, shape)
3. Compare sub-tree functions to known mathematical functions
4. Identify recurring patterns across multiple converged trees
5. Extract candidate "new primitives" — sub-compositions that could
   be named and reused

If a depth-2 sub-tree appears in both a Blasius-fit tree and a
Lane-Emden-fit tree with similar functional behavior, it may encode
a genuinely useful unnamed function.
"""

import copy
import math
import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass, field

from torch_eml.head import EMLHead
from torch_eml.tree import EMLTree
from torch_eml.node import EMLNode


# ============================================================
# Sub-tree evaluation
# ============================================================

@dataclass
class SubTreeFunction:
    """A sub-tree extracted and characterized as a standalone function."""
    root_idx: int
    depth: int
    n_leaves: int
    # Functional signature: f(x) evaluated on [-3, 3]
    x_grid: np.ndarray = field(repr=False)
    y_values: np.ndarray = field(repr=False)
    # Properties
    is_even: bool = False
    is_odd: bool = False
    is_bounded: bool = False
    is_monotone: bool = False
    y_range: tuple = (0, 0)
    # Best known match
    best_match: str = "unknown"
    match_correlation: float = 0.0
    match_mae: float = float('inf')


def evaluate_all_nodes(head: EMLHead, x_grid: torch.Tensor = None) -> dict:
    """Evaluate all internal nodes of an EMLHead as functions of x.

    Uses hooks to capture each node's output during a forward pass.
    Returns {node_idx: y_values_array}.
    """
    if x_grid is None:
        x_grid = torch.linspace(-3, 3, 500).unsqueeze(1)
    elif x_grid.dim() == 1:
        x_grid = x_grid.unsqueeze(1)

    node_outputs = {}
    hooks = []

    for idx, node in enumerate(head.tree.nodes):
        def make_hook(i):
            def hook_fn(module, input, output):
                node_outputs[i] = output.detach().cpu().numpy()
            return hook_fn
        h = node.register_forward_hook(make_hook(idx))
        hooks.append(h)

    with torch.no_grad():
        head(x_grid)

    for h in hooks:
        h.remove()

    return node_outputs


def _characterize(root_idx: int, tree_depth: int,
                  x_np: np.ndarray, y_values: np.ndarray) -> SubTreeFunction:
    """Characterize a node's output as a mathematical function."""
    # Determine sub-tree depth from node index
    level = 0
    level_start = 0
    while level_start + 2**level <= root_idx:
        level_start += 2**level
        level += 1
    sub_depth = tree_depth - level
    n_sub_leaves = 2 ** sub_depth

    # Filter NaN/Inf
    valid = np.isfinite(y_values)
    if valid.sum() < 10:
        return SubTreeFunction(
            root_idx=root_idx, depth=sub_depth, n_leaves=n_sub_leaves,
            x_grid=x_np, y_values=y_values,
            best_match="degenerate", match_correlation=0.0
        )

    y_clean = np.where(valid, y_values, 0)

    # Characterize
    stf = SubTreeFunction(
        root_idx=root_idx, depth=sub_depth, n_leaves=n_sub_leaves,
        x_grid=x_np, y_values=y_clean
    )

    # Symmetry (using valid points in positive range)
    mid = len(x_np) // 2
    if mid > 10:
        y_pos = y_clean[mid+1:mid+1+mid]
        y_neg_reversed = y_clean[:mid][::-1]
        n_sym = min(len(y_pos), len(y_neg_reversed))
        if n_sym > 5:
            even_err = np.mean(np.abs(y_pos[:n_sym] - y_neg_reversed[:n_sym]))
            odd_err = np.mean(np.abs(y_pos[:n_sym] + y_neg_reversed[:n_sym]))
            scale = max(np.std(y_clean), 1e-6)
            stf.is_even = even_err / scale < 0.05
            stf.is_odd = odd_err / scale < 0.05

    # Boundedness
    stf.y_range = (float(np.nanmin(y_clean)), float(np.nanmax(y_clean)))
    stf.is_bounded = abs(stf.y_range[0]) < 50 and abs(stf.y_range[1]) < 50

    # Monotonicity
    dy = np.diff(y_clean)
    if np.all(dy[valid[1:] & valid[:-1]] >= -1e-6):
        stf.is_monotone = True
    elif np.all(dy[valid[1:] & valid[:-1]] <= 1e-6):
        stf.is_monotone = True

    # Compare to known functions
    known_fns = {
        'exp': np.exp(np.clip(x_np, -10, 10)),
        'exp(-x²)': np.exp(-x_np**2),
        'tanh': np.tanh(x_np),
        'sigmoid': 1 / (1 + np.exp(-x_np)),
        'sech': 1 / np.cosh(x_np),
        'sin': np.sin(x_np),
        'cos': np.cos(x_np),
        'erf': np.array([math.erf(xi) for xi in x_np]),
        'x': x_np,
        'x²': x_np**2,
        'ln(1+exp(x))': np.log1p(np.exp(np.clip(x_np, -20, 20))),  # softplus
        'abs': np.abs(x_np),
    }

    best_corr = 0
    best_name = "unknown"
    best_mae = float('inf')

    for name, ref in known_fns.items():
        if not np.all(np.isfinite(ref)):
            continue
        # Try both direct and scaled matching
        # Correlation
        if np.std(y_clean) > 1e-8 and np.std(ref) > 1e-8:
            corr = np.corrcoef(y_clean, ref)[0, 1]
            if np.isfinite(corr) and abs(corr) > abs(best_corr):
                best_corr = corr
                best_name = name

        # Also try affine match: a*ref + b ≈ y
        if np.std(ref) > 1e-8:
            A = np.column_stack([ref, np.ones_like(ref)])
            try:
                coeffs, _, _, _ = np.linalg.lstsq(A, y_clean, rcond=None)
                fitted = A @ coeffs
                mae = np.mean(np.abs(fitted - y_clean))
                if mae < best_mae:
                    best_mae = mae
            except:
                pass

    stf.best_match = best_name
    stf.match_correlation = best_corr
    stf.match_mae = best_mae

    return stf


def evaluate_subtree(head: EMLHead, root_idx: int,
                     x_grid: torch.Tensor = None) -> SubTreeFunction:
    """Evaluate a single node's output as a function of x."""
    if x_grid is None:
        x_grid = torch.linspace(-3, 3, 500).unsqueeze(1)
    elif x_grid.dim() == 1:
        x_grid = x_grid.unsqueeze(1)

    node_outputs = evaluate_all_nodes(head, x_grid)
    x_np = x_grid.squeeze().numpy()

    if root_idx not in node_outputs:
        return SubTreeFunction(
            root_idx=root_idx, depth=0, n_leaves=0,
            x_grid=x_np, y_values=np.zeros(len(x_grid)),
            best_match="degenerate", match_correlation=0.0
        )

    return _characterize(root_idx, head.tree.depth, x_np, node_outputs[root_idx])


# ============================================================
# Cross-tree motif comparison
# ============================================================

def compare_subtrees(stf1: SubTreeFunction, stf2: SubTreeFunction) -> float:
    """Compute similarity between two sub-tree functions.

    Returns correlation between their functional outputs.
    """
    # Interpolate to common grid if needed
    y1 = stf1.y_values
    y2 = stf2.y_values

    valid = np.isfinite(y1) & np.isfinite(y2)
    if valid.sum() < 10:
        return 0.0

    y1c = y1[valid]
    y2c = y2[valid]

    if np.std(y1c) < 1e-8 or np.std(y2c) < 1e-8:
        return 0.0

    corr = np.corrcoef(y1c, y2c)[0, 1]
    return corr if np.isfinite(corr) else 0.0


def find_cross_tree_motifs(trees: list[tuple[str, EMLHead]],
                           threshold: float = 0.95):
    """Find sub-tree functions that appear in multiple trees.

    Args:
        trees: list of (name, head) pairs
        threshold: minimum correlation to consider a match

    Returns recurring motifs.
    """
    print("\n  === Cross-Tree Motif Analysis ===")

    all_subtrees = {}  # (tree_name, node_idx) → SubTreeFunction

    x_grid = torch.linspace(-3, 3, 500).unsqueeze(1)
    for tree_name, head in trees:
        print(f"\n  Extracting sub-trees from '{tree_name}'...")
        node_outputs = evaluate_all_nodes(head, x_grid)
        x_np = x_grid.squeeze().numpy()

        for node_idx, y_vals in node_outputs.items():
            stf = _characterize(node_idx, head.tree.depth, x_np, y_vals)
            all_subtrees[(tree_name, node_idx)] = stf

    # Compare sub-trees across different trees
    print(f"\n  Comparing {len(all_subtrees)} sub-trees across {len(trees)} trees...")

    matches = []
    tree_names = [t[0] for t in trees]

    for i, ((name1, idx1), stf1) in enumerate(all_subtrees.items()):
        for (name2, idx2), stf2 in list(all_subtrees.items())[i+1:]:
            if name1 == name2:
                continue  # Only cross-tree

            sim = compare_subtrees(stf1, stf2)
            if abs(sim) > threshold:
                matches.append({
                    'tree1': name1, 'node1': idx1,
                    'tree2': name2, 'node2': idx2,
                    'correlation': sim,
                    'match1': stf1.best_match,
                    'match2': stf2.best_match,
                    'stf1': stf1,
                    'stf2': stf2,
                })

    print(f"\n  Found {len(matches)} cross-tree matches (|r| > {threshold}):")
    for m in sorted(matches, key=lambda x: -abs(x['correlation']))[:20]:
        print(f"    {m['tree1']}[{m['node1']}] ↔ {m['tree2']}[{m['node2']}]: "
              f"r={m['correlation']:+.4f} "
              f"({m['match1']}, {m['match2']})")

    # Identify novel motifs: high cross-tree correlation but no known match
    novel = [m for m in matches
             if m['stf1'].best_match == 'unknown' or m['stf2'].best_match == 'unknown'
             or m['stf1'].match_mae > 0.1]

    if novel:
        print(f"\n  *** {len(novel)} POTENTIALLY NOVEL MOTIFS ***")
        for m in novel[:10]:
            stf = m['stf1']
            print(f"    Cross-tree match r={m['correlation']:+.4f}")
            print(f"      Properties: even={stf.is_even} odd={stf.is_odd} "
                  f"bounded={stf.is_bounded} monotone={stf.is_monotone}")
            print(f"      Range: [{stf.y_range[0]:.3f}, {stf.y_range[1]:.3f}]")
            print(f"      Closest known: {stf.best_match} (r={stf.match_correlation:.3f})")

    return matches, all_subtrees


# ============================================================
# Train trees on multiple targets, then extract motifs
# ============================================================

def train_tree(name: str, x: torch.Tensor, y: torch.Tensor,
               depth: int = 5, steps: int = 8000) -> EMLHead:
    """Train an EML tree on target data."""
    print(f"\n  Training '{name}' (depth={depth})...")

    best_head = None
    best_loss = float('inf')

    for trial in range(4):
        head = EMLHead(n_inputs=1, depth=depth)
        with torch.no_grad():
            gain = 0.05 * (1 + trial)
            nn.init.xavier_uniform_(head.projection.weight, gain=gain)
            for j, node in enumerate(head.tree.nodes):
                node.w_left.uniform_(0.05, 0.3 + trial * 0.1)
                node.w_right.uniform_(0.05, 0.3 + trial * 0.1)
                node.bias_left.normal_(0, 0.02)
                node.bias_right.normal_(0, 0.02)

        opt = torch.optim.Adam(head.parameters(), lr=0.005)
        sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=1000)

        trial_best = float('inf')
        for step in range(steps):
            pred = head(x)
            loss = ((pred - y) ** 2).mean()
            if torch.isnan(loss):
                break
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            opt.step()
            sched.step()
            if loss.item() < trial_best:
                trial_best = loss.item()

        if trial_best < best_loss:
            best_loss = trial_best
            best_head = copy.deepcopy(head)

    with torch.no_grad():
        mae = (best_head(x) - y).abs().mean().item()
    print(f"    Best loss: {best_loss:.2e}, MAE: {mae:.2e}")
    return best_head


def main():
    torch.manual_seed(42)
    np.random.seed(42)

    print("╔══════════════════════════════════════════════════════════╗")
    print("║     MOTIF EXTRACTION FROM CONVERGED EML TREES           ║")
    print("╚══════════════════════════════════════════════════════════╝\n")

    x = torch.linspace(-3, 3, 500).unsqueeze(1)

    # Train trees on diverse target functions
    targets = {
        'tanh': torch.tanh(x),
        'erf': torch.erf(x),
        'sech': (1.0 / torch.cosh(x)),
        'gaussian': torch.exp(-x**2),
        'softplus': torch.log1p(torch.exp(x)),
    }

    trained_trees = []
    for name, y in targets.items():
        head = train_tree(name, x, y, depth=4, steps=6000)
        trained_trees.append((name, head))

    # Also train on Blasius and Lane-Emden reference data
    from scipy.integrate import solve_ivp

    # Blasius reference
    def blasius_ode(eta, y):
        return [y[1], y[2], -0.5 * y[0] * y[2]]
    sol = solve_ivp(blasius_ode, [0, 8], [0, 0, 0.332057336215196],
                    t_eval=np.linspace(0.01, 8, 300), rtol=1e-10)
    x_blas = torch.tensor(sol.t, dtype=torch.float32).unsqueeze(1)
    y_blas = torch.tensor(sol.y[0], dtype=torch.float32).unsqueeze(1)
    head_blas = train_tree('blasius', x_blas, y_blas, depth=5, steps=8000)
    trained_trees.append(('blasius', head_blas))

    # Lane-Emden n=3 reference
    def lane_emden(x, y):
        if x < 1e-10:
            return [y[1], -1/3]
        return [y[1], -2*y[1]/x - y[0]**3]
    sol2 = solve_ivp(lane_emden, [1e-6, 6.5], [1.0, 0.0],
                     t_eval=np.linspace(0.01, 6.5, 300), rtol=1e-10)
    x_le = torch.tensor(sol2.t, dtype=torch.float32).unsqueeze(1)
    y_le = torch.tensor(sol2.y[0], dtype=torch.float32).unsqueeze(1)
    head_le = train_tree('lane_emden', x_le, y_le, depth=5, steps=8000)
    trained_trees.append(('lane_emden', head_le))

    # ============================================================
    # Per-tree analysis
    # ============================================================
    print("\n" + "=" * 60)
    print("PER-TREE SUBTREE ANALYSIS")
    print("=" * 60)

    for name, head in trained_trees:
        print(f"\n  --- {name} ---")
        n_nodes = len(head.tree.nodes)

        # Evaluate each node as a sub-tree function
        interesting = []
        for node_idx in range(min(n_nodes, 15)):  # Analyze first 15 nodes
            stf = evaluate_subtree(head, node_idx)
            if stf.best_match != "degenerate" and stf.is_bounded:
                interesting.append(stf)
                if stf.match_correlation > 0.9 or stf.best_match == "unknown":
                    props = []
                    if stf.is_even: props.append("even")
                    if stf.is_odd: props.append("odd")
                    if stf.is_monotone: props.append("monotone")
                    if stf.is_bounded: props.append("bounded")
                    print(f"    Node {stf.root_idx}: match={stf.best_match} "
                          f"(r={stf.match_correlation:+.3f}) "
                          f"range=[{stf.y_range[0]:.2f},{stf.y_range[1]:.2f}] "
                          f"{'|'.join(props)}")

    # ============================================================
    # Cross-tree comparison
    # ============================================================
    print("\n" + "=" * 60)
    print("CROSS-TREE MOTIF COMPARISON")
    print("=" * 60)

    # Compare all pairs of "unsolved PDE" trees with "known function" trees
    unsolved = [(n, h) for n, h in trained_trees if n in ('blasius', 'lane_emden')]
    known = [(n, h) for n, h in trained_trees if n not in ('blasius', 'lane_emden')]

    matches, all_subtrees = find_cross_tree_motifs(
        unsolved + known, threshold=0.90
    )

    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "=" * 60)
    print("MOTIF EXTRACTION SUMMARY")
    print("=" * 60)

    # Count how many cross-tree motifs involve unsolved PDEs
    unsolved_names = {'blasius', 'lane_emden'}
    cross_unsolved = [m for m in matches
                      if m['tree1'] in unsolved_names or m['tree2'] in unsolved_names]

    print(f"""
  Trees trained: {len(trained_trees)}
  Total cross-tree matches (|r| > 0.90): {len(matches)}
  Matches involving unsolved PDEs: {len(cross_unsolved)}

  Interpretation:
    When a sub-tree from the Blasius or Lane-Emden solution matches
    a sub-tree from a known function (tanh, erf, sech, ...), it means
    the unsolved PDE's solution SHARES STRUCTURAL COMPONENTS with
    known functions — even though the full solution is not any
    known function.

    When sub-trees from Blasius and Lane-Emden match EACH OTHER
    but don't match any known function, those are candidates for
    genuinely new mathematical primitives — function fragments
    that appear in multiple unsolved equations.
    """)


if __name__ == "__main__":
    main()
