from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Sequence

import sympy
import torch

from torch_eml.tree import EMLTree

logger = logging.getLogger(__name__)


SNAP_TARGETS: dict[float, str] = {
    0.0: "0",
    1.0: "1",
    -1.0: "-1",
    2.0: "2",
    -2.0: "-2",
    3.0: "3",
    -3.0: "-3",
    0.5: "1/2",
    -0.5: "-1/2",
    1 / 3: "1/3",
    -1 / 3: "-1/3",
    2 / 3: "2/3",
    -2 / 3: "-2/3",
    0.25: "1/4",
    -0.25: "-1/4",
    0.2: "1/5",
    -0.2: "-1/5",
    1.5: "3/2",
    -1.5: "-3/2",
    math.pi: "pi",
    -math.pi: "-pi",
    math.pi / 2: "pi/2",
    -math.pi / 2: "-pi/2",
    math.pi / 4: "pi/4",
    -math.pi / 4: "-pi/4",
    math.e: "e",
    -math.e: "-e",
    math.sqrt(2): "sqrt(2)",
    -math.sqrt(2): "-sqrt(2)",
    math.log(2): "ln(2)",
    -math.log(2): "-ln(2)",
}


@dataclass
class SymbolicExpression:
    """A symbolic expression extracted from a trained EML tree."""

    _expr: sympy.Expr
    _input_names: list[str]

    @property
    def sympy(self) -> sympy.Expr:
        return self._expr

    @property
    def string(self) -> str:
        return str(self._expr)

    @property
    def latex(self) -> str:
        return sympy.latex(self._expr)

    @property
    def python(self) -> str:
        code = sympy.printing.pycode(self._expr)
        lines = [
            "import math",
            "",
            f"def f({', '.join(self._input_names)}):",
            f"    return {code}",
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        s = self.string
        if len(s) > 60:
            s = s[:57] + "..."
        return f"SymbolicExpression({s})"


def _node_to_sympy(
    node: torch.nn.Module,
    left_expr: sympy.Expr,
    right_expr: sympy.Expr,
    epsilon: float,
) -> sympy.Expr:
    """Convert a single EMLNode to a SymPy expression."""
    w_l = float(node.w_left.item())
    w_r = float(node.w_right.item())
    b_l = float(node.bias_left.item())
    b_r = float(node.bias_right.item())

    left_term = w_l * left_expr + b_l
    right_inner = w_r * right_expr + b_r
    return sympy.exp(left_term) - sympy.log(sympy.Abs(right_inner) + epsilon)


def to_symbolic(
    tree: EMLTree,
    input_names: Sequence[str] | None = None,
) -> SymbolicExpression:
    """Convert a trained EMLTree to a symbolic expression."""
    if input_names is None:
        input_names = [f"x{i}" for i in range(tree.n_leaves)]
    else:
        input_names = list(input_names)

    if len(input_names) != tree.n_leaves:
        raise ValueError(
            f"Expected {tree.n_leaves} input names, got {len(input_names)}"
        )

    epsilon = tree.nodes[0].epsilon

    symbols = [sympy.Symbol(name) for name in input_names]
    current_level: list[sympy.Expr] = list(symbols)

    # Process bottom-up, mirroring tree.forward()
    node_idx = len(tree.nodes) - 1
    for level in range(tree.depth - 1, -1, -1):
        n_nodes_at_level = 2 ** level
        next_level = []
        for i in range(n_nodes_at_level):
            left = current_level[2 * i]
            right = current_level[2 * i + 1]
            node = tree.nodes[node_idx - (n_nodes_at_level - 1 - i)]
            next_level.append(_node_to_sympy(node, left, right, epsilon))
        node_idx -= n_nodes_at_level
        current_level = next_level

    return SymbolicExpression(_expr=current_level[0], _input_names=input_names)


def snap_value(value: float, tolerance: float = 0.05) -> tuple[float, str | None]:
    """Find nearest snap target within tolerance."""
    best_dist = float("inf")
    best_target = value
    best_label = None
    for target, label in SNAP_TARGETS.items():
        dist = abs(value - target)
        if dist < best_dist and dist <= tolerance:
            best_dist = dist
            best_target = target
            best_label = label
    return best_target, best_label


def snap(
    tree: EMLTree,
    tolerance: float = 0.05,
    validation_data: tuple[torch.Tensor, torch.Tensor] | None = None,
    interactive: bool = False,
    input_names: Sequence[str] | None = None,
    max_loss_ratio: float = 2.0,
) -> SymbolicExpression:
    """Snap tree weights to clean values and return symbolic expression.

    When validation_data is provided, each weight is snapped individually
    and reverted if the loss increases beyond max_loss_ratio times the
    pre-snap loss. This prevents cascading blowup in nested exp() chains.

    Args:
        tree: The EMLTree to snap.
        tolerance: Maximum distance to snap target.
        validation_data: Optional (X, y) tuple for loss-aware snapping.
        interactive: Log snap candidates for each weight.
        input_names: Names for leaf inputs in the symbolic expression.
        max_loss_ratio: Maximum allowed loss increase ratio (default 2.0).
    """
    pre_loss = None
    X, y = None, None
    if validation_data is not None:
        X, y = validation_data
        with torch.no_grad():
            pred = tree(X)
            pre_loss = torch.nn.functional.mse_loss(pred, y).item()

    snapped_count = 0
    reverted_count = 0
    total_count = 0
    # Track the running loss threshold — each accepted snap updates it
    current_max_loss = pre_loss * max_loss_ratio if pre_loss is not None else None

    with torch.no_grad():
        for node in tree.nodes:
            for param_name in ["w_left", "w_right", "bias_left", "bias_right"]:
                param = getattr(node, param_name)
                val = param.item()
                total_count += 1

                if interactive:
                    candidates = sorted(
                        [(abs(val - t), t, label) for t, label in SNAP_TARGETS.items()],
                        key=lambda c: c[0],
                    )[:3]
                    parts = [f"{label}({target:.4f}, d={dist:.4f})" for dist, target, label in candidates]
                    logger.info(f"  {param_name}={val:.6f}  candidates: {' '.join(parts)}")

                snapped, label = snap_value(val, tolerance)
                if label is not None:
                    param.fill_(snapped)

                    # Check if this snap blows up the loss
                    if current_max_loss is not None:
                        new_loss = torch.nn.functional.mse_loss(tree(X), y).item()
                        if not math.isfinite(new_loss) or new_loss > current_max_loss:
                            param.fill_(val)  # revert
                            reverted_count += 1
                            continue

                    snapped_count += 1

    if validation_data is not None:
        with torch.no_grad():
            post_loss = torch.nn.functional.mse_loss(tree(X), y).item()
        logger.info(
            f"Snapped {snapped_count}/{total_count} weights "
            f"({reverted_count} reverted). "
            f"Loss: {pre_loss:.6f} -> {post_loss:.6f}"
        )
    else:
        logger.info(f"Snapped {snapped_count}/{total_count} weights.")

    return to_symbolic(tree, input_names=input_names)
