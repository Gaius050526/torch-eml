import math
import torch
import sympy
from torch_eml.node import EMLNode
from torch_eml.tree import EMLTree
from torch_eml.symbolic import SymbolicExpression, to_symbolic


class TestSymbolicExpression:
    def test_has_string(self):
        expr = SymbolicExpression(sympy.exp(sympy.Symbol("x0")), ["x0"])
        assert isinstance(expr.string, str)
        assert "x0" in expr.string

    def test_has_sympy(self):
        s = sympy.exp(sympy.Symbol("x0"))
        expr = SymbolicExpression(s, ["x0"])
        assert expr.sympy == s

    def test_has_latex(self):
        s = sympy.exp(sympy.Symbol("x0"))
        expr = SymbolicExpression(s, ["x0"])
        assert isinstance(expr.latex, str)
        assert len(expr.latex) > 0

    def test_has_python(self):
        s = sympy.exp(sympy.Symbol("x0"))
        expr = SymbolicExpression(s, ["x0"])
        assert isinstance(expr.python, str)
        assert "math" in expr.python or "exp" in expr.python


class TestToSymbolic:
    def test_single_node_tree(self):
        """Depth-1 tree: one node, two leaves."""
        tree = EMLTree(depth=1)
        with torch.no_grad():
            tree.nodes[0].w_left.fill_(1.0)
            tree.nodes[0].w_right.fill_(1.0)
            tree.nodes[0].bias_left.fill_(0.0)
            tree.nodes[0].bias_right.fill_(0.0)
        expr = to_symbolic(tree, input_names=["x0", "x1"])
        assert "exp" in expr.string.lower() or "exp" in str(expr.sympy)
        assert "x0" in expr.string
        assert "x1" in expr.string

    def test_round_trip_depth_1(self):
        """Symbolic expression should evaluate to same result as tree."""
        tree = EMLTree(depth=1)
        with torch.no_grad():
            tree.nodes[0].w_left.fill_(1.0)
            tree.nodes[0].w_right.fill_(1.0)
            tree.nodes[0].bias_left.fill_(0.0)
            tree.nodes[0].bias_right.fill_(0.0)
        expr = to_symbolic(tree, input_names=["x0", "x1"])

        x = torch.tensor([[1.0, 2.0]])
        tree_out = tree(x).item()

        x0, x1 = sympy.symbols("x0 x1")
        sympy_out = float(expr.sympy.subs({x0: 1.0, x1: 2.0}).evalf())

        assert abs(tree_out - sympy_out) < 1e-4, (
            f"Tree={tree_out}, Symbolic={sympy_out}"
        )

    def test_round_trip_depth_2(self):
        """Symbolic expression matches tree output for depth-2 tree."""
        torch.manual_seed(123)
        tree = EMLTree(depth=2)
        expr = to_symbolic(tree, input_names=["a", "b", "c", "d"])

        x = torch.tensor([[0.5, 1.0, 1.5, 2.0]])
        tree_out = tree(x).item()

        syms = sympy.symbols("a b c d")
        sym_dict = dict(zip(syms, [0.5, 1.0, 1.5, 2.0]))
        sympy_out = float(expr.sympy.subs(sym_dict).evalf())

        assert abs(tree_out - sympy_out) < 1e-3, (
            f"Tree={tree_out}, Symbolic={sympy_out}"
        )

    def test_custom_input_names(self):
        tree = EMLTree(depth=1)
        expr = to_symbolic(tree, input_names=["market_size", "burn_rate"])
        assert "market_size" in expr.string
        assert "burn_rate" in expr.string

    def test_default_input_names(self):
        tree = EMLTree(depth=2)
        expr = to_symbolic(tree)
        assert "x0" in expr.string


from torch_eml.symbolic import snap, snap_value


class TestSnapValue:
    def test_near_pi_snaps_to_pi(self):
        val, label = snap_value(3.14, tolerance=0.05)
        assert abs(val - math.pi) < 1e-10
        assert label == "pi"

    def test_near_one_snaps(self):
        val, label = snap_value(1.02, tolerance=0.05)
        assert val == 1.0
        assert label == "1"

    def test_far_from_targets_unchanged(self):
        val, label = snap_value(7.77, tolerance=0.05)
        assert val == 7.77
        assert label is None

    def test_exact_match(self):
        val, label = snap_value(0.5, tolerance=0.05)
        assert val == 0.5
        assert label == "1/2"

    def test_negative_pi(self):
        val, label = snap_value(-3.14, tolerance=0.05)
        assert abs(val - (-math.pi)) < 1e-10
        assert label == "-pi"


class TestSnap:
    def test_snap_modifies_weights(self):
        torch.manual_seed(42)
        tree = EMLTree(depth=1)
        with torch.no_grad():
            tree.nodes[0].w_left.fill_(1.02)
        snap(tree, tolerance=0.05)
        assert tree.nodes[0].w_left.item() == 1.0

    def test_snap_returns_symbolic_expression(self):
        tree = EMLTree(depth=1)
        expr = snap(tree, tolerance=0.05)
        assert isinstance(expr, SymbolicExpression)

    def test_snap_with_validation_data(self):
        torch.manual_seed(42)
        tree = EMLTree(depth=1)
        X = torch.randn(32, 2)
        y = torch.randn(32, 1)
        expr = snap(tree, tolerance=0.05, validation_data=(X, y))
        assert isinstance(expr, SymbolicExpression)
