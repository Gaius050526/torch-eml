import torch
from torch_eml import EMLHead, EMLTree, EMLNode
from torch_eml.symbolic import SymbolicExpression
from torch_eml.pruning import PruneReport


class TestPublicAPI:
    def test_imports(self):
        """All public names are importable from torch_eml."""
        from torch_eml import EMLHead, EMLTree, EMLNode
        assert EMLHead is not None
        assert EMLTree is not None
        assert EMLNode is not None


class TestEMLHeadConvenience:
    def test_to_symbolic(self):
        head = EMLHead(n_inputs=4, depth=2)
        expr = head.to_symbolic()
        assert isinstance(expr, SymbolicExpression)
        assert len(expr.string) > 0

    def test_to_symbolic_with_names(self):
        head = EMLHead(n_inputs=4, depth=2)
        expr = head.to_symbolic(input_names=["a", "b", "c", "d"])
        assert "a" in expr.string

    def test_snap(self):
        head = EMLHead(n_inputs=4, depth=2)
        expr = head.snap(tolerance=0.1)
        assert isinstance(expr, SymbolicExpression)

    def test_prune(self):
        head = EMLHead(n_inputs=4, depth=2)
        X = torch.randn(32, 4)
        report = head.prune(threshold=0.1, calibration_data=X)
        assert isinstance(report, PruneReport)

    def test_full_pipeline(self):
        """End-to-end: train -> prune -> snap -> symbolic."""
        torch.manual_seed(42)
        head = EMLHead(n_inputs=2, depth=2)
        optimizer = torch.optim.Adam(head.parameters(), lr=0.01)

        X = torch.randn(256, 2)
        y = X.sum(dim=1, keepdim=True)

        for _ in range(50):
            pred = head(X)
            loss = torch.nn.functional.mse_loss(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        report = head.prune(threshold=0.1, calibration_data=X)
        assert isinstance(report, PruneReport)

        expr = head.snap(tolerance=0.1)
        assert isinstance(expr, SymbolicExpression)

        assert len(expr.string) > 0
        assert len(expr.latex) > 0
        assert "def f(" in expr.python

    def test_symbolic_round_trip(self):
        """Symbolic expression evaluates close to model output."""
        import sympy

        torch.manual_seed(42)
        head = EMLHead(n_inputs=2, depth=2)

        X = torch.tensor([[1.0, 2.0]])
        with torch.no_grad():
            model_out = head(X).item()
            # Project inputs to leaf space for symbolic evaluation
            leaf_vals = head.projection(X).squeeze(0)

        # depth=2 tree has 4 leaves
        names = ["l0", "l1", "l2", "l3"]
        expr = head.to_symbolic(input_names=names)
        symbols = [sympy.Symbol(n) for n in names]
        subs = {s: float(leaf_vals[i]) for i, s in enumerate(symbols)}
        sympy_out = float(expr.sympy.subs(subs).evalf())

        assert abs(model_out - sympy_out) < 1e-3, (
            f"Model={model_out}, Symbolic={sympy_out}"
        )
