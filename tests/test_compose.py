"""Tests for compositional EML head."""

import torch
import pytest

from torch_eml.compose import (
    ComposeHead, PrimitiveTerm, ProductTerm, TripleTerm, ComposeExpression,
)


class TestPrimitiveTerm:
    def test_sin_term_output_shape(self):
        term = PrimitiveTerm("sin", n_inputs=3)
        x = torch.randn(32, 3)
        out = term(x)
        assert out.shape == (32,)

    def test_all_primitives_run(self):
        for name in ["sin", "cos", "exp", "ln", "id", "sq", "inv"]:
            term = PrimitiveTerm(name, n_inputs=2)
            x = torch.randn(10, 2)
            out = term(x)
            assert out.shape == (10,)
            assert torch.isfinite(out).all()

    def test_symbolic_returns_sympy(self):
        term = PrimitiveTerm("sin", n_inputs=2)
        expr = term.symbolic(["x", "y"])
        assert expr is not None


class TestProductTerm:
    def test_output_shape(self):
        term = ProductTerm("sin", "cos", n_inputs=3)
        x = torch.randn(16, 3)
        out = term(x)
        assert out.shape == (16,)

    def test_func_name(self):
        term = ProductTerm("sin", "cos", n_inputs=2)
        assert term.func_name == "sin*cos"


class TestTripleTerm:
    def test_output_shape(self):
        term = TripleTerm("sin", "cos", "exp", n_inputs=3)
        x = torch.randn(16, 3)
        out = term(x)
        assert out.shape == (16,)

    def test_func_name(self):
        term = TripleTerm("sin", "cos", "exp", n_inputs=3)
        assert term.func_name == "sin*cos*exp"


class TestComposeHead:
    def test_output_shape(self):
        head = ComposeHead(n_inputs=3)
        x = torch.randn(32, 3)
        out = head(x)
        assert out.shape == (32, 1)

    def test_no_products(self):
        head = ComposeHead(n_inputs=2, primitives=["sin", "cos"], repeat=3, products=False)
        assert len(head.terms) == 6

    def test_with_products_has_more_terms(self):
        head_no = ComposeHead(n_inputs=2, primitives=["sin", "cos"], repeat=2, products=False)
        head_yes = ComposeHead(n_inputs=2, primitives=["sin", "cos"], repeat=2, products=True)
        assert len(head_yes.terms) > len(head_no.terms)

    def test_training_reduces_loss(self):
        torch.manual_seed(42)
        X = torch.linspace(-3, 3, 200).unsqueeze(1)
        y = torch.sin(X)

        head = ComposeHead(n_inputs=1, primitives=["sin", "cos", "id"], repeat=2, products=False)
        opt = torch.optim.Adam(head.parameters(), lr=0.01)

        initial_loss = torch.nn.functional.mse_loss(head(X), y).item()
        for _ in range(500):
            loss = torch.nn.functional.mse_loss(head(X), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        final_loss = loss.item()
        assert final_loss < initial_loss

    def test_prune_removes_small_terms(self):
        head = ComposeHead(n_inputs=2, primitives=["sin", "cos", "id"], repeat=2, products=False)
        n_before = len(head.terms)
        with torch.no_grad():
            head.terms[0].coeff.fill_(0.001)
            head.terms[1].coeff.fill_(0.002)
        pruned = head.prune_terms(threshold=0.01)
        assert pruned == 2
        assert len(head.terms) == n_before - 2

    def test_to_symbolic(self):
        head = ComposeHead(n_inputs=2, primitives=["sin"], repeat=1, products=False)
        expr = head.to_symbolic(input_names=["x", "y"])
        assert isinstance(expr, ComposeExpression)
        assert len(expr.string) > 0

    def test_snap_coefficients(self):
        head = ComposeHead(n_inputs=1, primitives=["sin"], repeat=1, products=False)
        with torch.no_grad():
            head.terms[0].coeff.fill_(0.98)
        head.snap_coefficients(tolerance=0.1)
        assert head.terms[0].coeff.item() == 1.0


class TestComposeExpression:
    def test_properties(self):
        head = ComposeHead(n_inputs=1, primitives=["sin"], repeat=1, products=False)
        expr = head.to_symbolic()
        assert hasattr(expr, "string")
        assert hasattr(expr, "latex")
        assert hasattr(expr, "python")
        assert hasattr(expr, "sympy")

    def test_python_is_executable(self):
        head = ComposeHead(n_inputs=1, primitives=["sin"], repeat=1, products=False)
        with torch.no_grad():
            head.terms[0].coeff.fill_(1.0)
            head.terms[0].w[0] = 1.0
            head.terms[0].b.fill_(0.0)
            head.bias.fill_(0.0)
        expr = head.to_symbolic(input_names=["x"])
        code = expr.python
        ns = {}
        exec(code, ns)
        result = ns["f"](1.0)
        import math
        assert abs(result - math.sin(1.0)) < 0.01
