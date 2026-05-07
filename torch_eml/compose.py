"""Compositional EML: build expressions from verified primitive blocks.

Instead of training a flat tree of raw eml nodes, this module lets you
compose verified primitives (exp, ln, sin, cos, mul, add, pow) with
trainable scalar parameters. The search space is:

    "which functions compose, with what coefficients"

instead of:

    "find 60 arbitrary weights in a raw eml tree"

Architecture:
    ComposeHead = trainable linear combination of primitive compositions.
    Each term is: coeff * primitive(w * input + b)
    The model learns which terms matter and prunes the rest.

After training, snap + prune yields a clean symbolic expression
built from named elementary functions.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Sequence

import sympy
import torch
import torch.nn as nn

from torch_eml.primitives import EMLExp, EMLLn, EMLSin, EMLCos, EMLTanh, EMLSech

logger = logging.getLogger(__name__)


class PrimitiveTerm(nn.Module):
    """A single trainable term: coeff * f(w * x + b).

    Where f is a verified EML primitive (exp, ln, sin, cos, identity).
    """

    def __init__(self, func_name: str, n_inputs: int):
        super().__init__()
        self.func_name = func_name
        self.w = nn.Parameter(torch.randn(n_inputs) * 0.1)
        self.b = nn.Parameter(torch.zeros(1))
        self.coeff = nn.Parameter(torch.randn(1) * 0.1)

        self._funcs = {
            "sin": EMLSin(),
            "cos": EMLCos(),
            "exp": EMLExp(),
            "ln": EMLLn(),
            "tanh": EMLTanh(),
            "sech": EMLSech(),
            "id": None,       # identity: f(x) = x
            "sq": None,       # square: f(x) = x²
            "inv": None,      # inverse: f(x) = 1/x
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [batch, n_inputs] -> [batch]"""
        inner = (x * self.w).sum(dim=-1) + self.b.squeeze()

        if self.func_name == "id":
            out = inner
        elif self.func_name == "sq":
            out = inner ** 2
        elif self.func_name == "inv":
            out = 1.0 / (inner + 1e-7 * inner.sign().clamp(min=1e-7))
        elif self.func_name in self._funcs:
            out = self._funcs[self.func_name](inner)
        else:
            raise ValueError(f"Unknown function: {self.func_name}")

        return self.coeff.squeeze() * out

    def symbolic(self, input_names: list[str]) -> sympy.Expr:
        """Convert this term to a SymPy expression."""
        symbols = [sympy.Symbol(n) for n in input_names]

        # Build inner expression: w0*x0 + w1*x1 + ... + b
        inner = sum(
            float(self.w[i].item()) * symbols[i]
            for i in range(len(symbols))
        ) + float(self.b.item())

        func_map = {
            "sin": sympy.sin,
            "cos": sympy.cos,
            "exp": sympy.exp,
            "ln": sympy.log,
            "tanh": sympy.tanh,
            "sech": lambda x: 1 / sympy.cosh(x),
            "id": lambda x: x,
            "sq": lambda x: x ** 2,
            "inv": lambda x: 1 / x,
        }

        outer = func_map[self.func_name](inner)
        return float(self.coeff.item()) * outer


# Default set of primitives to try
DEFAULT_PRIMITIVES = ["sin", "cos", "exp", "ln", "id", "sq", "inv"]


class ProductTerm(nn.Module):
    """Product of two primitive terms: coeff * f(w1.x+b1) * g(w2.x+b2).

    Enables expressions like sin(x)*cos(y)*exp(-t) which cannot be
    represented as a sum of individual primitives.
    """

    def __init__(self, func_a: str, func_b: str, n_inputs: int):
        super().__init__()
        self.func_name = f"{func_a}*{func_b}"
        self.term_a = PrimitiveTerm(func_a, n_inputs)
        self.term_b = PrimitiveTerm(func_b, n_inputs)
        # Override individual coefficients — use a single product coeff
        with torch.no_grad():
            self.term_a.coeff.fill_(1.0)
            self.term_b.coeff.fill_(1.0)
        self.term_a.coeff.requires_grad_(False)
        self.term_b.coeff.requires_grad_(False)
        self.coeff = nn.Parameter(torch.randn(1) * 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.term_a(x)  # [batch]
        b = self.term_b(x)  # [batch]
        return self.coeff.squeeze() * a * b

    def symbolic(self, input_names: list[str]) -> sympy.Expr:
        a_expr = self.term_a.symbolic(input_names) / float(self.term_a.coeff.item())
        b_expr = self.term_b.symbolic(input_names) / float(self.term_b.coeff.item())
        return float(self.coeff.item()) * a_expr * b_expr


class TripleTerm(nn.Module):
    """Product of three primitives: coeff * f(.) * g(.) * h(.).

    For separable functions like sin(x)*cos(y)*exp(-2vt).
    """

    def __init__(self, func_a: str, func_b: str, func_c: str, n_inputs: int):
        super().__init__()
        self.func_name = f"{func_a}*{func_b}*{func_c}"
        self.term_a = PrimitiveTerm(func_a, n_inputs)
        self.term_b = PrimitiveTerm(func_b, n_inputs)
        self.term_c = PrimitiveTerm(func_c, n_inputs)
        with torch.no_grad():
            self.term_a.coeff.fill_(1.0)
            self.term_b.coeff.fill_(1.0)
            self.term_c.coeff.fill_(1.0)
        self.term_a.coeff.requires_grad_(False)
        self.term_b.coeff.requires_grad_(False)
        self.term_c.coeff.requires_grad_(False)
        self.coeff = nn.Parameter(torch.randn(1) * 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.term_a(x)
        b = self.term_b(x)
        c = self.term_c(x)
        return self.coeff.squeeze() * a * b * c

    def symbolic(self, input_names: list[str]) -> sympy.Expr:
        a_expr = self.term_a.symbolic(input_names) / float(self.term_a.coeff.item())
        b_expr = self.term_b.symbolic(input_names) / float(self.term_b.coeff.item())
        c_expr = self.term_c.symbolic(input_names) / float(self.term_c.coeff.item())
        return float(self.coeff.item()) * a_expr * b_expr * c_expr


class AxisTerm(nn.Module):
    """A primitive applied to a single input dimension: coeff * f(a * x_i + b).

    Unlike PrimitiveTerm which uses w·[x0,x1,...], this only looks at one
    input dimension, making the structure explicit and easier to learn.
    """

    def __init__(self, func_name: str, input_dim: int, n_inputs: int):
        super().__init__()
        self.func_name = func_name
        self.input_dim = input_dim
        self.n_inputs = n_inputs
        self.a = nn.Parameter(torch.randn(1) * 0.5 + 1.0)  # scale, init ~1
        self.b = nn.Parameter(torch.zeros(1))
        self.coeff = nn.Parameter(torch.randn(1) * 0.1)

        self._funcs = {
            "sin": EMLSin(),
            "cos": EMLCos(),
            "exp": EMLExp(),
            "ln": EMLLn(),
            "tanh": EMLTanh(),
            "sech": EMLSech(),
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        inner = self.a.squeeze() * x[:, self.input_dim] + self.b.squeeze()

        if self.func_name == "id":
            out = inner
        elif self.func_name in self._funcs:
            out = self._funcs[self.func_name](inner)
        else:
            raise ValueError(f"Unknown function: {self.func_name}")

        return self.coeff.squeeze() * out

    def symbolic(self, input_names: list[str]) -> sympy.Expr:
        sym = sympy.Symbol(input_names[self.input_dim])
        inner = float(self.a.item()) * sym + float(self.b.item())

        func_map = {
            "sin": sympy.sin, "cos": sympy.cos,
            "exp": sympy.exp, "ln": sympy.log,
            "tanh": sympy.tanh, "sech": lambda x: 1 / sympy.cosh(x),
            "id": lambda x: x,
        }
        outer = func_map[self.func_name](inner)
        return float(self.coeff.item()) * outer


class SeparableTerm(nn.Module):
    """Product of axis-aligned primitives: coeff * f(a*x_i+b) * g(c*x_j+d) [* h(e*x_k+f)].

    Each factor looks at exactly one input dimension. This is the natural
    representation for separable functions like sin(x)*cos(y)*exp(-2vt).
    """

    def __init__(self, specs: list[tuple[str, int]], n_inputs: int):
        """specs: list of (func_name, input_dim) pairs."""
        super().__init__()
        self.func_name = "*".join(f"{fn}[{d}]" for fn, d in specs)
        self.factors = nn.ModuleList([
            AxisTerm(fn, dim, n_inputs) for fn, dim in specs
        ])
        # Disable individual coefficients — use single product coeff
        for f in self.factors:
            with torch.no_grad():
                f.coeff.fill_(1.0)
            f.coeff.requires_grad_(False)
        self.coeff = nn.Parameter(torch.randn(1) * 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.factors[0](x)
        for f in self.factors[1:]:
            result = result * f(x)
        return self.coeff.squeeze() * result

    def normalize(self) -> None:
        """Canonicalize the term for cleaner symbolic output.

        - For exp factors: absorb exp(b) into the coefficient, set b=0
        - For sin/cos factors: reduce b to [-π, π] range, absorb sign flips
        - Convert sin↔cos when bias ≈ ±π/2
        """
        with torch.no_grad():
            for f in self.factors:
                if f.func_name == "exp":
                    # exp(a*x + b) = exp(b) * exp(a*x)
                    # Absorb exp(b) into coeff
                    self.coeff.mul_(math.exp(f.b.item()))
                    f.b.fill_(0.0)
                elif f.func_name in ("sin", "cos"):
                    # Reduce b to [-π, π]
                    b = f.b.item()
                    b = b % (2 * math.pi)
                    if b > math.pi:
                        b -= 2 * math.pi
                    # Absorb sign: sin(x + π) = -sin(x), cos(x + π) = -cos(x)
                    if abs(b) > math.pi / 2 + 0.1:
                        # Close to ±π: flip sign and reduce
                        if b > 0:
                            b -= math.pi
                        else:
                            b += math.pi
                        self.coeff.mul_(-1.0)
                    # Convert sin↔cos when bias ≈ ±π/2
                    # sin(x + π/2) = cos(x), sin(x - π/2) = -cos(x)
                    # cos(x + π/2) = -sin(x), cos(x - π/2) = sin(x)
                    if abs(abs(b) - math.pi / 2) < 0.3:
                        if f.func_name == "sin":
                            if b > 0:  # sin(x + π/2) = cos(x)
                                f.func_name = "cos"
                                f._funcs["cos"] = EMLCos()
                            else:  # sin(x - π/2) = -cos(x)
                                f.func_name = "cos"
                                f._funcs["cos"] = EMLCos()
                                self.coeff.mul_(-1.0)
                        else:  # cos
                            if b > 0:  # cos(x + π/2) = -sin(x)
                                f.func_name = "sin"
                                f._funcs["sin"] = EMLSin()
                                self.coeff.mul_(-1.0)
                            else:  # cos(x - π/2) = sin(x)
                                f.func_name = "sin"
                                f._funcs["sin"] = EMLSin()
                        b = 0.0
                    f.b.fill_(b)
                elif f.func_name == "tanh":
                    # tanh(-x) = -tanh(x): ensure a > 0, absorb sign into coeff
                    if f.a.item() < 0:
                        f.a.mul_(-1.0)
                        f.b.mul_(-1.0)
                        self.coeff.mul_(-1.0)
                    # Absorb bias: tanh has no clean phase identities like sin/cos
                    # but tanh(a*x + b) with b ≈ 0 → set b = 0
                    if abs(f.b.item()) < 0.05:
                        f.b.fill_(0.0)
                elif f.func_name == "sech":
                    # sech(-x) = sech(x): ensure a > 0
                    if f.a.item() < 0:
                        f.a.mul_(-1.0)
                        f.b.mul_(-1.0)
                    # sech is even, so b sign doesn't matter for sign of output
                    if abs(f.b.item()) < 0.05:
                        f.b.fill_(0.0)
            # If coeff is negative for a product, we can flip sign of one odd factor
            # sin(x) = -sin(-x), tanh(x) = -tanh(-x) → flip a and b signs
            if self.coeff.item() < 0:
                for f in self.factors:
                    if f.func_name in ("sin", "tanh"):
                        self.coeff.mul_(-1.0)
                        f.a.mul_(-1.0)
                        f.b.mul_(-1.0)
                        break

    def symbolic(self, input_names: list[str]) -> sympy.Expr:
        expr = sympy.Integer(1)
        for f in self.factors:
            f_expr = f.symbolic(input_names) / float(f.coeff.item())
            expr = expr * f_expr
        return float(self.coeff.item()) * expr


class ComposeHead(nn.Module):
    """Compositional EML head: sum of primitive terms and product terms.

    f(x) = bias + sum[ coeff_i * func_i(w_i . x + b_i) ]            (additive)
                + sum[ coeff_j * f_j(.) * g_j(.) ]                   (pairwise products)
                + sum[ coeff_k * f_k(.) * g_k(.) * h_k(.) ]          (triple products)
                + sum[ separable axis-aligned products ]               (separable)

    After training, terms with near-zero coefficients are pruned,
    yielding a clean symbolic expression.

    Args:
        n_inputs: Number of input features.
        primitives: List of primitive function names for additive terms.
        repeat: Number of copies of each additive primitive (default 2).
        products: If True, add pairwise and triple product terms (default True).
        separable: If True, add axis-aligned separable product terms (default False).
    """

    def __init__(
        self,
        n_inputs: int,
        primitives: list[str] | None = None,
        repeat: int = 2,
        products: bool = True,
        separable: bool = False,
    ):
        super().__init__()
        if primitives is None:
            primitives = DEFAULT_PRIMITIVES

        terms = []
        # Additive terms
        for func_name in primitives:
            for _ in range(repeat):
                terms.append(PrimitiveTerm(func_name, n_inputs))

        # Product terms (pairwise) — free weight vectors
        if products:
            product_pairs = [
                ("sin", "cos"), ("sin", "exp"), ("cos", "exp"),
                ("sin", "sin"), ("cos", "cos"),
                ("sin", "id"), ("cos", "id"), ("exp", "id"),
            ]
            for fa, fb in product_pairs:
                terms.append(ProductTerm(fa, fb, n_inputs))

            # Triple products (for separable 3-variable functions)
            if n_inputs >= 3:
                triple_combos = [
                    ("sin", "cos", "exp"),
                    ("cos", "sin", "exp"),
                    ("sin", "sin", "exp"),
                    ("cos", "cos", "exp"),
                ]
                for fa, fb, fc in triple_combos:
                    terms.append(TripleTerm(fa, fb, fc, n_inputs))

        # Axis-aligned separable terms
        if separable:
            from itertools import combinations, product as cart_product
            sep_funcs = [f for f in primitives if f in ("sin", "cos", "exp")]
            # Pairwise: f(x_i) * g(x_j) for i < j, all func combos
            for i, j in combinations(range(n_inputs), 2):
                for fi, fj in cart_product(sep_funcs, repeat=2):
                    terms.append(SeparableTerm(
                        [(fi, i), (fj, j)], n_inputs
                    ))
            # Triple: f(x_i) * g(x_j) * h(x_k), canonical dim order
            if n_inputs >= 3:
                for dims in combinations(range(n_inputs), 3):
                    for funcs in cart_product(sep_funcs, repeat=3):
                        terms.append(SeparableTerm(
                            list(zip(funcs, dims)), n_inputs
                        ))

        self.terms = nn.ModuleList(terms)
        self.bias = nn.Parameter(torch.zeros(1))
        self.n_inputs = n_inputs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [batch, n_inputs] -> [batch, 1]"""
        result = self.bias.expand(x.shape[0])
        for term in self.terms:
            result = result + term(x)
        return result.unsqueeze(-1)

    def prune_terms(self, threshold: float = 0.01) -> int:
        """Remove terms with |coeff| below threshold. Returns count pruned."""
        pruned = 0
        keep = []
        for term in self.terms:
            if abs(term.coeff.item()) < threshold:
                pruned += 1
            else:
                keep.append(term)
        self.terms = nn.ModuleList(keep)
        logger.info(f"Pruned {pruned} terms, {len(keep)} remaining")
        return pruned

    def snap_coefficients(self, tolerance: float = 0.1) -> None:
        """Snap coefficients and weights to clean values."""
        from torch_eml.symbolic import snap_value

        def _snap_param(param: torch.Tensor) -> None:
            """Snap a single scalar parameter."""
            val = param.item()
            snapped, label = snap_value(val, tolerance)
            if label is not None:
                param.fill_(snapped)

        def _snap_primitive(term: PrimitiveTerm) -> None:
            _snap_param(term.coeff)
            _snap_param(term.b)
            for i in range(term.w.shape[0]):
                val = term.w[i].item()
                snapped, label = snap_value(val, tolerance)
                if label is not None:
                    term.w[i] = snapped

        def _snap_axis(term: AxisTerm) -> None:
            _snap_param(term.coeff)
            _snap_param(term.a)
            _snap_param(term.b)

        with torch.no_grad():
            for term in self.terms:
                if isinstance(term, PrimitiveTerm):
                    _snap_primitive(term)
                elif isinstance(term, ProductTerm):
                    _snap_param(term.coeff)
                    _snap_primitive(term.term_a)
                    _snap_primitive(term.term_b)
                elif isinstance(term, TripleTerm):
                    _snap_param(term.coeff)
                    _snap_primitive(term.term_a)
                    _snap_primitive(term.term_b)
                    _snap_primitive(term.term_c)
                elif isinstance(term, SeparableTerm):
                    _snap_param(term.coeff)
                    for f in term.factors:
                        _snap_axis(f)

            _snap_param(self.bias)

    def to_symbolic(
        self, input_names: Sequence[str] | None = None
    ) -> ComposeExpression:
        """Extract symbolic expression from current weights."""
        if input_names is None:
            input_names = [f"x{i}" for i in range(self.n_inputs)]
        input_names = list(input_names)

        expr = sympy.Float(float(self.bias.item()))
        for term in self.terms:
            expr = expr + term.symbolic(input_names)

        expr = sympy.nsimplify(expr, rational=False)
        return ComposeExpression(_expr=expr, _input_names=input_names)

    def __repr__(self) -> str:
        funcs = [t.func_name for t in self.terms]
        return f"ComposeHead(n_inputs={self.n_inputs}, terms={funcs})"


@dataclass
class ComposeExpression:
    """Symbolic expression from a ComposeHead."""

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
        return f"ComposeExpression({s})"
