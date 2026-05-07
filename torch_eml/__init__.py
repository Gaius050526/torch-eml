"""torch-eml: EML tree heads for interpretable neuro-symbolic models."""

__version__ = "0.1.0"

from torch_eml.node import EMLNode
from torch_eml.tree import EMLTree
from torch_eml.head import EMLHead
from torch_eml.symbolic import SymbolicExpression, to_symbolic, snap
from torch_eml.pruning import prune, PruneReport
from torch_eml.auto import auto_depth, search, SearchResult
from torch_eml.viz import tree_to_html, save_html
from torch_eml.primitives import (
    EMLExp, EMLLn, EMLSin, EMLCos, EMLPi,
    PRIMITIVES, verify_constructions,
)

__all__ = [
    "EMLNode",
    "EMLTree",
    "EMLHead",
    "SymbolicExpression",
    "to_symbolic",
    "snap",
    "prune",
    "PruneReport",
    "auto_depth",
    "search",
    "SearchResult",
    "tree_to_html",
    "save_html",
    "EMLExp",
    "EMLLn",
    "EMLSin",
    "EMLCos",
    "EMLPi",
    "PRIMITIVES",
    "verify_constructions",
]
