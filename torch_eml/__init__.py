"""torch-eml: EML tree heads for interpretable neuro-symbolic models."""

__version__ = "0.1.0"

from torch_eml.node import EMLNode
from torch_eml.tree import EMLTree
from torch_eml.head import EMLHead
from torch_eml.symbolic import SymbolicExpression, to_symbolic, snap
from torch_eml.pruning import prune, PruneReport

__all__ = [
    "EMLNode",
    "EMLTree",
    "EMLHead",
    "SymbolicExpression",
    "to_symbolic",
    "snap",
    "prune",
    "PruneReport",
]
