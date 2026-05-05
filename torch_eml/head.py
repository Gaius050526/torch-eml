from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from torch_eml.tree import EMLTree
from torch_eml.symbolic import SymbolicExpression, to_symbolic, snap
from torch_eml.pruning import PruneReport, prune


class EMLHead(nn.Module):
    """High-level EML head: linear projection + EML tree.

    Maps arbitrary input features to a single scalar output through
    a trainable EML tree. After training, use snap() and to_symbolic()
    to extract a closed-form equation.

    Args:
        n_inputs: Number of input features.
        depth: Depth of the EML tree (default 4). Creates 2^depth - 1 nodes.
        epsilon: Numerical stability constant for log (default 1e-7).
    """

    def __init__(self, n_inputs: int, depth: int = 4, epsilon: float = 1e-7):
        super().__init__()
        self.tree = EMLTree(depth=depth, epsilon=epsilon)
        self.projection = nn.Linear(n_inputs, self.tree.n_leaves)
        self._init_weights()

    def _init_weights(self):
        """Initialize weights for numerical stability.

        The projection layer uses small weights to keep leaf values near zero.
        Tree node left-branch weights start small to prevent cascading
        exponential blowup through multiple tree levels.
        """
        nn.init.xavier_uniform_(self.projection.weight, gain=0.1)
        nn.init.zeros_(self.projection.bias)
        for node in self.tree.nodes:
            node.w_left.data.fill_(0.1)
            node.w_right.data.fill_(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: project inputs to leaf values, evaluate tree.

        Args:
            x: Tensor of shape [batch, n_inputs].

        Returns:
            Tensor of shape [batch, 1].
        """
        leaves = self.projection(x)
        return self.tree(leaves)

    def to_symbolic(
        self, input_names: Sequence[str] | None = None
    ) -> SymbolicExpression:
        """Extract symbolic expression from current weights.

        Note: This operates on the tree only, not the projection layer.
        Leaf inputs correspond to projected feature combinations.
        If input_names is not provided, defaults to x0, x1, ...
        """
        return to_symbolic(self.tree, input_names=input_names)

    def snap(
        self,
        tolerance: float = 0.05,
        interactive: bool = False,
        validation_data: tuple[torch.Tensor, torch.Tensor] | None = None,
        input_names: Sequence[str] | None = None,
    ) -> SymbolicExpression:
        """Snap tree weights to clean values and return symbolic expression."""
        tree_val_data = None
        if validation_data is not None:
            X, y = validation_data
            with torch.no_grad():
                leaves = self.projection(X)
            tree_val_data = (leaves, y)
        return snap(
            self.tree,
            tolerance=tolerance,
            validation_data=tree_val_data,
            interactive=interactive,
            input_names=input_names,
        )

    def prune(
        self,
        threshold: float = 0.01,
        calibration_data: torch.Tensor | None = None,
    ) -> PruneReport:
        """Prune low-contribution branches from the tree.

        Args:
            threshold: Maximum allowed output change per node.
            calibration_data: Tensor of shape [n_samples, n_inputs]. Required.
        """
        if calibration_data is None:
            raise ValueError("calibration_data is required for pruning")
        with torch.no_grad():
            calibration_data = self.projection(calibration_data)
        return prune(self.tree, threshold=threshold, calibration_data=calibration_data)

    def __repr__(self) -> str:
        return (
            f"EMLHead(n_inputs={self.projection.in_features}, "
            f"depth={self.tree.depth}, "
            f"params={sum(p.numel() for p in self.parameters())})"
        )
