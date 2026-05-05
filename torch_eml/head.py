import torch
import torch.nn as nn

from torch_eml.tree import EMLTree


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
