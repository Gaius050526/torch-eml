import torch
import torch.nn as nn

from torch_eml.node import EMLNode


class EMLTree(nn.Module):
    """Complete binary tree of EMLNodes.

    A tree of depth d has 2^d - 1 nodes and 2^d leaf positions.
    Forward pass evaluates bottom-up: leaf values feed into the deepest
    nodes, outputs propagate up to the root.
    """

    def __init__(self, depth: int = 4, epsilon: float = 1e-7):
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1")
        self.depth = depth
        self.n_leaves = 2 ** depth
        n_nodes = 2 ** depth - 1
        self.nodes = nn.ModuleList([EMLNode(epsilon=epsilon) for _ in range(n_nodes)])

    def forward(self, leaves: torch.Tensor) -> torch.Tensor:
        """Evaluate tree bottom-up.

        Args:
            leaves: Tensor of shape [batch, 2^depth].

        Returns:
            Tensor of shape [batch, 1].
        """
        if leaves.shape[-1] != self.n_leaves:
            raise ValueError(
                f"Expected {self.n_leaves} leaf values (depth={self.depth}), "
                f"got {leaves.shape[-1]}"
            )

        current_level = [leaves[:, i] for i in range(self.n_leaves)]

        node_idx = len(self.nodes) - 1
        for level in range(self.depth - 1, -1, -1):
            n_nodes_at_level = 2 ** level
            next_level = []
            for i in range(n_nodes_at_level):
                left = current_level[2 * i]
                right = current_level[2 * i + 1]
                node = self.nodes[node_idx - (n_nodes_at_level - 1 - i)]
                next_level.append(node(left, right))
            node_idx -= n_nodes_at_level
            current_level = next_level

        return current_level[0].unsqueeze(-1)
