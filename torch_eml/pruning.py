from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from torch_eml.tree import EMLTree
from torch_eml.node import EMLNode


@dataclass
class PruneReport:
    """Report from pruning a tree."""

    nodes_before: int
    nodes_after: int
    nodes_pruned: int
    max_output_diff: float


class ConstantNode(nn.Module):
    """A pruned node that always returns a constant value."""

    def __init__(self, value: float, epsilon: float = 1e-7):
        super().__init__()
        self.value = value
        self.epsilon = epsilon
        self.w_left = nn.Parameter(torch.tensor([0.0]), requires_grad=False)
        self.w_right = nn.Parameter(torch.tensor([0.0]), requires_grad=False)
        self.bias_left = nn.Parameter(torch.tensor([0.0]), requires_grad=False)
        self.bias_right = nn.Parameter(torch.tensor([0.0]), requires_grad=False)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.full_like(x, self.value)


def prune(
    tree: EMLTree,
    threshold: float,
    calibration_data: torch.Tensor,
) -> PruneReport:
    """Prune low-contribution nodes from the tree.

    For each node, measures how much the output changes when the node is
    replaced with its mean constant output. If the change is below threshold,
    the node is replaced with a ConstantNode.

    Args:
        tree: The EMLTree to prune (modified in-place).
        threshold: Maximum allowed output change per node.
        calibration_data: Tensor of shape [n_samples, n_leaves].

    Returns:
        PruneReport with pruning statistics.
    """
    nodes_before = len(tree.nodes)
    pruned_count = 0

    with torch.no_grad():
        original_output = tree(calibration_data).clone()

        for idx in range(len(tree.nodes) - 1, -1, -1):
            node = tree.nodes[idx]

            if isinstance(node, ConstantNode):
                continue

            current_output = tree(calibration_data)

            mean_val = current_output.mean().item()
            original_node = tree.nodes[idx]
            tree.nodes[idx] = ConstantNode(mean_val, epsilon=original_node.epsilon)

            new_output = tree(calibration_data)
            max_diff = (original_output - new_output).abs().max().item()

            if max_diff <= threshold:
                pruned_count += 1
            else:
                tree.nodes[idx] = original_node

    with torch.no_grad():
        final_output = tree(calibration_data)
        max_output_diff = (original_output - final_output).abs().max().item()

    return PruneReport(
        nodes_before=nodes_before,
        nodes_after=nodes_before - pruned_count,
        nodes_pruned=pruned_count,
        max_output_diff=max_output_diff,
    )
