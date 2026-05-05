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


def _get_node_outputs(
    tree: EMLTree, calibration_data: torch.Tensor
) -> dict[int, torch.Tensor]:
    """Run the tree forward and capture each node's output."""
    outputs: dict[int, torch.Tensor] = {}
    current_level = [calibration_data[:, i] for i in range(tree.n_leaves)]

    node_idx = len(tree.nodes) - 1
    for level in range(tree.depth - 1, -1, -1):
        n_nodes_at_level = 2 ** level
        next_level = []
        for i in range(n_nodes_at_level):
            left = current_level[2 * i]
            right = current_level[2 * i + 1]
            idx = node_idx - (n_nodes_at_level - 1 - i)
            node = tree.nodes[idx]
            out = node(left, right)
            outputs[idx] = out
            next_level.append(out)
        node_idx -= n_nodes_at_level
        current_level = next_level

    return outputs


def prune(
    tree: EMLTree,
    threshold: float,
    calibration_data: torch.Tensor,
) -> PruneReport:
    """Prune low-contribution nodes from the tree.

    For each node, measures how much the tree output changes when the node
    is replaced with its mean output value. If the change is below threshold,
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

        # Capture each node's individual output for mean computation
        node_outputs = _get_node_outputs(tree, calibration_data)

        # Try pruning each node from leaves up (reverse order)
        for idx in range(len(tree.nodes) - 1, -1, -1):
            node = tree.nodes[idx]

            if isinstance(node, ConstantNode):
                continue

            # Use this node's mean output as the replacement constant
            mean_val = node_outputs[idx].mean().item()
            original_node = tree.nodes[idx]
            tree.nodes[idx] = ConstantNode(mean_val, epsilon=original_node.epsilon)

            new_output = tree(calibration_data)
            max_diff = (original_output - new_output).abs().max().item()

            if max_diff <= threshold:
                pruned_count += 1
            else:
                # Restore original node
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
