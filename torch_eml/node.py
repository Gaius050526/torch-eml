import torch
import torch.nn as nn


class EMLNode(nn.Module):
    """Single EML node: eml(x, y) = exp(w_l * x + b_l) - ln(|w_r * y + b_r| + epsilon)."""

    def __init__(self, epsilon: float = 1e-7):
        super().__init__()
        self.w_left = nn.Parameter(torch.ones(1))
        self.w_right = nn.Parameter(torch.ones(1))
        self.bias_left = nn.Parameter(torch.zeros(1))
        self.bias_right = nn.Parameter(torch.zeros(1))
        self.epsilon = epsilon

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        left = self.w_left * x + self.bias_left
        left = left.clamp(min=-80.0, max=80.0)
        right = self.w_right * y + self.bias_right
        return torch.exp(left) - torch.log(torch.abs(right) + self.epsilon)

    def __repr__(self) -> str:
        return (
            f"EMLNode(w_left={self.w_left.item():.4f}, w_right={self.w_right.item():.4f}, "
            f"bias_left={self.bias_left.item():.4f}, bias_right={self.bias_right.item():.4f})"
        )
