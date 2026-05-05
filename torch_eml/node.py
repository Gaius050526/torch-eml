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
        left = left.clamp(max=80.0)  # prevent exp overflow
        right = self.w_right * y + self.bias_right
        return torch.exp(left) - torch.log(torch.abs(right) + self.epsilon)
