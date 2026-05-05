import torch
from torch_eml.node import EMLNode


class TestEMLNodeForward:
    def test_output_shape_scalar_inputs(self):
        node = EMLNode()
        x = torch.randn(32)
        y = torch.randn(32)
        out = node(x, y)
        assert out.shape == (32,)

    def test_output_shape_batched(self):
        node = EMLNode()
        x = torch.randn(16)
        y = torch.randn(16)
        out = node(x, y)
        assert out.shape == (16,)

    def test_known_values_default_weights(self):
        """With default weights (w=1, b=0), eml(x,y) = exp(x) - ln(|y| + eps)."""
        node = EMLNode()
        with torch.no_grad():
            node.w_left.fill_(1.0)
            node.w_right.fill_(1.0)
            node.bias_left.fill_(0.0)
            node.bias_right.fill_(0.0)
        x = torch.tensor([0.0, 1.0])
        y = torch.tensor([1.0, 1.0])
        out = node(x, y)
        expected = torch.exp(x) - torch.log(torch.abs(y) + 1e-7)
        torch.testing.assert_close(out, expected)

    def test_different_weights_produce_different_output(self):
        node1 = EMLNode()
        node2 = EMLNode()
        with torch.no_grad():
            node1.w_left.fill_(1.0)
            node2.w_left.fill_(2.0)
        x = torch.tensor([1.0])
        y = torch.tensor([1.0])
        out1 = node1(x, y)
        out2 = node2(x, y)
        assert not torch.allclose(out1, out2)


class TestEMLNodeGradient:
    def test_gradients_flow(self):
        node = EMLNode()
        x = torch.randn(8, requires_grad=True)
        y = torch.randn(8, requires_grad=True)
        out = node(x, y)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert y.grad is not None
        assert node.w_left.grad is not None
        assert node.w_right.grad is not None
        assert node.bias_left.grad is not None
        assert node.bias_right.grad is not None

    def test_no_nan_gradients(self):
        node = EMLNode()
        x = torch.randn(8, requires_grad=True)
        y = torch.randn(8, requires_grad=True)
        out = node(x, y)
        loss = out.sum()
        loss.backward()
        assert not torch.isnan(x.grad).any()
        assert not torch.isnan(node.w_left.grad).any()


class TestEMLNodeStability:
    def test_near_zero_y_no_nan(self):
        node = EMLNode()
        x = torch.randn(8)
        y = torch.zeros(8)
        out = node(x, y)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_large_x_no_inf(self):
        """Clamp exp input to prevent overflow."""
        node = EMLNode()
        x = torch.tensor([50.0, 100.0])
        y = torch.tensor([1.0, 1.0])
        out = node(x, y)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_large_negative_x(self):
        node = EMLNode()
        x = torch.tensor([-100.0, -200.0])
        y = torch.tensor([1.0, 1.0])
        out = node(x, y)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()
