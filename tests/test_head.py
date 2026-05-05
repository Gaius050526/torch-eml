import torch
from torch_eml.head import EMLHead


class TestEMLHeadForward:
    def test_output_shape(self):
        head = EMLHead(n_inputs=5, depth=3)
        x = torch.randn(32, 5)
        out = head(x)
        assert out.shape == (32, 1)

    def test_output_shape_single_input(self):
        head = EMLHead(n_inputs=1, depth=2)
        x = torch.randn(16, 1)
        out = head(x)
        assert out.shape == (16, 1)

    def test_output_shape_large_input(self):
        head = EMLHead(n_inputs=64, depth=4)
        x = torch.randn(8, 64)
        out = head(x)
        assert out.shape == (8, 1)

    def test_no_nan_output(self):
        head = EMLHead(n_inputs=8, depth=3)
        x = torch.randn(32, 8)
        out = head(x)
        assert not torch.isnan(out).any()


class TestEMLHeadTraining:
    def test_loss_decreases(self):
        """Train on y = sum(x) and verify loss goes down."""
        torch.manual_seed(42)
        head = EMLHead(n_inputs=4, depth=3)
        optimizer = torch.optim.Adam(head.parameters(), lr=0.01)

        x = torch.randn(256, 4)
        y = x.sum(dim=1, keepdim=True)

        initial_loss = None
        for step in range(100):
            pred = head(x)
            loss = torch.nn.functional.mse_loss(pred, y)
            if step == 0:
                initial_loss = loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        final_loss = loss.item()
        assert final_loss < initial_loss, (
            f"Loss did not decrease: {initial_loss:.4f} -> {final_loss:.4f}"
        )

    def test_parameters_update(self):
        head = EMLHead(n_inputs=4, depth=2)
        params_before = [p.clone() for p in head.parameters()]

        optimizer = torch.optim.SGD(head.parameters(), lr=0.1)
        x = torch.randn(16, 4)
        y = torch.randn(16, 1)
        pred = head(x)
        loss = torch.nn.functional.mse_loss(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        any_changed = False
        for p_before, p_after in zip(params_before, head.parameters()):
            if not torch.allclose(p_before, p_after):
                any_changed = True
                break
        assert any_changed, "No parameters changed after optimization step"


class TestEMLHeadProperties:
    def test_depth_attribute(self):
        head = EMLHead(n_inputs=8, depth=5)
        assert head.tree.depth == 5

    def test_n_inputs_attribute(self):
        head = EMLHead(n_inputs=8, depth=3)
        assert head.projection.in_features == 8

    def test_projection_output_matches_tree_leaves(self):
        head = EMLHead(n_inputs=8, depth=3)
        assert head.projection.out_features == head.tree.n_leaves
