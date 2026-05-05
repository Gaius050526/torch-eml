import torch
from torch_eml.tree import EMLTree


class TestEMLTreeConstruction:
    def test_depth_2_has_3_nodes(self):
        tree = EMLTree(depth=2)
        assert len(tree.nodes) == 3

    def test_depth_3_has_7_nodes(self):
        tree = EMLTree(depth=3)
        assert len(tree.nodes) == 7

    def test_depth_4_has_15_nodes(self):
        tree = EMLTree(depth=4)
        assert len(tree.nodes) == 15

    def test_depth_attribute(self):
        tree = EMLTree(depth=3)
        assert tree.depth == 3

    def test_n_leaves(self):
        tree = EMLTree(depth=4)
        assert tree.n_leaves == 16


class TestEMLTreeForward:
    def test_output_shape(self):
        tree = EMLTree(depth=3)
        x = torch.randn(32, 8)
        out = tree(x)
        assert out.shape == (32, 1)

    def test_output_shape_depth_2(self):
        tree = EMLTree(depth=2)
        x = torch.randn(16, 4)
        out = tree(x)
        assert out.shape == (16, 1)

    def test_wrong_leaf_count_raises(self):
        tree = EMLTree(depth=3)
        x = torch.randn(16, 5)
        try:
            tree(x)
            assert False, "Should have raised"
        except ValueError:
            pass

    def test_no_nan_output(self):
        tree = EMLTree(depth=3)
        x = torch.randn(32, 8)
        out = tree(x)
        assert not torch.isnan(out).any()

    def test_different_depths_different_param_count(self):
        tree2 = EMLTree(depth=2)
        tree4 = EMLTree(depth=4)
        p2 = sum(p.numel() for p in tree2.parameters())
        p4 = sum(p.numel() for p in tree4.parameters())
        assert p4 > p2


class TestEMLTreeGradient:
    def test_gradients_flow_to_all_nodes(self):
        tree = EMLTree(depth=3)
        x = torch.randn(8, 8, requires_grad=True)
        out = tree(x)
        loss = out.sum()
        loss.backward()
        for i, node in enumerate(tree.nodes):
            assert node.w_left.grad is not None, f"No gradient for node {i} w_left"
            assert node.w_right.grad is not None, f"No gradient for node {i} w_right"

    def test_gradients_flow_to_input(self):
        tree = EMLTree(depth=3)
        x = torch.randn(8, 8, requires_grad=True)
        out = tree(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
