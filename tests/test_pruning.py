import torch
from torch_eml.tree import EMLTree
from torch_eml.pruning import prune, PruneReport


class TestPrune:
    def test_returns_prune_report(self):
        tree = EMLTree(depth=3)
        X = torch.randn(64, 8)
        report = prune(tree, threshold=0.01, calibration_data=X)
        assert isinstance(report, PruneReport)

    def test_report_has_fields(self):
        tree = EMLTree(depth=3)
        X = torch.randn(64, 8)
        report = prune(tree, threshold=0.01, calibration_data=X)
        assert hasattr(report, "nodes_before")
        assert hasattr(report, "nodes_after")
        assert hasattr(report, "nodes_pruned")
        assert hasattr(report, "max_output_diff")

    def test_nodes_before_correct(self):
        tree = EMLTree(depth=3)
        X = torch.randn(64, 8)
        report = prune(tree, threshold=0.01, calibration_data=X)
        assert report.nodes_before == 7

    def test_aggressive_threshold_prunes_nodes(self):
        """With a very high threshold, most nodes should be prunable."""
        torch.manual_seed(42)
        tree = EMLTree(depth=3)
        X = torch.randn(64, 8)
        report = prune(tree, threshold=1000.0, calibration_data=X)
        assert report.nodes_pruned > 0

    def test_zero_threshold_prunes_nothing(self):
        torch.manual_seed(42)
        tree = EMLTree(depth=3)
        X = torch.randn(64, 8)
        report = prune(tree, threshold=0.0, calibration_data=X)
        assert report.nodes_pruned == 0

    def test_output_within_tolerance(self):
        """After pruning, output should be close to original."""
        torch.manual_seed(42)
        tree = EMLTree(depth=3)
        X = torch.randn(64, 8)

        with torch.no_grad():
            before = tree(X).clone()

        threshold = 0.1
        prune(tree, threshold=threshold, calibration_data=X)

        with torch.no_grad():
            after = tree(X)

        max_diff = (before - after).abs().max().item()
        assert max_diff < threshold * 10, f"Max diff {max_diff} too large"
