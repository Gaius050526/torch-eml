"""Tests for EML tree visualization."""

import os
import tempfile

import torch

from torch_eml import EMLHead, EMLTree
from torch_eml.viz import tree_to_html, save_html


class TestTreeToHtml:
    def test_returns_html_string(self):
        tree = EMLTree(depth=2)
        html = tree_to_html(tree)
        assert "<!DOCTYPE html>" in html
        assert "eml" in html

    def test_works_with_head(self):
        head = EMLHead(n_inputs=2, depth=2)
        html = tree_to_html(head)
        assert "<!DOCTYPE html>" in html

    def test_includes_equation(self):
        tree = EMLTree(depth=2)
        html = tree_to_html(tree, equation="exp(x0) - ln(x1)")
        assert "exp(x0) - ln(x1)" in html

    def test_includes_title(self):
        tree = EMLTree(depth=2)
        html = tree_to_html(tree, title="My Tree")
        assert "My Tree" in html

    def test_shows_pruned_nodes(self):
        head = EMLHead(n_inputs=2, depth=2)
        X = torch.randn(50, 2)
        head.prune(threshold=999.0, calibration_data=X)  # prune everything
        html = tree_to_html(head)
        assert "pruned" in html

    def test_all_nodes_present(self):
        tree = EMLTree(depth=3)
        html = tree_to_html(tree)
        # depth=3 tree has 7 nodes
        assert html.count("<circle") == 7


class TestSaveHtml:
    def test_saves_file(self):
        tree = EMLTree(depth=2)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "tree.html")
            result = save_html(tree, path)
            assert os.path.exists(result)
            with open(result) as f:
                content = f.read()
            assert "<!DOCTYPE html>" in content

    def test_creates_parent_dirs(self):
        tree = EMLTree(depth=2)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "sub", "dir", "tree.html")
            result = save_html(tree, path)
            assert os.path.exists(result)
