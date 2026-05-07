"""Tests for auto-tuning and structure search."""

import torch
import pytest

from torch_eml.auto import auto_depth, search, SearchResult


class TestAutoDepth:
    def test_returns_search_result(self):
        X = torch.randn(100, 2)
        y = (X[:, 0] + X[:, 1]).unsqueeze(1)
        result = auto_depth(n_inputs=2, X=X, y=y, depths=(2, 3), epochs=50, lr=0.01)
        assert isinstance(result, SearchResult)
        assert result.head is not None
        assert result.depth in (2, 3)
        assert result.val_loss >= 0

    def test_trials_recorded(self):
        X = torch.randn(80, 2)
        y = X[:, 0:1]
        result = auto_depth(n_inputs=2, X=X, y=y, depths=(2, 3, 4), epochs=30, lr=0.01)
        assert len(result.trials) == 3
        for trial in result.trials:
            assert "depth" in trial
            assert "val_loss" in trial
            assert "train_loss" in trial

    def test_picks_reasonable_depth(self):
        """Simple linear target should not need deep trees."""
        torch.manual_seed(42)
        X = torch.randn(200, 2)
        y = (0.5 * X[:, 0] + 0.3 * X[:, 1]).unsqueeze(1)
        result = auto_depth(n_inputs=2, X=X, y=y, depths=(2, 3, 4, 5), epochs=200, lr=0.01)
        # Should pick a small depth — at least not the maximum
        assert result.depth <= 5


class TestSearch:
    def test_returns_expression(self):
        torch.manual_seed(42)
        X = torch.randn(100, 2)
        y = (X[:, 0] * 0.5).unsqueeze(1)
        result = search(
            n_inputs=2, X=X, y=y,
            max_depth=3, epochs=100, finetune_epochs=50, lr=0.01,
        )
        assert isinstance(result, SearchResult)
        assert result.expression is not None
        assert len(result.expression.string) > 0

    def test_search_reduces_loss(self):
        torch.manual_seed(42)
        X = torch.randn(100, 2)
        y = (X[:, 0] + X[:, 1]).unsqueeze(1)
        result = search(
            n_inputs=2, X=X, y=y,
            max_depth=3, epochs=200, finetune_epochs=50, lr=0.01,
        )
        assert result.val_loss < 1.0
