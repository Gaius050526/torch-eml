"""Auto-tuning and structure search for EML trees."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import torch
import torch.nn as nn

from torch_eml.head import EMLHead
from torch_eml.symbolic import SymbolicExpression

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Result from auto-depth or structure search."""

    head: EMLHead
    depth: int
    val_loss: float
    expression: SymbolicExpression | None = None
    trials: list[dict] = field(default_factory=list)


def auto_depth(
    n_inputs: int,
    X: torch.Tensor,
    y: torch.Tensor,
    depths: tuple[int, ...] = (2, 3, 4, 5, 6),
    epochs: int = 1000,
    lr: float = 0.01,
    val_split: float = 0.2,
) -> SearchResult:
    """Try multiple tree depths, return the best by validation loss.

    Args:
        n_inputs: Number of input features.
        X: Training inputs, shape [n_samples, n_inputs].
        y: Training targets, shape [n_samples, 1].
        depths: Tuple of depths to try.
        epochs: Training epochs per depth.
        lr: Learning rate.
        val_split: Fraction of data for validation.

    Returns:
        SearchResult with the best head and metadata.
    """
    n = X.shape[0]
    n_val = max(1, int(n * val_split))
    perm = torch.randperm(n)
    X_train, X_val = X[perm[n_val:]], X[perm[:n_val]]
    y_train, y_val = y[perm[n_val:]], y[perm[:n_val]]

    best: SearchResult | None = None
    trials = []

    for depth in depths:
        head = EMLHead(n_inputs=n_inputs, depth=depth)
        optimizer = torch.optim.Adam(head.parameters(), lr=lr)

        for _ in range(epochs):
            pred = head(X_train)
            loss = nn.functional.mse_loss(pred, y_train)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            val_loss = nn.functional.mse_loss(head(X_val), y_val).item()
            train_loss = nn.functional.mse_loss(head(X_train), y_train).item()

        trial = {"depth": depth, "val_loss": val_loss, "train_loss": train_loss}
        trials.append(trial)
        logger.info(
            f"depth={depth}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}"
        )

        if best is None or val_loss < best.val_loss:
            best = SearchResult(head=head, depth=depth, val_loss=val_loss, trials=trials)

    best.trials = trials
    logger.info(f"Best depth: {best.depth} (val_loss={best.val_loss:.6f})")
    return best


def search(
    n_inputs: int,
    X: torch.Tensor,
    y: torch.Tensor,
    max_depth: int = 6,
    epochs: int = 1000,
    finetune_epochs: int = 200,
    lr: float = 0.01,
    prune_threshold: float = 0.05,
    snap_tolerance: float = 0.1,
) -> SearchResult:
    """Full structure search: auto-depth, prune, fine-tune, snap.

    Finds the best tree depth, then iteratively prunes low-contribution
    nodes and fine-tunes remaining weights, producing the simplest
    accurate symbolic expression.

    Args:
        n_inputs: Number of input features.
        X: Training inputs, shape [n_samples, n_inputs].
        y: Training targets, shape [n_samples, 1].
        max_depth: Maximum tree depth to try.
        epochs: Training epochs for initial depth search.
        finetune_epochs: Epochs for fine-tuning after each prune round.
        lr: Learning rate.
        prune_threshold: Pruning sensitivity threshold.
        snap_tolerance: Weight snapping tolerance.

    Returns:
        SearchResult with pruned/snapped head and symbolic expression.
    """
    depths = tuple(range(2, max_depth + 1))
    result = auto_depth(
        n_inputs, X, y, depths=depths, epochs=epochs, lr=lr
    )
    head = result.head

    # Iterative prune + fine-tune
    for round_num in range(5):
        report = head.prune(threshold=prune_threshold, calibration_data=X)
        if report.nodes_pruned == 0:
            break

        logger.info(
            f"Prune round {round_num + 1}: "
            f"{report.nodes_pruned} nodes pruned "
            f"({report.nodes_after} remaining)"
        )

        # Fine-tune remaining weights
        optimizer = torch.optim.Adam(
            [p for p in head.parameters() if p.requires_grad], lr=lr * 0.1
        )
        for _ in range(finetune_epochs):
            pred = head(X)
            loss = nn.functional.mse_loss(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Snap weights and extract expression
    expr = head.snap(tolerance=snap_tolerance, validation_data=(X, y))

    with torch.no_grad():
        final_loss = nn.functional.mse_loss(head(X), y).item()

    logger.info(f"Final: depth={head.tree.depth}, loss={final_loss:.6f}")
    logger.info(f"Equation: {expr.string}")

    return SearchResult(
        head=head,
        depth=head.tree.depth,
        val_loss=final_loss,
        expression=expr,
        trials=result.trials,
    )
