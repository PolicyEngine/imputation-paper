"""Run one method on one donor/receiver split and score the imputation.

This is the inner cell of the sweep: fit a registered method on the donor rows,
draw target values for the receiver rows, and evaluate the drawn columns with
the weighted metrics. It is method-agnostic -- it only knows the registry's
:class:`~imputation_paper.methods.FitFn` contract and the metric functions -- so
the candidate, the ablations, and the baselines all run through the same path.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from imputation_paper import methods as method_registry
from imputation_paper.experiments import metrics


@dataclass(frozen=True)
class ConditionResult:
    """The scored outcome of one (method, seed) cell.

    Attributes:
        method: Registry key of the method run.
        seed: Seed of the donor/receiver split.
        metrics: Per-target metric values, keyed ``"{target}.{metric}"``.
        imputed: The drawn receiver target columns (kept for figures/diagnostics).
    """

    method: str
    seed: int
    metrics: dict[str, float]
    imputed: pd.DataFrame = field(repr=False)


def run_condition(
    method_key: str,
    train: pd.DataFrame,
    test: pd.DataFrame,
    predictors: Sequence[str],
    targets: Sequence[str],
    *,
    weight_column: str,
    seed: int = 0,
) -> ConditionResult:
    """Fit ``method_key`` on ``train`` and score its receiver draws on ``test``.

    Scores each target with the implemented weighted metrics: pinball loss and
    Wasserstein-1 against the donor for every target, plus zero-share error for
    the zero-inflated ones. Receiver metrics are weighted by the receiver
    weights; the Wasserstein donor reference is weighted by the donor weights.

    Args:
        method_key: Key into :data:`imputation_paper.methods.REGISTRY`.
        train: Donor rows.
        test: Receiver rows (their target columns are the held-out truth).
        predictors: Predictor column names.
        targets: Target column names.
        weight_column: The weight column present in both ``train`` and ``test``.
        seed: Seed recorded on the result (and available to the fit).

    Returns:
        A :class:`ConditionResult`.
    """
    method = method_registry.get_method(method_key)
    fit = method.constructor()
    predict = fit(train, list(predictors), list(targets), weight_column, seed=seed)
    imputed = predict(test)

    train_weights = train[weight_column].to_numpy(dtype=np.float64)
    test_weights = test[weight_column].to_numpy(dtype=np.float64)

    scores: dict[str, float] = {}
    for target in targets:
        y_true = test[target].to_numpy(dtype=np.float64)
        y_pred = imputed[target].to_numpy(dtype=np.float64)
        donor = train[target].to_numpy(dtype=np.float64)

        scores[f"{target}.pinball_loss"] = metrics.weighted_pinball_loss(
            y_true, y_pred, weights=test_weights
        )
        scores[f"{target}.wasserstein1"] = metrics.weighted_wasserstein1(
            y_pred,
            donor,
            imputed_weights=test_weights,
            donor_weights=train_weights,
        )
        # Zero-share error is only meaningful where the donor target actually
        # carries a zero mass.
        if np.any(np.abs(donor) <= 1e-6):
            scores[f"{target}.zero_share_error"] = metrics.zero_share_error(
                y_true, y_pred, true_weights=test_weights, pred_weights=test_weights
            )

    return ConditionResult(
        method=method_key, seed=seed, metrics=scores, imputed=imputed
    )


def rows_from_result(result: ConditionResult) -> list[dict[str, Any]]:
    """Flatten a :class:`ConditionResult` into long-format metric rows.

    One row per (method, seed, target, metric), matching the schema the sweep
    writes to ``metrics_long.csv``.
    """
    rows: list[dict[str, Any]] = []
    for name, value in result.metrics.items():
        target, metric = name.split(".", 1)
        rows.append(
            {
                "method": result.method,
                "seed": result.seed,
                "target": target,
                "metric": metric,
                "value": value,
            }
        )
    return rows
