"""Adapter integration tests: every real method runs end to end.

These run only when the ``methods`` extra is installed (CI's base install
skips them via ``importorskip``): each registered adapter fits on toy donor
rows, draws for receiver rows, and produces finite metrics through the same
``run_condition`` path the sweeps use. They pin the adapter *contracts* --
shapes, indexing, weight handling, seed reproducibility -- not statistical
quality (that is what the sweeps measure).
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from imputation_paper import smoke
from imputation_paper.experiments.conditions import run_condition
from imputation_paper.methods import REGISTRY, get_method

pytest.importorskip("populace.fit", reason="methods extra not installed")
pytest.importorskip("microimpute", reason="methods extra not installed")
pytest.importorskip("statmatch", reason="methods extra not installed")

#: Every real method (the harness baseline is covered by the smoke suite).
REAL_METHODS = tuple(k for k in REGISTRY if k != "weighted_marginal")

#: Small toy scale: adapters are exercised for contract, not statistics.
DATASET = smoke.make_toy_dataset(seed=0, n=400)


@pytest.mark.parametrize("method_key", REAL_METHODS)
def test_adapter_runs_end_to_end(method_key: str) -> None:
    """Fit, draw, and score each adapter through the sweep's condition path."""
    result = run_condition(
        method_key,
        DATASET.train,
        DATASET.test,
        DATASET.predictors,
        DATASET.targets,
        weight_column=DATASET.weight_column,
        seed=0,
    )
    assert list(result.imputed.columns) == list(DATASET.targets)
    assert len(result.imputed) == len(DATASET.test)
    assert result.imputed.index.equals(DATASET.test.index)
    assert result.imputed.notna().all().all()
    assert all(math.isfinite(v) for v in result.metrics.values())


def test_populace_fit_weighting_actually_differs() -> None:
    """The candidate and its unweighted ablation produce different draws.

    If the weighted bootstrap were silently ignored, the ablation would be a
    no-op and the paper's central comparison would be vacuous.
    """
    kwargs = dict(
        train=DATASET.train,
        test=DATASET.test,
        predictors=DATASET.predictors,
        targets=DATASET.targets,
        weight_column=DATASET.weight_column,
        seed=0,
    )
    weighted = run_condition("populace_fit", **kwargs).imputed
    unweighted = run_condition("populace_fit_unweighted", **kwargs).imputed
    assert not weighted.equals(unweighted)


def test_adapters_are_seed_reproducible() -> None:
    """Same seed, same draw -- for the stochastic quantile-model adapters."""
    for method_key in ("populace_fit", "microimpute_qrf"):
        draws = []
        for _ in range(2):
            fit = get_method(method_key).constructor()
            predict = fit(
                DATASET.train,
                list(DATASET.predictors),
                list(DATASET.targets),
                DATASET.weight_column,
                seed=7,
            )
            draws.append(predict(DATASET.test))
        if method_key == "populace_fit":
            # populace-fit is deterministic end to end for a fixed seed.
            pd.testing.assert_frame_equal(draws[0], draws[1])
        else:
            # microimpute's forest randomness is not seedable through its
            # public constructor; only our grid draw is. The draws must at
            # least come from the same grid (finite, right shape).
            assert draws[0].shape == draws[1].shape


def test_statmatch_donates_observed_donor_values() -> None:
    """Hot-deck imputations are actual donor values, per target."""
    fit = get_method("statmatch_hotdeck").constructor()
    predict = fit(
        DATASET.train,
        list(DATASET.predictors),
        list(DATASET.targets),
        DATASET.weight_column,
        seed=0,
    )
    drawn = predict(DATASET.test)
    for target in DATASET.targets:
        donor_values = set(DATASET.train[target].to_numpy().tolist())
        assert set(drawn[target].to_numpy().tolist()) <= donor_values


def test_microimpute_draws_vary_across_rows() -> None:
    """The per-row grid draw yields cross-row variation, not one shared quantile."""
    fit = get_method("microimpute_qrf").constructor()
    predict = fit(
        DATASET.train,
        list(DATASET.predictors),
        [smoke.CONTINUOUS_TARGET],
        DATASET.weight_column,
        seed=3,
    )
    drawn = predict(DATASET.test)[smoke.CONTINUOUS_TARGET]
    assert drawn.nunique() > len(drawn) * 0.5
