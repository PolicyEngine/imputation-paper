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

import numpy as np
import pandas as pd
import pytest

from imputation_paper import smoke
from imputation_paper.experiments.conditions import run_condition
from imputation_paper.methods import REGISTRY, VARIANT_KEYS, get_method

pytest.importorskip("populace.fit", reason="methods extra not installed")
pytest.importorskip("microimpute", reason="methods extra not installed")
pytest.importorskip("statmatch", reason="methods extra not installed")

#: Every real method that runs on the toy task (the harness baseline is
#: covered by the smoke suite; the derived-net-worth variants need a balance
#: sheet and are exercised on one below).
REAL_METHODS = tuple(
    k for k in REGISTRY if k != "weighted_marginal" and k not in VARIANT_KEYS
)

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


@pytest.mark.parametrize(
    "method_key",
    [
        "populace_fit",
        "populace_fit_ungated",
        "microimpute_qrf",
        "statmatch_rand_hotdeck",
    ],
)
def test_adapters_are_seed_reproducible(method_key: str) -> None:
    """Same seed, same draw, for every stochastic adapter.

    microimpute's QRF takes no seed, so its forest and its sampler run on the
    package's fixed internal seed: its draws repeat exactly as well.
    """
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
    pd.testing.assert_frame_equal(draws[0], draws[1])


def test_gates_off_actually_differs() -> None:
    """The gates-off ablation changes the draws of a multi-class target."""
    kwargs = dict(
        train=DATASET.train,
        test=DATASET.test,
        predictors=DATASET.predictors,
        targets=DATASET.targets,
        weight_column=DATASET.weight_column,
        seed=0,
    )
    gated = run_condition("populace_fit", **kwargs).imputed
    ungated = run_condition("populace_fit_ungated", **kwargs).imputed
    assert not gated.equals(ungated)


def _toy_balance_sheet(seed: int, n: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    age = rng.integers(20, 85, n).astype(float)
    income = rng.lognormal(10.5, 0.8, n)
    owns = rng.random(n) < 0.6
    frame = pd.DataFrame(
        {
            "age": age,
            "income": income,
            "fin": rng.lognormal(8.0, 1.5, n) * (rng.random(n) < 0.9),
            "nfin": np.where(owns, income * rng.uniform(2.0, 6.0, n), 0.0),
            "debt": np.where(owns, income * rng.uniform(0.0, 3.0, n), 0.0),
            "weight": rng.uniform(1.0, 8.0, n),
        }
    )
    frame["networth"] = frame["fin"] + frame["nfin"] - frame["debt"]
    return frame


@pytest.mark.parametrize("method_key", VARIANT_KEYS)
def test_derived_net_worth_arms_impose_the_identity(method_key: str) -> None:
    """On a balance sheet the variants run, and every drawn row adds up."""
    donors, receivers = _toy_balance_sheet(1, 500), _toy_balance_sheet(2, 200)
    targets = ["fin", "nfin", "debt", "networth"]
    fit = get_method(method_key).constructor()
    drawn = fit(donors, ["age", "income"], targets, "weight", seed=0)(receivers)

    assert list(drawn.columns) == targets
    assert drawn.index.equals(receivers.index)
    assert np.isfinite(drawn.to_numpy()).all()
    np.testing.assert_array_equal(
        drawn["networth"].to_numpy(),
        (drawn["fin"] + drawn["nfin"] - drawn["debt"]).to_numpy(),
    )
    # The components are the base method's own draws for a three-target fit.
    base_key = {
        "populace_fit_derived": "populace_fit",
        "populace_fit_unchained_derived": "populace_fit_unchained",
    }[method_key]
    base = get_method(base_key).constructor()
    components = base(donors, ["age", "income"], targets[:3], "weight", seed=0)(
        receivers
    )
    pd.testing.assert_frame_equal(drawn.loc[:, targets[:3]], components)


@pytest.mark.parametrize("method_key", VARIANT_KEYS)
def test_derived_net_worth_arms_are_skipped_without_a_balance_sheet(
    method_key: str,
) -> None:
    """The toy task has no balance sheet; the sweep records the arm as skipped."""
    with pytest.raises(NotImplementedError):
        run_condition(
            method_key,
            DATASET.train,
            DATASET.test,
            DATASET.predictors,
            DATASET.targets,
            weight_column=DATASET.weight_column,
            seed=0,
        )


@pytest.mark.parametrize(
    "method_key",
    ["statmatch_hotdeck", "statmatch_rand_hotdeck", "statmatch_constrained_hotdeck"],
)
def test_statmatch_donates_observed_donor_values(method_key: str) -> None:
    """Hot-deck imputations are actual donor values, per target."""
    fit = get_method(method_key).constructor()
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
