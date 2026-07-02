"""CI smoke suite: the harness must run end to end on the base install.

Nothing here imports a method package (populace-fit, microimpute,
py-statmatch): the suite proves the plumbing -- registry, splits, metrics,
condition runner, CLI -- with the dependency-free ``weighted_marginal``
baseline, exactly the path ``imp demo`` walks.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from imputation_paper import smoke
from imputation_paper.cli import main
from imputation_paper.cli.figures import make_figures
from imputation_paper.cli.sweep import run_sweep
from imputation_paper.experiments import metrics
from imputation_paper.experiments.conditions import rows_from_result, run_condition
from imputation_paper.experiments.holdout import paired_splits, split_frame
from imputation_paper.methods import (
    ABLATION_KEYS,
    BASELINE_KEYS,
    CANDIDATE_KEYS,
    REGISTRY,
    get_method,
    list_methods,
)

#: The method surface the paper commits to; the registry must expose exactly
#: these (adding a method is a deliberate PLAN.md change, not drift).
EXPECTED_METHOD_KEYS = {
    "populace_fit",
    "populace_fit_unweighted",
    "populace_fit_unchained",
    "plain_qrf",
    "microimpute_qrf",
    "microimpute_ols",
    "microimpute_quantreg",
    "microimpute_matching",
    "statmatch_hotdeck",
    "weighted_marginal",
}


def test_registry_exposes_the_committed_method_surface() -> None:
    """The registry lists the PLAN.md method surface, categorized."""
    assert set(REGISTRY) == EXPECTED_METHOD_KEYS
    assert set(list_methods()) == EXPECTED_METHOD_KEYS
    assert CANDIDATE_KEYS == ("populace_fit",)
    assert set(ABLATION_KEYS) == {
        "populace_fit_unweighted",
        "populace_fit_unchained",
        "plain_qrf",
    }
    assert set(BASELINE_KEYS) == EXPECTED_METHOD_KEYS - set(CANDIDATE_KEYS) - set(
        ABLATION_KEYS
    )
    with pytest.raises(KeyError, match="Unknown method"):
        get_method("nope")


def test_registry_imports_without_method_packages() -> None:
    """Constructors are lazy: building a FitFn must not import heavy packages.

    Calling the stub constructors is fine (they return a closure); only calling
    the returned fit without the package installed may raise, and the sweep
    accounts for that as a skip.
    """
    for key in EXPECTED_METHOD_KEYS:
        fit = get_method(key).constructor()
        assert callable(fit)


def test_toy_dataset_has_the_structures_the_metrics_need() -> None:
    """Zero-inflated target carries three signs; weights vary; split is clean."""
    dataset = smoke.make_toy_dataset(seed=0, n=800)
    zi = dataset.train[smoke.ZERO_INFLATED_TARGET]
    assert (zi == 0).any() and (zi > 0).any() and (zi < 0).any()
    assert dataset.train[smoke.WEIGHT_COLUMN].nunique() > 1
    assert len(dataset.train) + len(dataset.test) == 800
    # Deterministic: the same seed reproduces the same table.
    again = smoke.make_toy_dataset(seed=0, n=800)
    pd.testing.assert_frame_equal(dataset.train, again.train)


def test_paired_splits_are_deterministic_partitions() -> None:
    """Same seed => same split; every split is a disjoint, complete partition."""
    frame = smoke.make_toy_dataset(seed=1, n=400).train
    first = split_frame(frame, seed=7)
    second = split_frame(frame, seed=7)
    pd.testing.assert_frame_equal(first.train, second.train)
    pd.testing.assert_frame_equal(first.test, second.test)
    splits = list(paired_splits(frame, seeds=(0, 1)))
    assert [s.seed for s in splits] == [0, 1]
    for split in splits:
        assert len(split.train) + len(split.test) == len(frame)


def test_metric_identities() -> None:
    """Cheap invariants: zero distance/error on identical inputs, finite loss."""
    rng = np.random.default_rng(0)
    values = rng.lognormal(1.0, 0.5, 300)
    weights = rng.uniform(1.0, 5.0, 300)
    assert metrics.weighted_wasserstein1(
        values, values, imputed_weights=weights, donor_weights=weights
    ) == pytest.approx(0.0, abs=1e-9)
    assert metrics.zero_share_error(values, values) == 0.0
    loss = metrics.weighted_pinball_loss(values, values, weights=weights)
    assert math.isfinite(loss) and loss >= 0.0
    with pytest.raises(ValueError, match="non-negative"):
        metrics.weighted_pinball_loss(values, values, weights=-weights)


def test_unimplemented_metrics_refuse_loudly() -> None:
    """The PRDC and fragility stubs raise rather than return silent scores."""
    points = np.zeros((4, 2))
    with pytest.raises(NotImplementedError):
        metrics.prdc_coverage(points, points)
    with pytest.raises(NotImplementedError):
        metrics.reweight_fragility(np.ones(4), np.ones(4))


def test_condition_runs_end_to_end_with_finite_metrics() -> None:
    """The demo path: weighted_marginal through the real condition runner."""
    dataset = smoke.make_toy_dataset(seed=0, n=600)
    result = run_condition(
        "weighted_marginal",
        dataset.train,
        dataset.test,
        dataset.predictors,
        dataset.targets,
        weight_column=dataset.weight_column,
        seed=0,
    )
    expected_keys = {
        f"{smoke.CONTINUOUS_TARGET}.pinball_loss",
        f"{smoke.CONTINUOUS_TARGET}.wasserstein1",
        f"{smoke.ZERO_INFLATED_TARGET}.pinball_loss",
        f"{smoke.ZERO_INFLATED_TARGET}.wasserstein1",
        f"{smoke.ZERO_INFLATED_TARGET}.zero_share_error",
    }
    assert expected_keys <= set(result.metrics)
    assert all(math.isfinite(v) for v in result.metrics.values())
    assert list(result.imputed.columns) == list(dataset.targets)
    assert len(result.imputed) == len(dataset.test)
    rows = rows_from_result(result)
    assert {row["metric"] for row in rows} >= {"pinball_loss", "wasserstein1"}


def test_sweep_writes_artifacts_and_accounts_for_skips(tmp_path) -> None:
    """A toy sweep writes metrics_long.csv; unimplemented methods land in skipped.csv."""
    out = tmp_path / "run"
    code = run_sweep(
        task="toy",
        out=out,
        methods=["weighted_marginal", "populace_fit"],
        n_seeds=2,
    )
    assert code == 0
    long = pd.read_csv(out / "metrics_long.csv")
    assert set(long["method"]) == {"weighted_marginal"}
    assert set(long["seed"]) == {0, 1}
    skipped = pd.read_csv(out / "skipped.csv")
    # populace_fit skips on either frontier: package missing (base install) or
    # adapter pending (methods extra installed) -- both are recorded reasons.
    assert list(skipped["method"]) == ["populace_fit"]

    assert make_figures(out) == 0
    summary = pd.read_csv(out / "summary.csv")
    assert {"method", "target", "metric", "mean", "sd"} <= set(summary.columns)
    assert (out / "summary.tex").read_text().startswith("% Generated")


def test_cli_demo_and_unknown_task(capsys) -> None:
    """`imp demo` exits 0 and prints metrics; a non-toy sweep task refuses."""
    assert main(["demo", "--n", "300"]) == 0
    printed = capsys.readouterr().out
    assert "weighted_marginal" in printed and "pinball_loss" in printed
    with pytest.raises(SystemExit, match="toy"):
        main(["sweep", "--task", "scf_to_cps"])
