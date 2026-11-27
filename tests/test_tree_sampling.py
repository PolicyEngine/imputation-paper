"""The per-tree sampling diagnostic and the fixed-donor seed-stability runs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from imputation_paper.cli.stability import FAR_FROM_MEDIAN, summarize
from imputation_paper.cli.tables import PER_TREE_SUFFIX, _read_long
from imputation_paper.methods import (
    DIAGNOSTICS,
    REGISTRY,
    MethodCategory,
    get_method,
    list_methods,
)

RUNS = Path(__file__).resolve().parents[1] / "runs"
PER_TREE = "populace_fit_per_tree"


def test_the_variant_is_a_diagnostic_outside_the_registry() -> None:
    assert PER_TREE in DIAGNOSTICS
    assert PER_TREE not in REGISTRY
    assert PER_TREE not in list_methods()
    assert get_method(PER_TREE).category == MethodCategory.DIAGNOSTIC
    with pytest.raises(KeyError, match="diagnostic variants"):
        get_method("nope")


def _two_groups(n: int = 3000) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Half the rows hold 2.0 at weight 1, half hold 1.0 at weight 9."""
    y = np.r_[np.full(n // 2, 1.0), np.full(n // 2, 2.0)]
    weights = np.r_[np.full(n // 2, 9.0), np.full(n // 2, 1.0)]
    x = np.zeros((n, 1))
    return x, y, weights


def test_each_tree_samples_rows_in_proportion_to_weight() -> None:
    qrf = pytest.importorskip("populace.fit.qrf")
    from imputation_paper.experiments.tree_sampling import per_tree_weighted_sampling

    x, y, weights = _two_groups()
    with per_tree_weighted_sampling():
        forest = qrf._fit_forest(
            x,
            y,
            ("x",),
            weights,
            seed=0,
            n_estimators=40,
            max_samples_leaf=None,
            rng=np.random.default_rng(0),
        )
    # A tree's root value is the mean of the target over the rows it drew.
    # The weighted mean is 1.1 and the unweighted mean 1.5.
    roots = [float(tree.tree_.value[0].ravel()[0]) for tree in forest.model.estimators_]
    assert np.mean(roots) == pytest.approx(1.1, abs=0.01)
    # No tree saw the unweighted mix: every root mean is nearer 1.1 than 1.5.
    assert max(roots) < 1.3


def test_the_block_restores_the_package_and_keeps_the_generator_in_step() -> None:
    qrf = pytest.importorskip("populace.fit.qrf")
    import quantile_forest._quantile_forest as qf_module
    from sklearn.ensemble import _forest as sk_forest

    from imputation_paper.experiments.tree_sampling import per_tree_weighted_sampling

    before = (
        qrf._fit_forest,
        sk_forest._generate_sample_indices,
        qf_module._generate_sample_indices,
    )
    x, y, weights = _two_groups(400)
    kwargs = dict(seed=3, n_estimators=5, max_samples_leaf=None)

    shipped_rng = np.random.default_rng(7)
    qrf._fit_forest(x, y, ("x",), weights, rng=shipped_rng, **kwargs)
    variant_rng = np.random.default_rng(7)
    with per_tree_weighted_sampling():
        assert qrf._fit_forest is not before[0]
        qrf._fit_forest(x, y, ("x",), weights, rng=variant_rng, **kwargs)

    after = (
        qrf._fit_forest,
        sk_forest._generate_sample_indices,
        qf_module._generate_sample_indices,
    )
    assert after == before
    # The variant leaves the fit's generator where the shipped code leaves it,
    # so the gates and forests fitted afterwards keep their seeds.
    assert shipped_rng.integers(1 << 30) == variant_rng.integers(1 << 30)


def test_an_unweighted_forest_is_fitted_as_shipped() -> None:
    qrf = pytest.importorskip("populace.fit.qrf")
    from imputation_paper.experiments.tree_sampling import per_tree_weighted_sampling

    x, y, _ = _two_groups(400)
    kwargs = dict(seed=3, n_estimators=5, max_samples_leaf=None)
    shipped = qrf._fit_forest(
        x, y, ("x",), None, rng=np.random.default_rng(1), **kwargs
    )
    with per_tree_weighted_sampling():
        variant = qrf._fit_forest(
            x, y, ("x",), None, rng=np.random.default_rng(1), **kwargs
        )
    quantiles = [0.1, 0.5, 0.9]
    assert np.array_equal(
        shipped.model.predict(x[:5], quantiles=quantiles),
        variant.model.predict(x[:5], quantiles=quantiles),
    )


def test_summary_pools_the_spread_within_splits_and_counts_far_repeats() -> None:
    long = pd.DataFrame(
        {
            "split": [0] * 4 + [1] * 4,
            "arm": "populace_fit",
            "kind": "refit",
            "target": "networth",
            "metric": "q99_ratio",
            # Split 0 has one repeat far from its median; split 1 has none.
            "value": [1.0, 1.0, 1.0, 2.0, 0.5, 0.5, 0.5, 0.5],
        }
    )
    row = summarize(long).iloc[0]
    assert row["repeats"] == 8
    assert row["far_from_median"] == 1
    expected = np.sqrt((np.var([1.0, 1.0, 1.0, 2.0], ddof=1) + 0.0) / 2)
    assert row["pooled_sd"] == pytest.approx(expected)
    assert FAR_FROM_MEDIAN == 0.25


@pytest.mark.parametrize(
    "run",
    [
        "conditional-fit-scf-wealth",
        "conditional-fit-scf-wealth-2025",
        "conditional-fit-cps-components",
    ],
)
def test_committed_variant_runs_share_the_estimators_draws(run: str) -> None:
    """The variant's run reran the estimator as a control, with identical rows."""
    if not (RUNS / f"{run}-per-tree" / "metrics_long.csv").exists():
        pytest.skip(f"{run}-per-tree is not committed")
    # _read_long raises unless the control rows agree and the seeds match.
    merged = _read_long(RUNS / run, "metrics_long.csv", suffix=PER_TREE_SUFFIX)
    assert merged is not None
    variant = merged[merged["method"] == PER_TREE]
    assert set(variant["seed"]) == set(merged["seed"])
    added = pd.read_csv(RUNS / f"{run}-per-tree" / "metrics_long.csv")
    assert set(added["method"]) == {"populace_fit", PER_TREE}


@pytest.mark.parametrize("run", ["seed-stability", "seed-stability-2025"])
def test_committed_stability_runs_have_every_cell(run: str) -> None:
    path = RUNS / run / "stability_long.csv"
    if not path.exists():
        pytest.skip(f"{run} is not committed")
    long = pd.read_csv(path)
    assert set(long["arm"]) == {"populace_fit", PER_TREE}
    assert set(long["split"]) == {0, 1, 2}
    counts = long.groupby(["arm", "split", "kind", "target", "metric"]).size()
    assert set(counts) == {20}
    summary = pd.read_csv(RUNS / run / "stability_summary.csv")
    pd.testing.assert_frame_equal(
        summary, summarize(long), check_exact=False, rtol=1e-9
    )
