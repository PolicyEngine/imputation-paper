"""The paired classifier test of conditional fit, and its committed runs."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from imputation_paper.cli.conditional import (
    GROUPINGS,
    JOINT,
    run_conditional_fit,
)
from imputation_paper.experiments.conditional_fit import (
    ReceiverGroups,
    paired_folds,
    paired_two_sample_auc,
    ratios_to_sample,
    swap_mask,
    within_group_wasserstein1,
)

RUNS = Path(__file__).resolve().parents[1] / "runs"

#: Each committed conditional-fit run, with the sweep runs whose draws it
#: scores again.
COMMITTED = {
    "conditional-fit-scf-wealth": ("scf-wealth", "scf-wealth-added-arms"),
    "conditional-fit-scf-wealth-2025": (
        "scf-wealth-2025",
        "scf-wealth-2025-added-arms",
    ),
    "conditional-fit-cps-components": (
        "cps-components",
        "cps-components-added-arms",
    ),
    **{
        f"conditional-fit-openml-{name}": (
            f"openml-{name}",
            f"openml-{name}-added-arms",
        )
        for name in (
            "space_ga",
            "elevators",
            "brazilian_houses",
            "onlinenewspopularity",
            "abalone",
            "house_sales",
        )
    },
}


def _toy(n: int = 3000, seed: int = 0):
    """Predictor, a target that depends on it, and uneven weights."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    weights = rng.uniform(0.5, 2.0, size=n)

    def draw() -> np.ndarray:
        return 2.0 * x + rng.normal(size=n)

    return x, draw, weights, rng


def test_folds_keep_both_copies_of_a_row_together() -> None:
    n_rows = 53
    seen: list[int] = []
    for train_idx, test_idx in paired_folds(n_rows, n_splits=5, seed=3):
        train_rows = set(train_idx % n_rows)
        test_rows = set(test_idx % n_rows)
        assert not train_rows & test_rows
        # Each side holds a row's true copy and its imputed copy.
        assert sorted(test_idx) == sorted(
            [row for row in test_rows] + [row + n_rows for row in test_rows]
        )
        assert len(train_idx) == 2 * len(train_rows)
        seen.extend(test_rows)
    assert sorted(seen) == list(range(n_rows))


def test_a_draw_from_the_true_conditional_is_not_detected() -> None:
    x, draw, weights, _ = _toy()
    auc = paired_two_sample_auc(x, draw(), draw(), weights=weights)
    assert abs(auc - 0.5) < 0.04


def test_a_draw_that_ignores_the_predictor_is_detected() -> None:
    x, draw, weights, rng = _toy()
    truth = draw()
    # Same marginal distribution as the truth, no relation to the predictor.
    shuffled = rng.permutation(draw())
    auc = paired_two_sample_auc(x, truth, shuffled, weights=weights)
    assert auc > 0.7


def test_identical_files_score_exactly_one_half() -> None:
    x, draw, weights, _ = _toy(n=400)
    truth = draw()
    assert paired_two_sample_auc(x, truth, truth, weights=weights) == 0.5


def test_label_swap_gives_the_null() -> None:
    x, draw, weights, rng = _toy()
    truth = draw()
    shuffled = rng.permutation(draw())
    swap = swap_mask(len(x), seed=0)
    assert 0.4 < swap.mean() < 0.6
    assert np.array_equal(swap, swap_mask(len(x), seed=0))
    null = paired_two_sample_auc(x, truth, shuffled, weights=weights, swap=swap)
    assert abs(null - 0.5) < 0.04


def test_several_targets_are_scored_together() -> None:
    """Two targets with the right marginals each, and the wrong relation."""
    rng = np.random.default_rng(1)
    n = 3000
    x = rng.normal(size=n)
    first = rng.normal(size=n)
    truth = np.column_stack([first, first + 0.1 * rng.normal(size=n)])
    other = rng.normal(size=n)
    imputed = np.column_stack(
        [other, rng.permutation(other + 0.1 * rng.normal(size=n))]
    )
    for column in (0, 1):
        alone = paired_two_sample_auc(x, truth[:, column], imputed[:, column])
        assert abs(alone - 0.5) < 0.05
    assert paired_two_sample_auc(x, truth, imputed) > 0.8


def test_misaligned_inputs_are_refused() -> None:
    x = np.zeros(10)
    with pytest.raises(ValueError, match="same rows"):
        paired_two_sample_auc(x, np.zeros(10), np.zeros(9))
    with pytest.raises(ValueError, match="weights"):
        paired_two_sample_auc(x, np.zeros(10), np.zeros(10), weights=np.ones(3))
    with pytest.raises(ValueError, match="swap"):
        paired_two_sample_auc(x, np.zeros(10), np.zeros(10), swap=np.ones(3, bool))


def test_command_writes_the_sweep_schema(tmp_path: Path) -> None:
    out = tmp_path / "run"
    code = run_conditional_fit(
        task="toy", out=out, methods=["weighted_marginal"], n_seeds=2
    )
    assert code == 0
    rows = list(csv.DictReader((out / "metrics_long.csv").open()))
    assert {row["method"] for row in rows} == {"weighted_marginal"}
    assert {row["seed"] for row in rows} == {"0", "1"}
    joint = {row["metric"] for row in rows if row["target"] == JOINT}
    assert joint == {"conditional_auc", "conditional_auc_null"}
    manifest = json.loads((out / "manifest.json").read_text())
    per_target = {row["target"] for row in rows} - {JOINT}
    assert per_target == set(manifest["targets"])
    assert manifest["test"]["folds"] == 5


def _households(n: int = 4000, seed: int = 0) -> pd.DataFrame:
    """Households whose wealth rises with income, with unequal weights."""
    rng = np.random.default_rng(seed)
    income = rng.lognormal(mean=11.0, sigma=0.8, size=n)
    return pd.DataFrame(
        {
            "income": income,
            "age": rng.integers(20, 90, size=n).astype(float),
            "wealth": income * rng.lognormal(mean=1.0, sigma=0.5, size=n),
            "weight": rng.uniform(1.0, 9.0, size=n),
        }
    )


def test_groups_are_income_bands_by_age_bands() -> None:
    frame = _households()
    groups = ReceiverGroups.from_sample(
        frame,
        income_column="income",
        levels=(0.2, 0.4, 0.6, 0.8),
        weight_column="weight",
    )
    labels = groups.labels(frame)
    assert len(np.unique(labels)) == 15
    # Each income band holds a fifth of the weight.
    band = labels // 3
    shares = np.bincount(band, weights=frame["weight"]) / frame["weight"].sum()
    assert shares == pytest.approx(np.full(5, 0.2), abs=0.005)
    # The age bands are under 40, 40 to 64, and 65 and over.
    assert set(frame["age"][labels % 3 == 0] < 40) == {True}
    assert set(frame["age"][labels % 3 == 2] >= 65) == {True}


def test_rows_with_no_income_form_their_own_band() -> None:
    frame = _households()
    frame.loc[: len(frame) // 4, "income"] = 0.0
    groups = ReceiverGroups.from_sample(
        frame,
        income_column="income",
        levels=(1 / 3, 2 / 3),
        weight_column="weight",
        zero_band=True,
    )
    labels = groups.labels(frame)
    assert len(np.unique(labels)) == 12
    assert set(frame["income"][labels // 3 == 0]) == {0.0}
    assert (frame["income"][labels // 3 > 0] > 0).all()


def test_within_group_distance_separates_what_the_marginal_cannot() -> None:
    """Shuffling wealth across households keeps the marginal and breaks the groups."""
    frame = _households()
    groups = ReceiverGroups.from_sample(
        frame,
        income_column="income",
        levels=(0.2, 0.4, 0.6, 0.8),
        weight_column="weight",
    )
    labels = groups.labels(frame)
    truth = frame["wealth"].to_numpy()
    weights = np.ones(len(frame))
    assert within_group_wasserstein1(truth, truth, weights, labels) == 0.0

    rng = np.random.default_rng(1)
    shuffled = rng.permutation(truth)
    within = np.empty_like(truth)
    for label in np.unique(labels):
        mask = labels == label
        within[mask] = rng.permutation(truth[mask])
    # Shuffling inside the groups leaves every group's distribution as it was.
    assert within_group_wasserstein1(within, truth, weights, labels) == pytest.approx(
        0.0, abs=1e-6
    )
    assert within_group_wasserstein1(shuffled, truth, weights, labels) > 0.2 * (
        truth.mean()
    )


def test_within_group_distance_refuses_misaligned_or_empty_input() -> None:
    with pytest.raises(ValueError, match="align"):
        within_group_wasserstein1(np.zeros(4), np.zeros(5), np.ones(4), np.zeros(4))
    with pytest.raises(ValueError, match="no group"):
        within_group_wasserstein1(
            np.zeros(4), np.zeros(4), np.ones(4), np.arange(4), min_rows=2
        )


def test_ratios_are_one_for_the_sample_itself() -> None:
    frame = _households()
    values = frame["wealth"].to_numpy()
    weights = frame["weight"].to_numpy()
    ratios = ratios_to_sample(values, weights, values, weights)
    assert ratios == pytest.approx(
        {"total_ratio": 1.0, "q90_ratio": 1.0, "q99_ratio": 1.0}
    )
    doubled = ratios_to_sample(2 * values, weights, values, weights)
    assert doubled["total_ratio"] == pytest.approx(2.0)
    assert ratios_to_sample(values, weights, np.zeros(3), np.ones(3)) == {
        "total_ratio": pytest.approx(float("nan"), nan_ok=True),
        "q90_ratio": pytest.approx(float("nan"), nan_ok=True),
        "q99_ratio": pytest.approx(float("nan"), nan_ok=True),
    }


#: The dollar measures a survey task's run carries for each target.
DOLLAR_MEASURES = {
    "within_group_wasserstein1",
    "total_ratio",
    "q90_ratio",
    "q99_ratio",
}


@pytest.mark.parametrize("run", sorted(COMMITTED))
def test_committed_survey_runs_carry_the_dollar_measures(run: str) -> None:
    """Survey runs have the dollar measures for every cell; the others none."""
    path = RUNS / run / "metrics_long.csv"
    if not path.exists():
        pytest.skip(f"{run} is not committed")
    manifest = json.loads((RUNS / run / "manifest.json").read_text())
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    cells: dict[tuple[str, str, str], set[str]] = {}
    for row in rows:
        if row["target"] != JOINT:
            cells.setdefault((row["method"], row["seed"], row["target"]), set()).add(
                row["metric"]
            )
    expected = DOLLAR_MEASURES if manifest["task"] in GROUPINGS else set()
    assert ("dollar_measures" in manifest) == bool(expected)
    for key, metrics in cells.items():
        assert metrics & DOLLAR_MEASURES == expected, key


def _wasserstein(run: str) -> dict[tuple[str, str, str], float]:
    with (RUNS / run / "metrics_long.csv").open() as handle:
        return {
            (row["method"], row["seed"], row["target"]): float(row["value"])
            for row in csv.DictReader(handle)
            if row["metric"] == "wasserstein1"
        }


@pytest.mark.parametrize("run", sorted(COMMITTED))
def test_committed_run_scores_the_draws_behind_the_tables(run: str) -> None:
    """The run's Wasserstein-1 values equal the sweep's, cell for cell.

    The conditional test refits every method. Equal marginal distances show
    that it scored the same draws the paper's tables report.
    """
    if not (RUNS / run / "metrics_long.csv").exists():
        pytest.skip(f"{run} is not committed")
    ours = _wasserstein(run)
    theirs: dict[tuple[str, str, str], float] = {}
    for sweep in COMMITTED[run]:
        theirs.update(_wasserstein(sweep))
    shared = sorted(set(ours) & set(theirs))
    assert shared, "no cell in common with the sweep runs"
    # Every method the sweeps ran is scored here too.
    assert {key[0] for key in theirs} <= {key[0] for key in ours}
    for key in shared:
        assert ours[key] == pytest.approx(theirs[key], rel=1e-9), key
