"""Real-data loader tests: each loader returns a sweep-ready, invariant TaskFrame.

These tests hit the network and the on-disk cache (Fed SCF download, Census
ASEC download, OpenML fetch), so the whole module is local-only: it skips
under ``CI=true`` rather than downloading survey microdata in CI. Run locally
with::

    uv run python -m pytest tests/test_data.py -q

The SCF tests cover both waves: the 2022 wave the paper's experiments use and
the 2025 replication wave, including the weight total, the balance-sheet
identity, and a differential check against the Fed's published Table 2.

Each test asserts the :class:`~imputation_paper.data.base.TaskFrame` contract
(float64 columns, no NaNs in used columns, positive weights, a fresh index, a
real row count, and the predictors/targets present) plus the task-specific
distributional properties the paper leans on (SCF: sign-mixed net worth and
zero-inflated debt; CPS components: zero-inflated interest and dividends). One
test also pushes a loaded frame through the sweep's real split + metric path
with the dependency-free ``weighted_marginal`` baseline, proving the loaders
produce frames the harness can actually score.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

# Network data tests are local-only: they download real survey files (the
# Census ASEC bundle is ~150MB) and never run in CI.
if os.environ.get("CI") == "true":  # pragma: no cover
    pytest.skip("network data tests are local-only", allow_module_level=True)

from imputation_paper.data import (  # noqa: E402 - after importorskip guard
    OPENML_TASKS,
    TaskFrame,
    load_cps,
    load_cps_households,
    load_openml,
    load_scf,
    load_sipp,
)

#: Every loader hits the network on a cold cache; keep the download to once per
#: session by caching the loaded frames across the tests that share them.
_MIN_ROWS = 1000


def _assert_taskframe_invariants(task: TaskFrame) -> None:
    """Assert the contract every loader promises on its returned TaskFrame."""
    frame = task.frame
    used = list(task.used_columns)

    # Predictors and targets are present and disjoint from the weight column.
    for column in used:
        assert column in frame.columns, f"{task.name}: missing used column {column!r}"
    assert task.weight_column not in task.predictors
    assert task.weight_column not in task.targets

    # Every used column is float64 and NaN-free.
    for column in used:
        assert frame[column].dtype == np.float64, (
            f"{task.name}: column {column!r} is {frame[column].dtype}, not float64"
        )
    assert not frame[used].isna().any().any(), f"{task.name}: NaNs in used columns"

    # Weights strictly positive; a real (not toy) row count; a fresh index.
    assert (frame[task.weight_column] > 0).all(), f"{task.name}: non-positive weights"
    assert len(frame) > _MIN_ROWS, f"{task.name}: only {len(frame)} rows"
    assert frame.index.equals(pd.RangeIndex(len(frame))), f"{task.name}: stale index"


@pytest.fixture(scope="module")
def scf_task() -> TaskFrame:
    return load_scf(2022)


@pytest.fixture(scope="module")
def cps_task() -> TaskFrame:
    return load_cps(2025)


@pytest.fixture(scope="module")
def scf_population() -> dict[int, dict[str, float | int]]:
    """Weight totals, identity errors, and net-worth statistics per wave."""
    from imputation_paper.data.scf_checks import population_summary

    return {year: population_summary(year) for year in (2022, 2025)}


def test_load_scf(scf_task: TaskFrame) -> None:
    """SCF wealth: invariants, plus sign-mixed net worth and zero-inflated debt."""
    _assert_taskframe_invariants(scf_task)
    assert scf_task.name == "scf_wealth"
    assert scf_task.weight_column == "wgt"
    assert scf_task.predictors == (
        "age",
        "hhsex",
        "edcl",
        "married",
        "kids",
        "income",
        "wageinc",
    )
    assert scf_task.targets == ("debt", "networth")

    networth = scf_task.frame["networth"]
    debt = scf_task.frame["debt"]
    # The regime-gate story: net worth spans both signs; debt has a zero mass.
    assert (networth < 0).any(), "net worth should include underwater households"
    assert (networth > 0).any(), "net worth should include positive households"
    assert (debt == 0).any(), "debt should be zero-inflated"
    assert (debt >= 0).all(), "debt should be nonnegative"

    # The int64 implicate filter keeps all households, not just the low-id ones
    # the naive int16 arithmetic would (the extract has 4595 households).
    assert len(scf_task.frame) == 4595


def test_load_scf_2025() -> None:
    """The replication wave loads through the same contract as 2022."""
    task = load_scf(2025)
    _assert_taskframe_invariants(task)
    assert task.name == "scf_wealth"
    assert task.targets == ("debt", "networth")
    assert len(task.frame) == 4360  # 21,800 rows / five implicates
    assert task.provenance["dollar_year"] == 2025
    assert task.provenance["rebase_factor"] == 1.0
    networth, debt = task.frame["networth"], task.frame["debt"]
    assert (networth < 0).any() and (networth > 0).any()
    assert (debt == 0).any() and (debt >= 0).all()

    rich = load_scf(2025, profile="populace-scale")
    _assert_taskframe_invariants(rich)
    assert set(rich.frame["race"].unique()) == {1.0, 2.0, 3.0, 4.0, 5.0}


def test_paper_2022_inputs_are_unchanged() -> None:
    """``load_scf(2022)`` still returns the 2022-dollar frame the runs used.

    The Fed re-issued the 2022 file in 2025 dollars on 2026-10-09; whichever
    copy is cached, the default load is on the 2022 basis.
    """
    task = load_scf(2022)
    assert task.provenance["dollar_year"] == 2022
    weight = task.frame["wgt"].sum()
    assert weight == pytest.approx(131.3e6 / 5, rel=1e-3)  # implicate 1 only
    rebased = load_scf(2022, dollar_year=2025)
    np.testing.assert_allclose(
        rebased.frame["networth"], task.frame["networth"] * 4789 / 4376, rtol=1e-12
    )
    np.testing.assert_array_equal(rebased.frame["wgt"], task.frame["wgt"])


@pytest.mark.parametrize("year", [2022, 2025])
def test_scf_weight_total_and_balance_sheet_identity(
    scf_population: dict[int, dict[str, float | int]], year: int
) -> None:
    """Weights sum to the families represented; the balance sheet adds up.

    ``wgt`` is the analysis weight divided by five, so summed over all five
    implicates it is the number of families the survey represents: 131.3
    million (2022) and 135.0 million (2025) in the Fed's Table B.1, published
    to the nearest 0.1 million. The identity networth = fin + nfin - debt,
    which the populace-scale chaining result relies on, must hold on every
    row of every implicate (to storage rounding: some 2022 columns are
    stored as 32-bit numbers).
    """
    from imputation_paper.data.scf_checks import PUBLISHED_FAMILIES_MILLIONS

    wave = scf_population[year]
    assert wave["implicates"] == 5
    assert wave["rows"] == 5 * wave["families"]
    assert wave["families"] == {2022: 4595, 2025: 4360}[year]
    published = PUBLISHED_FAMILIES_MILLIONS[year] * 1e6
    assert wave["weight_total"] == pytest.approx(published, abs=0.05e6)
    assert wave["net_worth_identity_max_abs_error"] < 0.01
    assert wave["asset_identity_max_abs_error"] < 0.01


def test_scf_net_worth_matches_the_feds_table_2(
    scf_population: dict[int, dict[str, float | int]],
) -> None:
    """All-family median and mean net worth against the Fed's Table 2.

    Differential check: the public extract, pooled over its five implicates
    under ``wgt``, against the published figures in thousands of 2025 dollars
    (2022: median 211.1, mean 1,164.1; 2025: median 215.9, mean 1,241.5).

    A gap is expected and is not an error in the loader. The Fed computes its
    tables from the internal data set. Per each wave's codebook ("Disclosure
    review"), the public file rounds continuous variables, blurs and
    re-imputes some values, and removes the families whose net worth reaches
    the minimum for the Forbes 400 list: seven in each wave (4,595 of 4,602
    families in 2022; 4,360 of 4,367 in 2025). Removing the very top is
    consistent with the public mean sitting below the published one in both
    waves, while the median barely moves. Rather than guess the gap, the
    tolerance on 2025 is calibrated on 2022, the wave the paper used: the
    larger of one percent and twice the 2022 gap. On the 2022 file the gaps
    are -0.10 percent (median) and -0.40 percent (mean), so both 2025
    tolerances are one percent.
    """
    from imputation_paper.data.scf_checks import published_gaps

    gaps = published_gaps(scf_population[2022], scf_population[2025])
    for statistic in ("net_worth_median", "net_worth_mean", "weight_total"):
        gap = gaps[statistic]
        # The calibration wave itself sits close to the published figure.
        assert abs(gap["gap_calibration"]) < 0.01, statistic
        assert gap["within_tolerance"], (
            f"{statistic}: 2025 public {gap['public_replication']:,.0f} vs "
            f"published {gap['published_replication']:,.0f} "
            f"({100 * gap['gap_replication']:+.2f}%; tolerance "
            f"{100 * gap['tolerance']:.2f}%)"
        )


def test_load_cps_components(cps_task: TaskFrame) -> None:
    """CPS components: invariants, plus zero-inflated interest and dividends."""
    _assert_taskframe_invariants(cps_task)
    assert cps_task.name == "cps_components"
    assert cps_task.weight_column == "person_weight"
    assert cps_task.predictors == ("age", "is_female", "employment_income")
    assert cps_task.targets == ("interest_income", "dividend_income")

    # Adults only.
    assert (cps_task.frame["age"] >= 18).all()

    # Weight scale: adult person weights should sum to the adult US population
    # (~260M), pinning the ASEC implied-decimal handling.
    adult_population = cps_task.frame["person_weight"].sum()
    assert 2.0e8 < adult_population < 3.2e8, f"{adult_population:,.0f}"

    for target in cps_task.targets:
        column = cps_task.frame[target]
        zero_share = float((column == 0).mean())
        assert zero_share > 0.30, (
            f"{target}: zero-share {zero_share:.3f} too low to be zero-inflated"
        )
        assert (column > 0).any(), f"{target}: needs a positive tail"
        assert (column >= 0).all(), f"{target}: income components are nonnegative"

    # Dividends are the strongly zero-inflated component (majority zero among
    # adults); interest is more broadly held but still substantially zero.
    dividend_zero = float((cps_task.frame["dividend_income"] == 0).mean())
    assert dividend_zero > 0.50, (
        f"dividend zero-share {dividend_zero:.3f} unexpectedly low"
    )


def test_load_cps_households() -> None:
    """The SCF->CPS receiver: one household row per household, SCF-coded columns."""
    households = load_cps_households(2025)
    shared = ["age", "hhsex", "married", "kids", "income", "wageinc"]
    assert list(households.columns) == [*shared, "household_weight"]

    assert (households.dtypes == np.float64).all()
    assert not households.isna().any().any()
    assert (households["household_weight"] > 0).all()
    assert len(households) > _MIN_ROWS
    # Weight scale: ~120-145M US households.
    household_population = households["household_weight"].sum()
    assert 1.1e8 < household_population < 1.5e8, f"{household_population:,.0f}"
    assert households.index.equals(pd.RangeIndex(len(households)))

    # SCF codings: hhsex in {1, 2}, married in {1, 2}; kids nonnegative.
    assert set(households["hhsex"].unique()) <= {1.0, 2.0}
    assert set(households["married"].unique()) <= {1.0, 2.0}
    assert (households["kids"] >= 0).all()
    # Both marital states and both sexes are represented (the derivation is not
    # collapsing everyone into one bucket).
    assert set(households["hhsex"].unique()) == {1.0, 2.0}
    assert set(households["married"].unique()) == {1.0, 2.0}


@pytest.mark.parametrize("name", sorted(OPENML_TASKS))
def test_load_openml(name: str) -> None:
    """Each OpenML task: invariants, expected row count, constant unit weight."""
    task = load_openml(name)
    _assert_taskframe_invariants(task)
    assert task.name == f"openml_{name}"
    assert task.weight_column == "weight"
    assert (task.frame["weight"] == 1.0).all()

    # Row count matches the pinned expectation (guards a silent version change).
    assert len(task.frame) == OPENML_TASKS[name]["rows"]

    # Predictor set: numeric, capped at 12, target excluded, no duplicates.
    assert 1 <= len(task.predictors) <= 12
    assert len(set(task.predictors)) == len(task.predictors)
    assert len(task.targets) == 1
    assert task.targets[0] not in task.predictors


def test_load_sipp_raises_when_file_absent() -> None:
    """SIPP is skipped honestly: a missing file raises, never fabricates data.

    The expected SIPP path is a dangling symlink on this machine, so the loader
    must raise :class:`FileNotFoundError` rather than substitute synthetic data.
    (If a real SIPP file is later dropped in place this test is skipped.)
    """
    from imputation_paper.data.sipp import SIPP_PATH

    if SIPP_PATH.exists():  # pragma: no cover - real file present
        pytest.skip(f"SIPP file {SIPP_PATH} is present; absence test not applicable.")
    with pytest.raises(FileNotFoundError, match="not present"):
        load_sipp()


def test_loaded_frame_runs_through_the_sweep(scf_task: TaskFrame) -> None:
    """A loaded TaskFrame is sweep-ready: split it and score a real baseline.

    Uses the dependency-free ``weighted_marginal`` baseline through the real
    condition path, so this proves the loaders' output plugs into the harness
    without needing the ``methods`` extra.
    """
    from imputation_paper.experiments.conditions import run_condition
    from imputation_paper.experiments.holdout import split_frame

    split = split_frame(scf_task.frame, holdout_frac=0.2, seed=0)
    result = run_condition(
        "weighted_marginal",
        split.train,
        split.test,
        scf_task.predictors,
        scf_task.targets,
        weight_column=scf_task.weight_column,
        seed=0,
    )
    assert list(result.imputed.columns) == list(scf_task.targets)
    assert len(result.imputed) == len(split.test)
    assert result.imputed.notna().all().all()
    assert all(np.isfinite(value) for value in result.metrics.values())
