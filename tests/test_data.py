"""Real-data loader tests: each loader returns a sweep-ready, invariant TaskFrame.

These tests hit the network and the on-disk cache (Fed SCF download, Census
ASEC download, OpenML fetch), so the whole module is local-only: it skips
under ``CI=true`` rather than downloading survey microdata in CI. Run locally
with::

    uv run python -m pytest tests/test_data.py -q

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
