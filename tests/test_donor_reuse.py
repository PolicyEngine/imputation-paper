"""Donor-reuse summary: the aggregate behind hot deck's 2025 tail result.

``donor_reuse_summary`` reports how much receiver weight leans on single
donors, overall and above a threshold. Its invariants:

* every share lies in [0, 1], and the top tail donor's share never exceeds
  the tail's share or the top donor's share;
* a donor whose receivers carry more than a fraction ``1 - q`` of the weight,
  by more than the largest single receiver's share, puts the matched file's
  weighted ``q``-quantile at or above that donor's value (the fact the
  appendix's explanation rests on). The margin matters: the harness's
  quantile interpolates, and without it the claim is false;
* nothing depends on row order, on the scale of the weights, or on how donors
  are labelled.

The committed diagnostic (``runs/scf-hotdeck-reuse``) is checked for
agreement with the harness runs it explains.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from imputation_paper.cli import replication as report_cli
from imputation_paper.experiments.donor_reuse import donor_reuse_summary
from imputation_paper.experiments.metrics import _weighted_quantile

RUNS = Path(__file__).resolve().parents[1] / "runs"


def test_worked_example() -> None:
    """Four receivers, three donors, one donor reused in the tail."""
    summary = donor_reuse_summary(
        donor_index=np.array([0, 1, 1, 2]),
        receiver_weights=np.array([1.0, 2.0, 3.0, 4.0]),
        donated=np.array([10.0, 500.0, 500.0, 900.0]),
        threshold=100.0,
    )
    assert summary == {
        "donors_used": 3,
        "top_donor_weight_share": 0.5,  # donor 1 carries 5 of 10
        "tail_weight_share": 0.9,
        "tail_donors": 2,
        "top_tail_donor_weight_share": 0.5,
        "max_receiver_weight_share": 0.4,
    }


def test_empty_tail_and_bad_inputs() -> None:
    summary = donor_reuse_summary(
        np.array([0, 1]), np.array([1.0, 1.0]), np.array([1.0, 2.0]), threshold=5.0
    )
    assert summary["tail_weight_share"] == 0.0
    assert summary["tail_donors"] == 0
    assert summary["top_tail_donor_weight_share"] == 0.0
    with pytest.raises(ValueError, match="align"):
        donor_reuse_summary(np.array([0]), np.array([1.0, 1.0]), np.array([1.0]), 0.0)
    with pytest.raises(ValueError, match="non-negative"):
        donor_reuse_summary(np.array([0]), np.array([-1.0]), np.array([1.0]), 0.0)


@st.composite
def _matched_files(draw: st.DrawFn) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    n_donors = draw(st.integers(min_value=1, max_value=8))
    values = draw(
        st.lists(
            st.integers(min_value=-50, max_value=200),
            min_size=n_donors,
            max_size=n_donors,
        )
    )
    n = draw(st.integers(min_value=1, max_value=40))
    index = np.array(
        draw(st.lists(st.integers(0, n_donors - 1), min_size=n, max_size=n))
    )
    weights = np.array(
        draw(st.lists(st.integers(1, 64), min_size=n, max_size=n)), dtype=np.float64
    )
    threshold = float(draw(st.integers(min_value=-60, max_value=210)))
    return index, weights, np.array(values, dtype=np.float64)[index], threshold


@settings(max_examples=200, deadline=None)
@given(case=_matched_files(), data=st.data())
def test_shares_are_bounded_ordered_and_label_free(
    case: tuple[np.ndarray, np.ndarray, np.ndarray, float], data: st.DataObject
) -> None:
    index, weights, donated, threshold = case
    summary = donor_reuse_summary(index, weights, donated, threshold)
    for key in (
        "top_donor_weight_share",
        "tail_weight_share",
        "top_tail_donor_weight_share",
    ):
        assert 0.0 <= summary[key] <= 1.0
    assert (
        summary["top_tail_donor_weight_share"] <= summary["tail_weight_share"] + 1e-12
    )
    assert (
        summary["top_tail_donor_weight_share"]
        <= summary["top_donor_weight_share"] + 1e-12
    )
    assert summary["tail_donors"] <= summary["donors_used"] <= len(index)
    assert summary["top_donor_weight_share"] >= 1.0 / summary["donors_used"] - 1e-12

    # Row order, weight scale (a power of two: exact), and donor labels.
    order = np.array(data.draw(st.permutations(range(len(index)))))
    relabelled = (index * 7 + 3)[order]
    again = donor_reuse_summary(
        relabelled, 4.0 * weights[order], donated[order], threshold
    )
    assert again == pytest.approx(summary, rel=1e-12, abs=1e-12)


@settings(max_examples=300, deadline=None)
@given(case=_matched_files(), level=st.sampled_from([0.5, 0.9, 0.99]))
def test_a_heavy_tail_donor_sets_the_quantile(
    case: tuple[np.ndarray, np.ndarray, np.ndarray, float], level: float
) -> None:
    """A donor with enough of the weight sets a floor under the quantile.

    If one donor's receivers carry a weight share above ``1 - level`` plus
    the largest single receiver's share, the matched file's weighted
    ``level``-quantile is at least that donor's value. With the donor above
    the holdout's q99, the imputed q99 is therefore above it too: the
    appendix's explanation.
    """
    index, weights, donated, _ = case
    total = weights.sum()
    shares = pd.Series(weights).groupby(index).sum() / total
    donor = shares.idxmax()
    if shares[donor] <= (1 - level) + weights.max() / total:
        return
    value = donated[index == donor][0]
    quantile = float(_weighted_quantile(donated, weights, np.array([level]))[0])
    assert quantile >= value - 1e-9


def test_the_one_receiver_margin_is_needed() -> None:
    """Without the margin the claim fails: the quantile interpolates.

    One donor holds 1.01 percent of the weight at a value of 100, the rest sit
    at 0, and the interpolated q99 is 99.01: below the donor's value, and
    below a holdout q99 of, say, 99.5.
    """
    values, weights = np.array([0.0, 100.0]), np.array([98.99, 1.01])
    quantile = float(_weighted_quantile(values, weights, np.array([0.99]))[0])
    assert quantile == pytest.approx(99.01)
    summary = donor_reuse_summary(np.array([0, 1]), weights, values, threshold=99.5)
    share, margin = (
        summary["top_tail_donor_weight_share"],
        summary["max_receiver_weight_share"],
    )
    assert share > 0.01  # more than one percent of the weight ...
    assert share <= 0.01 + margin  # ... but not by a receiver's share
    assert quantile < 99.5


def test_committed_diagnostic_agrees_with_the_harness_runs() -> None:
    """The reuse run recomputed the committed q99 ratios, seed for seed."""
    reuse = json.loads((RUNS / "scf-hotdeck-reuse" / "manifest.json").read_text())
    assert reuse["matches_committed_runs"] is True
    assert reuse["largest_difference_from_committed_q99_ratio"] < 1e-9
    frame = pd.DataFrame(reuse["rows"])
    assert len(frame) == 40  # two profiles x two waves x ten seeds
    for (profile, year), group in frame.groupby(["profile", "scf_year"]):
        stem = "scf-to-cps-harness" + ("-scale" if profile == "scale" else "")
        suffix = "" if year == 2022 else f"-{year}"
        long = pd.read_csv(RUNS / f"{stem}{suffix}" / "harness_long.csv")
        committed = long[
            long["method"].eq("statmatch_hotdeck")
            & long["metric"].eq("q99_ratio.networth")
        ].set_index("seed")["value"]
        np.testing.assert_allclose(
            group.set_index("seed")["q99_ratio"], committed, rtol=1e-12
        )

    # Every tail-inflated seed has one tail donor past the sufficient
    # condition (one percent of the weight plus the largest receiver's share);
    # no other seed does. That is the whole of the explanation.
    inflated = frame["q99_ratio"] > report_cli.REUSE_INFLATED_RATIO
    heavy = frame["top_tail_donor_weight_share"] > (
        1 - report_cli.REUSE_QUANTILE + frame["max_receiver_weight_share"]
    )
    assert (inflated == heavy).all()
    assert int(inflated.sum()) == 9  # eight at scale and one minimal, all 2025
