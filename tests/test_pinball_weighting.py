"""Pinball loss with the imputed quantiles read under receiver weights.

The paper's ``weighted_pinball_loss`` reads the imputed draw's quantiles with
``np.quantile`` (each receiver row once) and scores them against weighted
truth. ``pinball_with_weighted_quantiles`` is the diagnostic variant that
reads them under the receiver weights. Its invariants:

* it is non-negative, scales with the dollars, and ignores the scale of the
  weights and, when the imputed values are distinct, the order of the rows;
* when the imputed draw is constant the two readings coincide, so the variant
  equals the paper's metric exactly (the only difference between them is how
  the imputed quantiles are read);
* with uniform weights the two agree closely (they use different
  interpolation rules, so not to the last digit).

Row order is stated for distinct imputed values only, and that is a property
of the harness's existing ``_weighted_quantile``, not of this variant.
Hypothesis found the counterexample, minimized here: the weighted sample
{0: weight 1, 1: weight 3} has a 30th percentile of 0.70, 0.47, or 0.35
depending on whether the mass at 1 is two rows of weight (1, 2), of (2, 1),
or one row. An interpolating weighted quantile depends on how tied mass is
split into rows. The effect is confined to the interpolation step beside a
tied block; over 200 random row orders of a 4,000-row sample with a quarter
of its mass tied at zero it moved no decile-grid quantile by more than 1e-14.
It is pre-existing, immaterial at the paper's sample sizes, and left alone.

The committed diagnostic (``runs/scf-pinball-weighting``) is checked cell by
cell against the committed within-SCF sweeps.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from imputation_paper.cli.replication import (
    pinball_table,
    pinball_with_weighted_quantiles,
)
from imputation_paper.experiments import metrics

RUNS = Path(__file__).resolve().parents[1] / "runs"

_values = st.integers(min_value=-2000, max_value=20000).map(float)


@st.composite
def _cases(draw: st.DrawFn) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = draw(st.integers(min_value=3, max_value=40))
    truth = np.array(draw(st.lists(_values, min_size=n, max_size=n)))
    # Distinct imputed values: see the module docstring on tied rows.
    drawn = np.array(draw(st.lists(_values, min_size=n, max_size=n, unique=True)))
    weights = np.array(
        draw(st.lists(st.integers(1, 512), min_size=n, max_size=n)), dtype=np.float64
    )
    return truth, drawn, weights


@settings(max_examples=200, deadline=None)
@given(case=_cases(), data=st.data())
def test_invariants(
    case: tuple[np.ndarray, np.ndarray, np.ndarray], data: st.DataObject
) -> None:
    truth, drawn, weights = case
    loss = pinball_with_weighted_quantiles(truth, drawn, weights)
    assert loss >= 0.0

    # Dollars scale the loss; the Fed's re-basing factor is the case in point.
    factor = 4789 / 4376
    rebased = pinball_with_weighted_quantiles(factor * truth, factor * drawn, weights)
    assert rebased == pytest.approx(factor * loss, rel=1e-9, abs=1e-9)

    # Weight scale and row order do not matter (imputed values are distinct).
    order = np.array(data.draw(st.permutations(range(len(truth)))))
    shuffled = pinball_with_weighted_quantiles(
        truth[order], drawn[order], 8.0 * weights[order]
    )
    assert shuffled == pytest.approx(loss, rel=1e-9, abs=1e-9)


@settings(max_examples=200, deadline=None)
@given(case=_cases(), constant=_values)
def test_equals_the_papers_metric_when_the_draw_is_constant(
    case: tuple[np.ndarray, np.ndarray, np.ndarray], constant: float
) -> None:
    """A constant draw has one quantile however it is read."""
    truth, _, weights = case
    drawn = np.full_like(truth, constant)
    assert pinball_with_weighted_quantiles(truth, drawn, weights) == pytest.approx(
        metrics.weighted_pinball_loss(truth, drawn, weights=weights),
        rel=1e-12,
        abs=1e-9,
    )


def test_close_to_the_papers_metric_under_uniform_weights() -> None:
    """With uniform weights only the interpolation rule differs."""
    rng = np.random.default_rng(0)
    truth = rng.lognormal(10, 2, size=4000)
    drawn = rng.lognormal(10, 2, size=4000)
    weights = np.ones(4000)
    variant = pinball_with_weighted_quantiles(truth, drawn, weights)
    paper = metrics.weighted_pinball_loss(truth, drawn, weights=weights)
    assert variant == pytest.approx(paper, rel=1e-3)


def test_weights_change_the_reading_when_they_track_the_value() -> None:
    """The case the diagnostic is about, in miniature.

    Rich rows are oversampled at low weight. A draw that reproduces every
    row's own value has the right weighted distribution, yet read unweighted
    its quantiles are too high, so the paper's metric charges it; read under
    the weights it is charged almost nothing beyond the truth's own spread.
    """
    rng = np.random.default_rng(1)
    poor = rng.lognormal(9, 1, size=3000)
    rich = rng.lognormal(14, 1, size=1000)
    truth = np.concatenate([poor, rich])
    weights = np.concatenate([np.full(3000, 100.0), np.full(1000, 1.0)])
    as_coded = metrics.weighted_pinball_loss(truth, truth, weights=weights)
    weighted = pinball_with_weighted_quantiles(truth, truth, weights)
    assert weighted < 0.5 * as_coded


def test_committed_diagnostic_agrees_with_the_committed_sweeps() -> None:
    pinball = json.loads((RUNS / "scf-pinball-weighting" / "manifest.json").read_text())
    assert pinball["matches_committed_runs"] is True
    frame = pd.DataFrame(pinball["rows"])
    assert len(frame) == 2 * 10 * 9 * 2  # waves x seeds x methods x targets
    for year, run in ((2022, "scf-wealth"), (2025, "scf-wealth-2025")):
        long = pd.read_csv(RUNS / run / "metrics_long.csv")
        committed = long[long["metric"].eq("pinball_loss")].set_index(
            ["method", "seed", "target"]
        )["value"]
        ours = frame[frame["scf_year"].eq(year)].set_index(
            ["method", "seed", "target"]
        )["pinball_as_coded"]
        np.testing.assert_allclose(ours, committed.loc[ours.index], rtol=1e-12)

    # The marginal draw's own row never counts itself.
    table = pinball_table(pinball)
    marginal = table[table["method"].eq("weighted_marginal")]
    assert marginal["seeds_marginal_lower_as_coded"].eq(0).all()
    assert marginal["seeds_marginal_lower_weighted"].eq(0).all()
