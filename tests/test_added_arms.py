"""The arms added to the method surface after the first sweeps.

Three groups, each with a claim the paper leans on:

* **Gates off.** ``populace_fit_ungated`` must differ from the full estimator
  only in that no sign gate is built: every target is fitted through the
  estimator's own single-sign path. Where a target already has one sign class
  the two must coincide exactly.
* **Derived net worth.** The arm models the three balance-sheet components
  and sets net worth by the accounting identity, so the identity holds on
  every drawn row; a task without the four columns is skipped, not faked.
* **Hot decks that use the weights.** The weighted random hot deck picks among
  the nearest donors in proportion to their weights; the constrained hot deck
  never uses a donor more often than its cap.

The pure helpers run on the base install. Tests that need a method package
skip without it, as CI does.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
import pytest

from imputation_paper import methods
from imputation_paper.methods import (
    ABLATION_KEYS,
    BASELINE_KEYS,
    VARIANT_KEYS,
    MethodCategory,
    get_method,
    single_sign_shift,
)

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_added_arms_are_registered_under_the_right_categories() -> None:
    assert {"populace_fit_ungated", "populace_fit_ungated_unchained"} <= set(
        ABLATION_KEYS
    )
    assert set(VARIANT_KEYS) == {
        "populace_fit_derived",
        "populace_fit_unchained_derived",
    }
    assert {
        "statmatch_rand_hotdeck_unweighted",
        "statmatch_rand_hotdeck",
        "statmatch_constrained_hotdeck",
    } <= set(BASELINE_KEYS)
    assert get_method("populace_fit_derived").category == MethodCategory.VARIANT


# ---------------------------------------------------------------------------
# Gates off: the shift that leaves one sign class
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ([0.0, 0.0, 5.0, 12.5], 1.0),  # zero-inflated, non-negative
        ([-40.0, 0.0, 7.0], 41.0),  # three sign classes
        ([-3.0, 2.0], 4.0),  # both signs, no zero
        ([-9.0, 0.0, -1.0], 10.0),  # zero-inflated, non-positive
        ([2.0, 7.5, 1e-3], 0.0),  # already strictly positive
        ([-2.0, -7.5], 0.0),  # already strictly negative
        ([0.0, 0.0], 0.0),  # constant zero: nothing to gate
    ],
)
def test_single_sign_shift(values: list[float], expected: float) -> None:
    y = np.array(values)
    shift = single_sign_shift(y)
    assert shift == expected
    if shift:
        # One class left, and it is the positive one.
        assert (y + shift >= 1.0).all()


def test_shift_round_trip_keeps_exact_zeros() -> None:
    """A zero moved by the shift and back is exactly zero again."""
    y = np.array([-1_234_567.89, 0.0, 0.0, 3.5e9])
    shift = single_sign_shift(y)
    assert ((y + shift) - shift)[1:3].tolist() == [0.0, 0.0]


# ---------------------------------------------------------------------------
# Derived net worth
# ---------------------------------------------------------------------------


def _balance_sheet(n: int = 40, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(
        {
            "age": rng.integers(20, 80, n).astype(float),
            "fin": rng.gamma(2.0, 1000.0, n),
            "nfin": rng.gamma(2.0, 5000.0, n),
            "debt": rng.gamma(1.0, 2000.0, n),
            "weight": rng.uniform(1.0, 9.0, n),
        }
    )
    frame["networth"] = frame["fin"] + frame["nfin"] - frame["debt"]
    return frame


class _RecordingBase:
    """A stand-in method: returns donor means and records what it was asked."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def __call__(self):
        def fit(
            train_df: pd.DataFrame,
            predictors: Sequence[str],
            targets: Sequence[str],
            weights,
            *,
            seed: int = 0,
        ):
            self.calls.append(
                {
                    "predictors": list(predictors),
                    "targets": list(targets),
                    "weights": weights,
                    "seed": seed,
                }
            )
            means = {target: float(train_df[target].mean()) for target in targets}

            def predict(test_df: pd.DataFrame) -> pd.DataFrame:
                return pd.DataFrame(means, index=test_df.index)

            return predict

        return fit


def _derived_fit(base: _RecordingBase):
    return methods._derived_target_constructor(
        base, derived="networth", components=methods.SCF_NET_WORTH_COMPONENTS
    )


def test_derived_arm_models_the_components_and_applies_the_identity() -> None:
    donors, receivers = _balance_sheet(seed=1), _balance_sheet(n=7, seed=2)
    base = _RecordingBase()
    targets = ["fin", "nfin", "debt", "networth"]
    predict = _derived_fit(base)(donors, ["age"], targets, "weight", seed=5)
    drawn = predict(receivers)

    assert base.calls == [
        {
            "predictors": ["age"],
            "targets": ["fin", "nfin", "debt"],
            "weights": "weight",
            "seed": 5,
        }
    ]
    assert list(drawn.columns) == targets
    assert drawn.index.equals(receivers.index)
    np.testing.assert_array_equal(
        drawn["networth"].to_numpy(),
        (drawn["fin"] + drawn["nfin"] - drawn["debt"]).to_numpy(),
    )


def test_derived_arm_is_skipped_where_the_components_are_absent() -> None:
    """Two-target tasks carry no balance sheet: the arm must refuse, loudly."""
    donors = _balance_sheet()
    with pytest.raises(NotImplementedError, match="fin.*nfin"):
        _derived_fit(_RecordingBase())(donors, ["age"], ["debt", "networth"], "weight")


def test_derived_arm_refuses_a_donor_file_that_breaks_the_identity() -> None:
    donors = _balance_sheet()
    donors.loc[3, "networth"] += 1.0
    with pytest.raises(ValueError, match="identity"):
        _derived_fit(_RecordingBase())(
            donors, ["age"], ["fin", "nfin", "debt", "networth"], "weight"
        )


# ---------------------------------------------------------------------------
# Gates off, against the estimator itself (needs the methods extra)
# ---------------------------------------------------------------------------


def _zero_inflated(n: int = 600, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    positive = rng.random(n) < 1.0 / (1.0 + np.exp(-x))
    return pd.DataFrame(
        {
            "x": x,
            "income": np.where(positive, np.exp(x + rng.normal(size=n)), 0.0),
            "wealth": x * 10.0 + rng.normal(size=n) * 5.0,  # both signs
            "price": np.exp(x) + 1.0,  # strictly positive
            "weight": rng.uniform(1.0, 5.0, n),
        }
    )


def test_ungated_arm_builds_no_gate() -> None:
    fit_module = pytest.importorskip("populace.fit")
    assert methods.ZERO_ATOL == fit_module.DEFAULT_ZERO_ATOL

    donors = _zero_inflated(seed=1)
    table = donors.loc[:, ["x", "income", "wealth"]]
    gated = fit_module.RegimeGatedQRF(seed=0).fit(
        table, ["x"], ["income", "wealth"], weights=donors["weight"].to_numpy()
    )
    assert set(gated.regimes().values()) == {
        fit_module.Regime.ZERO_INFLATED_POSITIVE,
        fit_module.Regime.SIGN_ONLY,
    }

    regimes = methods._ungated_regimes(
        donors, ["x"], ["income", "wealth"], "weight", seed=0
    )
    assert set(regimes.values()) == {fit_module.Regime.POSITIVE_ONLY}


@pytest.mark.parametrize(
    "key", ["populace_fit_ungated", "populace_fit_ungated_unchained"]
)
def test_ungated_arm_draws_on_the_original_scale(key: str) -> None:
    pytest.importorskip("populace.fit")
    donors, receivers = _zero_inflated(seed=1), _zero_inflated(n=300, seed=2)
    targets = ["income", "wealth"]
    fit = get_method(key).constructor()
    drawn = fit(donors, ["x"], targets, "weight", seed=3)(receivers)

    assert list(drawn.columns) == targets
    assert drawn.index.equals(receivers.index)
    assert np.isfinite(drawn.to_numpy()).all()
    # Back on the original scale: within the donors' observed range, zeros
    # exact where the forest reads a zero, both signs of wealth present.
    for target in targets:
        assert drawn[target].min() >= donors[target].min() - 1e-9
        assert drawn[target].max() <= donors[target].max() + 1e-9
    assert (drawn["income"] == 0.0).any() and (drawn["income"] > 0.0).any()
    assert (drawn["wealth"] < 0.0).any() and (drawn["wealth"] > 0.0).any()

    again = fit(donors, ["x"], targets, "weight", seed=3)(receivers)
    pd.testing.assert_frame_equal(drawn, again)


def test_ungated_arm_equals_the_full_estimator_on_a_single_sign_target() -> None:
    """With one sign class the estimator builds no gate, so nothing is knocked out."""
    pytest.importorskip("populace.fit")
    donors, receivers = _zero_inflated(seed=1), _zero_inflated(n=200, seed=2)
    full = get_method("populace_fit").constructor()
    ungated = get_method("populace_fit_ungated").constructor()
    pd.testing.assert_frame_equal(
        full(donors, ["x"], ["price"], "weight", seed=4)(receivers),
        ungated(donors, ["x"], ["price"], "weight", seed=4)(receivers),
    )


def test_single_sign_shift_agrees_with_the_estimators_regime_detection() -> None:
    qrf = pytest.importorskip("populace.fit.qrf")
    single = {
        qrf.Regime.POSITIVE_ONLY,
        qrf.Regime.NEGATIVE_ONLY,
        qrf.Regime.DEGENERATE_ZERO,
    }
    rng = np.random.default_rng(0)
    for _ in range(200):
        signs = rng.choice([-1.0, 0.0, 1.0], size=rng.integers(1, 6))
        y = signs * rng.uniform(0.5, 10.0, size=signs.size)
        regime = qrf.detect_regime(y, zero_atol=methods.ZERO_ATOL)
        assert (single_sign_shift(y) == 0.0) == (regime in single)
        shifted = qrf.detect_regime(
            y + single_sign_shift(y), zero_atol=methods.ZERO_ATOL
        )
        assert shifted in single


# ---------------------------------------------------------------------------
# Hot decks that use the weights (need the methods extra)
# ---------------------------------------------------------------------------


def _two_near_donors() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Receivers at the origin; two equally near donors with weights 9 and 1.

    Sixteen donors in all, so the random hot deck keeps the four nearest; the
    two extra near donors carry no weight worth mentioning.
    """
    x = np.array([0.1, -0.1, 0.2, -0.2] + [50.0 + i for i in range(12)])
    donors = pd.DataFrame(
        {
            "x": x,
            "y": np.arange(1.0, 17.0),
            "weight": np.array([9.0, 1.0, 1e-9, 1e-9] + [5.0] * 12),
        }
    )
    receivers = pd.DataFrame({"x": np.zeros(2000)})
    return donors, receivers


def test_weighted_random_hot_deck_picks_near_donors_in_proportion_to_weight() -> None:
    pytest.importorskip("statmatch")
    donors, receivers = _two_near_donors()
    fit = get_method("statmatch_rand_hotdeck").constructor()
    drawn = fit(donors, ["x"], ["y"], "weight", seed=0)(receivers)["y"]

    assert set(drawn.unique()) <= {1.0, 2.0}  # never a far donor
    share = float((drawn == 1.0).mean())
    assert 0.87 < share < 0.93  # 0.9 expected; sd of the share is 0.007


def test_random_hot_deck_without_weights_picks_near_donors_equally() -> None:
    """The control for the weighted draw: same near donors, weights left out."""
    pytest.importorskip("statmatch")
    donors, receivers = _two_near_donors()
    fit = get_method("statmatch_rand_hotdeck_unweighted").constructor()
    drawn = fit(donors, ["x"], ["y"], "weight", seed=0)(receivers)["y"]

    # The four nearest donors, whatever their weights (9, 1 and two of 1e-9).
    assert set(drawn.unique()) == {1.0, 2.0, 3.0, 4.0}
    shares = drawn.value_counts(normalize=True)
    assert shares.between(0.21, 0.29).all()  # 0.25 expected; sd is 0.01


def test_random_hot_decks_coincide_when_the_donor_weights_are_equal() -> None:
    """Under uniform weights the weighted draw is the unweighted one.

    The OpenML table relies on this to print the two as one row.
    """
    pytest.importorskip("statmatch")
    donors, receivers = _two_near_donors()
    donors = donors.assign(weight=1.0)
    drawn = [
        get_method(key).constructor()(donors, ["x"], ["y"], "weight", seed=3)(receivers)
        for key in ("statmatch_rand_hotdeck_unweighted", "statmatch_rand_hotdeck")
    ]
    pd.testing.assert_frame_equal(drawn[0], drawn[1])


def test_weighted_random_hot_deck_is_seeded_and_leaves_numpys_generator_alone() -> None:
    pytest.importorskip("statmatch")
    donors, receivers = _two_near_donors()
    fit = get_method("statmatch_rand_hotdeck").constructor()

    np.random.seed(123)
    before = np.random.get_state()
    first = fit(donors, ["x"], ["y"], "weight", seed=7)(receivers)
    after = np.random.get_state()
    assert before[0] == after[0] and (before[1] == after[1]).all()
    assert before[2:] == after[2:]

    pd.testing.assert_frame_equal(
        first, fit(donors, ["x"], ["y"], "weight", seed=7)(receivers)
    )
    other = fit(donors, ["x"], ["y"], "weight", seed=8)(receivers)
    assert not first["y"].equals(other["y"])


@pytest.mark.parametrize(("n_receivers", "cap"), [(10, 1), (30, 3)])
def test_constrained_hot_deck_caps_donor_use(n_receivers: int, cap: int) -> None:
    """Identical receivers all want one donor; the cap spreads them out."""
    pytest.importorskip("statmatch")
    donors = pd.DataFrame(
        {
            "x": np.arange(12.0),
            "y": np.arange(100.0, 112.0),
            "weight": np.linspace(1.0, 4.0, 12),
        }
    )
    receivers = pd.DataFrame({"x": np.zeros(n_receivers)})
    nearest = get_method("statmatch_hotdeck").constructor()
    constrained = get_method("statmatch_constrained_hotdeck").constructor()

    unconstrained = nearest(donors, ["x"], ["y"], "weight", seed=0)(receivers)["y"]
    assert unconstrained.nunique() == 1  # every receiver takes the same donor

    drawn = constrained(donors, ["x"], ["y"], "weight", seed=0)(receivers)["y"]
    assert drawn.index.equals(receivers.index)
    assert drawn.value_counts().max() <= cap
    assert methods._donor_use_cap(n_receivers, len(donors)) == cap


def test_constrained_hot_deck_is_skipped_above_its_size_limit(monkeypatch) -> None:
    """Too large an assignment problem is refused, so the sweep records a skip."""
    pytest.importorskip("statmatch")
    donors = pd.DataFrame(
        {"x": np.arange(12.0), "y": np.arange(12.0), "weight": np.ones(12)}
    )
    receivers = pd.DataFrame({"x": np.zeros(30)})
    predict = get_method("statmatch_constrained_hotdeck").constructor()(
        donors, ["x"], ["y"], "weight", seed=0
    )
    monkeypatch.setattr(methods, "CONSTRAINED_MAX_CELLS", 30 * 12 * 3 - 1)
    with pytest.raises(NotImplementedError, match="1,080 cells"):
        predict(receivers)
    monkeypatch.setattr(methods, "CONSTRAINED_MAX_CELLS", 30 * 12 * 3)
    assert len(predict(receivers)) == 30


def test_pooled_leaves_overstate_a_zero_atom() -> None:
    """Why the gates matter: an ungated forest draws too many zeros.

    In a forest grown to purity, a zero atom forms larger leaves than the
    positive values do. Pooling leaf samples across trees, which is the forest
    library's default and what the estimator's forests do, weights each tree
    by its leaf's size, so the atom is over-represented in a row's
    conditional. Giving each tree's leaf equal weight removes the excess.
    """
    quantile_forest = pytest.importorskip("quantile_forest")
    rng = np.random.default_rng(0)
    n, m = 4000, 2000
    x = rng.normal(size=(n + m, 2))
    positive = rng.random(n + m) < 1.0 / (1.0 + np.exp(-(x[:, 0] - 1.7)))
    y = np.where(positive, np.exp(rng.normal(size=n + m) + x[:, 1]), 0.0)
    forest = quantile_forest.RandomForestQuantileRegressor(
        n_estimators=100, max_samples_leaf=None, random_state=0
    ).fit(x[:n], y[:n])
    grid = list(np.linspace(1 / 202, 1 - 1 / 202, 201))
    level = rng.integers(0, len(grid), size=m)

    def drawn_zero_share(weighted_leaves: bool) -> float:
        predictions = np.asarray(
            forest.predict(x[n:], quantiles=grid, weighted_leaves=weighted_leaves)
        )
        return float((predictions[np.arange(m), level] == 0.0).mean())

    truth = float((y[n:] == 0.0).mean())
    assert drawn_zero_share(weighted_leaves=False) - truth > 0.08
    assert abs(drawn_zero_share(weighted_leaves=True) - truth) < 0.06
