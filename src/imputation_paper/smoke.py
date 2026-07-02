"""A seeded toy weighted dataset for the CI smoke path.

The generator builds a small weighted table with the two structural features the
paper's metrics and methods care about: a *zero-inflated, sign-mixed* target (a
mass at zero plus positive and negative values, like a net capital-gain or
net-income component) and a *continuous* target (strictly positive and
heavy-tailed, like wealth). Rows carry design weights so the harness exercises
the weighted-metric path end to end.

Nothing here imports a method package or any heavy dependency: it is pure numpy
and pandas, so ``imp demo`` and the tests run on the base install alone.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

#: Column names the toy dataset exposes, so callers do not hard-code strings.
PREDICTORS = ("age", "employment_income", "is_female")
ZERO_INFLATED_TARGET = "capital_gains"
CONTINUOUS_TARGET = "net_worth"
TARGETS = (ZERO_INFLATED_TARGET, CONTINUOUS_TARGET)
WEIGHT_COLUMN = "household_weight"


@dataclass(frozen=True)
class ToyDataset:
    """A seeded toy dataset split into donor (train) and receiver (test).

    Attributes:
        train: Donor rows -- predictors, targets, and weights all observed.
        test: Receiver rows -- same columns; the targets are the held-out truth
            the metrics score against (a real receiver would not observe them).
        predictors: Predictor column names.
        targets: Target column names (one zero-inflated/sign-mixed, one
            continuous).
        weight_column: The design-weight column name.
    """

    train: pd.DataFrame
    test: pd.DataFrame
    predictors: tuple[str, ...]
    targets: tuple[str, ...]
    weight_column: str

    @property
    def donor(self) -> pd.DataFrame:
        """Alias for :pyattr:`train` (statistical-matching terminology)."""
        return self.train

    @property
    def receiver(self) -> pd.DataFrame:
        """Alias for :pyattr:`test` (statistical-matching terminology)."""
        return self.test


def make_toy_dataset(
    *, seed: int = 0, n: int = 800, holdout_frac: float = 0.2
) -> ToyDataset:
    """Build a deterministic weighted toy dataset.

    The zero-inflated target has a genuine three-sign support (a zero mass plus
    positive and negative values), so a regime-gated method has something to gate
    on and the zero-share-preservation metric has a nonzero target zero share.
    The continuous target is strictly positive and right-skewed. Weights vary
    across rows so the weighted metrics differ from their unweighted counterparts.

    Args:
        seed: Seed for the (reproducible) draw.
        n: Total row count before the holdout split.
        holdout_frac: Fraction of rows placed in the receiver/test split.

    Returns:
        A :class:`ToyDataset`.
    """
    rng = np.random.default_rng(seed)

    age = rng.integers(18, 85, size=n).astype(np.float64)
    employment_income = rng.lognormal(mean=10.2, sigma=0.6, size=n)
    is_female = (rng.random(n) < 0.5).astype(np.float64)

    # Design weights: a lognormal spread so the weighted metrics are not a
    # relabelling of the unweighted ones.
    weights = rng.lognormal(mean=6.8, sigma=0.4, size=n)

    # Continuous target: strictly positive, heavy-tailed, correlated with the
    # predictors so a conditional model beats the marginal.
    net_worth = np.exp(
        1.6
        + 0.03 * (age - 40.0)
        + 0.9 * (np.log(employment_income) - 10.2)
        + rng.normal(0.0, 0.7, size=n)
    )

    # Zero-inflated, sign-mixed target: a zero mass, a positive cluster, and a
    # negative cluster. The zero probability depends on the predictors so the
    # gate is learnable rather than constant.
    p_nonzero = 1.0 / (1.0 + np.exp(-(-1.4 + 0.6 * (np.log(employment_income) - 10.2))))
    is_nonzero = rng.random(n) < p_nonzero
    sign = np.where(rng.random(n) < 0.75, 1.0, -1.0)
    magnitude = rng.lognormal(mean=7.0, sigma=1.0, size=n)
    capital_gains = np.where(is_nonzero, sign * magnitude, 0.0)

    table = pd.DataFrame(
        {
            "age": age,
            "employment_income": employment_income,
            "is_female": is_female,
            ZERO_INFLATED_TARGET: capital_gains,
            CONTINUOUS_TARGET: net_worth,
            WEIGHT_COLUMN: weights,
        }
    )

    order = rng.permutation(n)
    n_holdout = int(round(n * holdout_frac))
    test_idx = order[:n_holdout]
    train_idx = order[n_holdout:]
    train = table.iloc[np.sort(train_idx)].reset_index(drop=True)
    test = table.iloc[np.sort(test_idx)].reset_index(drop=True)

    return ToyDataset(
        train=train,
        test=test,
        predictors=PREDICTORS,
        targets=TARGETS,
        weight_column=WEIGHT_COLUMN,
    )


def weighted_empirical_quantile_draw(
    train: pd.DataFrame,
    test: pd.DataFrame,
    predictors: list[str],  # noqa: ARG001 - trivial baseline ignores predictors
    targets: list[str],
    weights: str | np.ndarray | None,
    *,
    seed: int = 0,
) -> pd.DataFrame:
    """A trivial unconditional weighted baseline used to exercise the harness.

    For each target it draws, per receiver row, one donor value sampled with
    probability proportional to the donor weights -- i.e. a draw from the
    *weighted marginal* of the target, ignoring predictors entirely. This is not
    a serious imputation method; it exists so ``imp demo`` and the CI tests can
    run a real method through the real metrics without importing any of the
    heavy method packages.

    Args:
        train: Donor rows (must carry ``targets`` and, if a name, ``weights``).
        test: Receiver rows to impute for.
        predictors: Ignored (a marginal draw uses no predictors).
        targets: Target columns to draw.
        weights: Donor weights -- a column name in ``train``, an array aligned to
            ``train``, or ``None`` for an unweighted draw.
        seed: Seed for the draw.

    Returns:
        A :class:`pandas.DataFrame` of drawn target values, indexed like ``test``.
    """
    rng = np.random.default_rng(seed)
    if weights is None:
        probabilities = None
    else:
        weight_values = (
            train[weights].to_numpy(dtype=np.float64)
            if isinstance(weights, str)
            else np.asarray(weights, dtype=np.float64)
        )
        probabilities = weight_values / weight_values.sum()

    drawn: dict[str, np.ndarray] = {}
    for target in targets:
        donor_values = train[target].to_numpy(dtype=np.float64)
        picks = rng.choice(
            len(donor_values), size=len(test), replace=True, p=probabilities
        )
        drawn[target] = donor_values[picks]
    return pd.DataFrame(drawn, index=test.index)
