"""Conditional fit within a survey: imputed against true values on the same rows.

The within-survey metrics of :mod:`imputation_paper.experiments.metrics` score
one imputed column at a time, so they cannot tell a method that gets each
target's distribution right while ignoring the predictors from one that also
gets the conditional distributions right. Within a survey the receivers' true
values are known, which allows a direct test.

:func:`paired_two_sample_auc` builds two files from the *same* receiver rows:
one carries the rows' true target values and the other the imputed ones. Both
files have identical predictors and identical weights, so a classifier that
sees predictors and targets together can only separate them through how the
targets vary with the predictors (and with each other). An AUC of 0.5 means
it cannot.

Two details keep the test honest:

* **Folds are over receiver rows.** Both copies of a row always sit in the
  same fold, so the classifier is never tested on a row whose other copy it
  was trained on.
* **A label-swap null.** Under the hypothesis that the imputed values follow
  the true conditional distribution, the two copies of a row are
  exchangeable. Swapping the labels of a random half of the rows therefore
  gives a draw from the test's null distribution on the same data, which
  shows how far from 0.5 the AUC moves by chance.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from imputation_paper.experiments.metrics import (
    _weighted_quantile,
    weighted_wasserstein1,
)

__all__ = [
    "ReceiverGroups",
    "paired_folds",
    "paired_two_sample_auc",
    "ratios_to_sample",
    "shared_value_share",
    "swap_mask",
    "within_group_wasserstein1",
]


def shared_value_share(values: np.ndarray) -> float:
    """Share of non-zero values that another row also has, exactly.

    In a survey, two households rarely have the same non-zero amount. A
    method that gives several receivers the same donor's amount raises this
    share, which a classifier can pick up. Zeros are left out because many
    households share a true zero.

    Args:
        values: One value per receiver.

    Returns:
        Among the rows whose value is not zero, the share whose value equals
        that of at least one other such row. ``nan`` if every value is zero.
    """
    array = np.asarray(values, dtype=np.float64)
    nonzero = array[array != 0.0]
    if nonzero.size == 0:
        return float("nan")
    _, inverse, counts = np.unique(nonzero, return_inverse=True, return_counts=True)
    return float((counts[inverse] > 1).mean())


def _as_matrix(values: np.ndarray, name: str) -> np.ndarray:
    """Coerce to a 2-D float64 array, one row per receiver."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    if array.ndim != 2:
        raise ValueError(f"{name} must be 1-D or 2-D, got {array.ndim}-D.")
    return array


def paired_folds(
    n_rows: int, *, n_splits: int = 5, seed: int = 0
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield train/test indices into the stacked file, folded by receiver row.

    The stacked file holds the ``n_rows`` true copies first and the ``n_rows``
    imputed copies after them, so row ``i`` and row ``i + n_rows`` are the two
    copies of receiver ``i``. Each fold keeps both copies on the same side.

    Args:
        n_rows: Number of receiver rows.
        n_splits: Number of folds.
        seed: Seed for the fold assignment.

    Yields:
        ``(train_idx, test_idx)`` pairs indexing the ``2 * n_rows`` stacked rows.
    """
    if n_rows < n_splits:
        raise ValueError(f"Need at least {n_splits} rows, got {n_rows}.")
    order = np.random.default_rng(seed).permutation(n_rows)
    for held_out in np.array_split(order, n_splits):
        kept = np.setdiff1d(order, held_out)
        yield (
            np.r_[kept, kept + n_rows],
            np.r_[held_out, held_out + n_rows],
        )


def swap_mask(n_rows: int, *, seed: int = 0) -> np.ndarray:
    """A random half of the receiver rows, for the label-swap null.

    The mask is a pure function of ``n_rows`` and ``seed``, so every method
    scored on the same split is given the same swap.
    """
    return np.random.default_rng([seed, 1]).random(n_rows) < 0.5


def paired_two_sample_auc(
    predictors: np.ndarray,
    truth: np.ndarray,
    imputed: np.ndarray,
    *,
    weights: np.ndarray | None = None,
    swap: np.ndarray | None = None,
    n_splits: int = 5,
    seed: int = 0,
) -> float:
    """Cross-validated AUC of a classifier telling true from imputed targets.

    Args:
        predictors: Receiver predictors, shape ``(n,)`` or ``(n, p)``.
        truth: The receivers' true target values, shape ``(n,)`` or ``(n, k)``.
        imputed: The values a method imputed for the same rows, same shape as
            ``truth``.
        weights: Receiver weights (``None`` for uniform). Both copies of a row
            carry the row's weight, so the two files have equal mass.
        swap: Optional boolean mask over the receiver rows. Where it is true,
            the row's true and imputed values exchange labels, which gives a
            draw from the null distribution (see the module docstring).
        n_splits: Number of folds over receiver rows.
        seed: Seed for the fold assignment and the classifier.

    Returns:
        Mean out-of-fold weighted AUC. 0.5 means the classifier cannot tell
        the imputed values from the true ones given the predictors.

    Raises:
        ValueError: If the inputs do not align.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score

    x = _as_matrix(predictors, "predictors")
    real = _as_matrix(truth, "truth")
    fake = _as_matrix(imputed, "imputed")
    n_rows = x.shape[0]
    if real.shape != fake.shape or real.shape[0] != n_rows:
        raise ValueError(
            f"predictors {x.shape}, truth {real.shape} and imputed "
            f"{fake.shape} must have the same rows, and truth and imputed "
            "the same columns."
        )
    if weights is None:
        w = np.ones(n_rows, dtype=np.float64)
    else:
        w = np.asarray(weights, dtype=np.float64)
        if w.shape != (n_rows,):
            raise ValueError(f"weights must have shape ({n_rows},), got {w.shape}.")
        if (w < 0).any() or w.sum() <= 0:
            raise ValueError("weights must be non-negative and sum above zero.")
    if swap is not None:
        mask = np.asarray(swap, dtype=bool)
        if mask.shape != (n_rows,):
            raise ValueError(f"swap must have shape ({n_rows},), got {mask.shape}.")
        real, fake = (
            np.where(mask[:, None], fake, real),
            np.where(mask[:, None], real, fake),
        )

    stacked = np.vstack([np.hstack([x, real]), np.hstack([x, fake])])
    labels = np.r_[np.zeros(n_rows), np.ones(n_rows)]
    stacked_weights = np.r_[w, w] / w.sum()

    aucs: list[float] = []
    for train_idx, test_idx in paired_folds(n_rows, n_splits=n_splits, seed=seed):
        model = HistGradientBoostingClassifier(random_state=seed)
        model.fit(
            stacked[train_idx],
            labels[train_idx],
            sample_weight=stacked_weights[train_idx],
        )
        scores = model.predict_proba(stacked[test_idx])[:, 1]
        aucs.append(
            float(
                roc_auc_score(
                    labels[test_idx], scores, sample_weight=stacked_weights[test_idx]
                )
            )
        )
    return float(np.mean(aucs))


@dataclass(frozen=True)
class ReceiverGroups:
    """Groups of similar receivers: bands of one income column by age band.

    The paired classifier counts households. A distance in the target's own
    units counts dollars. :func:`within_group_wasserstein1` is the second kind
    of measure made conditional: it compares imputed with true values inside
    each group of this class and averages over the groups.

    The cut points are computed once, on the task's full sample, so every
    split and every method is scored on the same groups.

    Attributes:
        income_column: The predictor the income bands are cut on.
        income_cuts: Ascending cut points of the income bands.
        zero_band: Whether rows with no income form a band of their own, with
            ``income_cuts`` then dividing the rows that have some.
        age_column: The predictor the age bands are cut on.
        age_cuts: Ascending cut points of the age bands, in years.
    """

    income_column: str
    income_cuts: tuple[float, ...]
    zero_band: bool = False
    age_column: str = "age"
    age_cuts: tuple[float, ...] = (40.0, 65.0)

    @classmethod
    def from_sample(
        cls,
        frame: pd.DataFrame,
        *,
        income_column: str,
        levels: Sequence[float],
        weight_column: str,
        zero_band: bool = False,
        age_column: str = "age",
        age_cuts: Sequence[float] = (40.0, 65.0),
    ) -> ReceiverGroups:
        """Cut the income bands at weighted quantiles of ``frame``.

        Args:
            frame: The task's full sample (donors and receivers together).
            income_column: Predictor to cut the income bands on.
            levels: Quantile levels of the cut points, in ``(0, 1)``.
            weight_column: Survey-weight column used for the quantiles.
            zero_band: Give rows with no income their own band and take the
                quantiles over the remaining rows.
            age_column: Predictor to cut the age bands on.
            age_cuts: Cut points of the age bands, in years.
        """
        income = frame[income_column].to_numpy(dtype=np.float64)
        weights = frame[weight_column].to_numpy(dtype=np.float64)
        if zero_band:
            keep = income > 0
            income, weights = income[keep], weights[keep]
        cuts = _weighted_quantile(income, weights, np.asarray(levels, dtype=float))
        return cls(
            income_column=income_column,
            income_cuts=tuple(float(cut) for cut in cuts),
            zero_band=zero_band,
            age_column=age_column,
            age_cuts=tuple(float(cut) for cut in age_cuts),
        )

    def labels(self, rows: pd.DataFrame) -> np.ndarray:
        """One integer group label per row of ``rows``."""
        income = rows[self.income_column].to_numpy(dtype=np.float64)
        band = np.digitize(income, self.income_cuts)
        if self.zero_band:
            band = np.where(income > 0, 1 + band, 0)
        age_band = np.digitize(
            rows[self.age_column].to_numpy(dtype=np.float64), self.age_cuts
        )
        return band * (len(self.age_cuts) + 1) + age_band


def within_group_wasserstein1(
    imputed: np.ndarray,
    truth: np.ndarray,
    weights: np.ndarray,
    labels: np.ndarray,
    *,
    min_rows: int = 5,
) -> float:
    """Wasserstein-1 between imputed and true values inside each group.

    Within every group the weighted distribution of the imputed values is
    compared with that of the same rows' true values. The result is the
    average of the group distances, each weighted by its group's total weight.
    It is in the target's own units.

    Args:
        imputed: Imputed values, one per receiver.
        truth: The same receivers' true values.
        weights: Receiver weights.
        labels: Group label of each receiver.
        min_rows: Groups with fewer receivers than this are left out.

    Returns:
        The weighted mean of the within-group distances.

    Raises:
        ValueError: If no group has at least ``min_rows`` receivers.
    """
    values = np.asarray(imputed, dtype=np.float64)
    actual = np.asarray(truth, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    groups = np.asarray(labels)
    if not (len(values) == len(actual) == len(w) == len(groups)):
        raise ValueError("imputed, truth, weights and labels must align by row.")
    total, mass = 0.0, 0.0
    for label in np.unique(groups):
        mask = groups == label
        if int(mask.sum()) < min_rows:
            continue
        distance = weighted_wasserstein1(
            values[mask],
            actual[mask],
            imputed_weights=w[mask],
            donor_weights=w[mask],
        )
        group_mass = float(w[mask].sum())
        total += distance * group_mass
        mass += group_mass
    if mass <= 0.0:
        raise ValueError(f"no group has {min_rows} or more receivers.")
    return total / mass


def ratios_to_sample(
    imputed: np.ndarray,
    weights: np.ndarray,
    sample: np.ndarray,
    sample_weights: np.ndarray,
    *,
    levels: Sequence[float] = (0.90, 0.99),
) -> dict[str, float]:
    """Imputed total and upper quantiles as multiples of the full sample's.

    Args:
        imputed: Imputed values, one per receiver.
        weights: Receiver weights.
        sample: The target's values in the task's full sample.
        sample_weights: The full sample's weights.
        levels: Quantile levels to report beside the total.

    Returns:
        ``total_ratio`` (ratio of weighted means) and one ``q{level}_ratio``
        per level. A ratio whose reference is zero is ``nan``.
    """
    values = np.asarray(imputed, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    reference = np.asarray(sample, dtype=np.float64)
    reference_w = np.asarray(sample_weights, dtype=np.float64)

    def ratio(numerator: float, denominator: float) -> float:
        return numerator / denominator if denominator else float("nan")

    out = {
        "total_ratio": ratio(
            float(np.dot(values, w) / w.sum()),
            float(np.dot(reference, reference_w) / reference_w.sum()),
        )
    }
    q = np.asarray(levels, dtype=float)
    ours = _weighted_quantile(values, w, q)
    theirs = _weighted_quantile(reference, reference_w, q)
    for level, mine, base in zip(levels, ours, theirs, strict=True):
        out[f"q{round(level * 100):d}_ratio"] = ratio(float(mine), float(base))
    return out
