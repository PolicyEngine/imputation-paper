"""A diagnostic variant of the estimator: sample by weight inside each tree.

The estimator applies donor weights by resampling the donor rows once per
forest, with probability proportional to weight. Every tree of the forest then
bootstraps uniformly from that one resample, so a donor row the resample left
out is seen by no tree.

:func:`per_tree_weighted_sampling` changes that one step and nothing else. For
the duration of a ``with`` block there is no one-off resample, and each tree
draws its own rows from all donor rows with probability proportional to
weight. Gates, sign classes, chaining and the draw are the package's own code.

This is the sampling scikit-learn 1.9 performs when weights are passed to a
forest's ``fit``. The paper pins scikit-learn 1.8, whose trees always
bootstrap uniformly, so the block swaps the function that draws a tree's
bootstrap indices in the two places that call it: scikit-learn's tree builder
and quantile-forest's leaf read-out, which regenerates each tree's draw to
file the rows under their leaves. Under scikit-learn 1.9 the same swap gives
bit-identical results to passing the weights to ``fit``.

Seeds. The package takes each forest's seed and its one-off resample from one
generator. The variant still makes that resample draw, and discards it, so
every gate and every forest keeps the seed it has in the estimator as shipped.

The variant is a diagnostic. It is not the estimator the paper evaluates and
it is not registered among the paper's methods.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator

import numpy as np

__all__ = ["chance_in_one_off_resample", "per_tree_weighted_sampling"]


def chance_in_one_off_resample(weights: np.ndarray) -> np.ndarray:
    """Each row's chance of appearing in the estimator's one-off resample.

    The resample draws as many rows as there are donors, with replacement and
    with probability proportional to weight. A row of weight share ``p`` among
    ``n`` rows is drawn at least once with probability ``1 - (1 - p) ** n``.
    The mean of the result is the expected share of distinct donor rows a
    forest can read. With equal weights it is about 0.63.
    """
    w = np.asarray(weights, dtype=np.float64)
    return 1.0 - (1.0 - w / w.sum()) ** len(w)


def _weighted_indices(
    random_state: object, n_rows: int, n_draws: int, sample_weight: np.ndarray
) -> np.ndarray:
    """One tree's bootstrap indices, drawn with probability proportional to weight.

    This is the weighted branch of scikit-learn 1.9's
    ``sklearn.ensemble._forest._generate_sample_indices``.
    """
    from sklearn.utils import check_random_state

    random_instance = check_random_state(random_state)
    probabilities = sample_weight / np.sum(sample_weight)
    indices = random_instance.choice(n_rows, n_draws, replace=True, p=probabilities)
    return indices.astype(np.int32)


@contextlib.contextmanager
def _weighted_tree_bootstrap(sample_weight: np.ndarray) -> Iterator[None]:
    """Make every tree bootstrap drawn inside the block weight-proportional."""
    import quantile_forest._quantile_forest as qf_module
    from sklearn.ensemble import _forest as sk_forest

    def generate(
        random_state: object, n_samples: int, n_samples_bootstrap: int, *_ignored
    ) -> np.ndarray:
        return _weighted_indices(
            random_state, n_samples, n_samples_bootstrap, sample_weight
        )

    saved = (sk_forest._generate_sample_indices, qf_module._generate_sample_indices)
    sk_forest._generate_sample_indices = generate
    qf_module._generate_sample_indices = generate
    try:
        yield
    finally:
        sk_forest._generate_sample_indices, qf_module._generate_sample_indices = saved


@contextlib.contextmanager
def per_tree_weighted_sampling() -> Iterator[None]:
    """Fit the estimator's forests with weights sampled inside each tree.

    Inside the block, a weighted forest is grown on all of its donor rows and
    each tree bootstraps them with probability proportional to weight. An
    unweighted forest is fitted as shipped.
    """
    from populace.fit import qrf

    original = qrf._fit_forest

    def fit_forest(
        x, y, columns, weights, *, seed, n_estimators, max_samples_leaf, rng
    ):  # noqa: ANN001, ANN202
        if weights is None:
            return original(
                x,
                y,
                columns,
                weights,
                seed=seed,
                n_estimators=n_estimators,
                max_samples_leaf=max_samples_leaf,
                rng=rng,
            )
        # The shipped resample, drawn as qrf._weighted_bootstrap draws it and
        # discarded, so the generator is left where the shipped code leaves it.
        rng.choice(len(x), size=len(x), replace=True, p=weights / float(weights.sum()))
        model = qrf.RandomForestQuantileRegressor(
            n_estimators=n_estimators,
            max_samples_leaf=max_samples_leaf,
            random_state=seed,
        )
        with _weighted_tree_bootstrap(weights):
            model.fit(x, y)
        return qrf._Forest(model=model, columns=columns)

    qrf._fit_forest = fit_forest
    try:
        yield
    finally:
        qrf._fit_forest = original
