"""The method surface: candidate, ablations, and baselines under one registry.

Every method the paper compares is registered here as a :class:`Method`: a key,
a human description, the BibTeX citation key that grounds it, and a lazy
``constructor`` that returns a fitted-imputer factory. The registry is the
single source of truth for "what is compared against what", so the sweep, the
tables, and the paper's method list all read from it.

**Lazy imports are load-bearing.** The method packages (``populace-fit``,
``microimpute``, ``py-statmatch``) live behind the ``methods`` optional extra and
are *not* installed in CI. So no method package may be imported at module import
time: each constructor's returned ``fit`` imports what it needs *inside its own
body*. This lets ``list_methods()`` and the registry be imported (and tested) on
the base install alone, while a fit raises a clear :class:`ModuleNotFoundError`
only if called without its package present -- which the sweep records as a skip.

**Adapter contract.** A constructor returns a ``fit`` callable with the uniform
signature::

    fit(train_df, predictors, targets, weights, *, seed=0) -> predict

    predict(test_df) -> pandas.DataFrame   # one column per target, indexed like test_df

where ``weights`` is a column name in ``train_df``, an array aligned to it, or
``None``, and ``seed`` controls the method's stochastic draw so paired-seed
sweeps are reproducible. Adapters resolve ``weights`` to a plain vector before
handing it to their package, so no package-specific weight-column conventions
leak through.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd

__all__ = [
    "Method",
    "MethodCategory",
    "FitFn",
    "PredictFn",
    "REGISTRY",
    "list_methods",
    "get_method",
    "CANDIDATE_KEYS",
    "ABLATION_KEYS",
    "BASELINE_KEYS",
]

#: Quantile grid used to convert quantile-imputer outputs into per-row draws:
#: each receiver row samples one grid level uniformly, so the drawn column is a
#: (discretized) sample from the estimated conditional rather than a point
#: prediction. 99 levels keep the tails while bounding the prediction cost.
DRAW_QUANTILE_GRID: tuple[float, ...] = tuple(
    float(q) for q in np.linspace(0.01, 0.99, 99)
)


class PredictFn(Protocol):
    """A fitted imputer: maps receiver rows to a frame of drawn target columns."""

    def __call__(self, test_df: pd.DataFrame) -> pd.DataFrame: ...


class FitFn(Protocol):
    """A method's fit entry point.

    Fits on donor rows and returns a :class:`PredictFn`. ``weights`` selects the
    donor weights: a column name in ``train_df``, an array aligned to it, or
    ``None`` for an unweighted fit. ``seed`` drives the method's stochastic
    draw.
    """

    def __call__(
        self,
        train_df: pd.DataFrame,
        predictors: Sequence[str],
        targets: Sequence[str],
        weights: str | np.ndarray | None,
        *,
        seed: int = 0,
    ) -> PredictFn: ...


class MethodCategory:
    """Registry categories, exposed as constants so callers avoid magic strings."""

    #: The method the paper defends: populace-fit's regime-gated, chained,
    #: weighted QRF, run in its default weight-aware configuration.
    CANDIDATE = "candidate"
    #: A deliberate knock-out of one populace-fit design choice, to attribute the
    #: gain (weighting off, chaining off, gates off).
    ABLATION = "ablation"
    #: A standard survey-imputation method the candidate is benchmarked against.
    BASELINE = "baseline"


@dataclass(frozen=True)
class Method:
    """One entry in the method surface.

    Attributes:
        key: Stable identifier used in configs, CSVs, and tables.
        category: One of :class:`MethodCategory`.
        description: One-line human description.
        citation_key: BibTeX key in ``paper/bibliography/references.bib`` that
            grounds the method (empty for the trivial harness baseline).
        constructor: Zero-argument callable returning the method's
            :class:`FitFn`. The returned fit imports its package lazily.
    """

    key: str
    category: str
    description: str
    citation_key: str
    constructor: Callable[[], FitFn]


def _resolve_weight_vector(
    train_df: pd.DataFrame, weights: str | np.ndarray | None
) -> np.ndarray | None:
    """Resolve the adapter contract's ``weights`` to a plain vector (or None)."""
    if weights is None:
        return None
    if isinstance(weights, str):
        return train_df[weights].to_numpy(dtype=np.float64)
    return np.asarray(weights, dtype=np.float64)


# ---------------------------------------------------------------------------
# Candidate + ablations: populace-fit (the subject of the paper)
# ---------------------------------------------------------------------------


def _populace_fit_constructor(
    *, weights_mode: str = "design", chained: bool = True
) -> FitFn:
    """Adapter for populace-fit's :class:`RegimeGatedQRF` over plain DataFrames.

    Uses the DataFrame front door (populace#290): weights are stated
    explicitly -- the donor weight vector, or the literal ``"none"`` for the
    unweighted ablation, which ignores whatever weights the task supplies
    (that is the point of the ablation).

    Args:
        weights_mode: ``"design"`` (weighted, the candidate) or ``"none"``
            (the unweighted ablation).
        chained: ``True`` fits all targets in one call so each conditions on
            the targets drawn before it; ``False`` fits each target
            independently (the chaining-off ablation).

    Returns:
        A :class:`FitFn`.
    """

    def fit(
        train_df: pd.DataFrame,
        predictors: Sequence[str],
        targets: Sequence[str],
        weights: str | np.ndarray | None,
        *,
        seed: int = 0,
    ) -> PredictFn:
        from populace.fit import RegimeGatedQRF

        predictors = list(predictors)
        targets = list(targets)
        if weights_mode == "none":
            spec: np.ndarray | str = "none"
        else:
            vector = _resolve_weight_vector(train_df, weights)
            spec = "none" if vector is None else vector
        table = train_df.loc[:, [*predictors, *targets]]

        if chained:
            fitted = [
                RegimeGatedQRF(seed=seed).fit(table, predictors, targets, weights=spec)
            ]
        else:
            # Chaining-off: each target sees only the predictors. Offsetting
            # the seed per target keeps the independent draws independent.
            fitted = [
                RegimeGatedQRF(seed=seed + position).fit(
                    table.loc[:, [*predictors, target]],
                    predictors,
                    [target],
                    weights=spec,
                )
                for position, target in enumerate(targets)
            ]

        def predict(test_df: pd.DataFrame) -> pd.DataFrame:
            features = test_df.loc[:, predictors]
            drawn = pd.concat([model.predict(features) for model in fitted], axis=1)
            return drawn.loc[:, targets]

        return predict

    return fit


# ---------------------------------------------------------------------------
# Baselines: microimpute (plain QRF, OLS, quantile regression, matching)
# ---------------------------------------------------------------------------


def _construct_with_seed(cls: type, seed: int):
    """Instantiate a microimpute imputer, passing ``seed`` if it accepts one."""
    try:
        parameters = inspect.signature(cls.__init__).parameters
    except (TypeError, ValueError):  # pragma: no cover - defensive
        parameters = {}
    if "seed" in parameters:
        return cls(seed=seed)
    return cls()


def _microimpute_constructor(model_name: str) -> FitFn:
    """Adapter for a :mod:`microimpute` imputer (QRF / OLS / QuantReg / Matching).

    microimpute imputers share ``fit(X_train, predictors, imputed_variables,
    weight_col=...)`` -- where ``weight_col`` accepts a plain vector and weights
    the fit by resampling the training data in proportion to weight -- and
    ``predict(X_test, quantiles)`` returning ``{quantile: DataFrame}``.

    The models are turned into samplers by predicting the
    :data:`DRAW_QUANTILE_GRID` once and drawing one grid level per receiver
    row, so each row's value is a draw from its estimated conditional (per-row
    independent quantiles, not one shared quantile). Matching is not in the
    released microimpute (it exists only behind an R dependency in the dev
    tree), so the matching baseline is py-statmatch's ``statmatch_hotdeck``.

    Args:
        model_name: ``"qrf"`` | ``"ols"`` | ``"quantreg"``.

    Returns:
        A :class:`FitFn`.
    """

    def fit(
        train_df: pd.DataFrame,
        predictors: Sequence[str],
        targets: Sequence[str],
        weights: str | np.ndarray | None,
        *,
        seed: int = 0,
    ) -> PredictFn:
        import microimpute.models as models

        cls = {
            "qrf": models.QRF,
            "ols": models.OLS,
            "quantreg": models.QuantReg,
        }[model_name]
        predictors = list(predictors)
        targets = list(targets)
        vector = _resolve_weight_vector(train_df, weights)

        imputer = _construct_with_seed(cls, seed)
        train = train_df.loc[:, [*predictors, *targets]].reset_index(drop=True)
        fit_kwargs: dict = {} if vector is None else {"weight_col": vector}
        if model_name == "quantreg":
            # QuantReg fits one linear model per quantile at fit time; the
            # draw grid must therefore be declared here, not at predict.
            fit_kwargs["quantiles"] = list(DRAW_QUANTILE_GRID)
        fitted = imputer.fit(train, predictors, targets, **fit_kwargs)

        def predict(test_df: pd.DataFrame) -> pd.DataFrame:
            features = test_df.loc[:, predictors].reset_index(drop=True)
            rng = np.random.default_rng(seed)
            grid = list(DRAW_QUANTILE_GRID)
            by_quantile = fitted.predict(features, quantiles=grid)
            # Stack (n_quantiles, n_rows, n_targets) and gather one uniformly
            # drawn grid level per receiver row.
            stacked = np.stack(
                [
                    by_quantile[q].loc[:, targets].to_numpy(dtype=np.float64)
                    for q in grid
                ]
            )
            picks = rng.integers(0, len(grid), size=stacked.shape[1])
            frame = pd.DataFrame(
                stacked[picks, np.arange(stacked.shape[1]), :],
                columns=targets,
            )
            frame = frame.loc[:, targets].copy()
            frame.index = test_df.index
            return frame

        return predict

    return fit


# ---------------------------------------------------------------------------
# Baseline: py-statmatch nearest-neighbour-distance hot deck
# ---------------------------------------------------------------------------


def _statmatch_hotdeck_constructor() -> FitFn:
    """Adapter for :func:`statmatch.nnd_hotdeck` (classical hot-deck matching).

    py-statmatch is a Python port of R's StatMatch. ``nnd_hotdeck`` matches
    each receiver record to its nearest donor on the (standardized) matching
    variables -- donor weights inform the standardization and tie-breaking --
    and ``create_fused`` donates the matched donor's target values. This is
    the traditional survey-fusion baseline; it is deterministic given the
    donor pool, so ``seed`` only varies with the split.

    Returns:
        A :class:`FitFn`.
    """

    def fit(
        train_df: pd.DataFrame,
        predictors: Sequence[str],
        targets: Sequence[str],
        weights: str | np.ndarray | None,
        *,
        seed: int = 0,  # noqa: ARG001 - deterministic method; contract uniformity
    ) -> PredictFn:
        from statmatch import create_fused, nnd_hotdeck

        predictors = list(predictors)
        targets = list(targets)
        donor = train_df.loc[:, [*predictors, *targets]].reset_index(drop=True)
        vector = _resolve_weight_vector(train_df, weights)

        def predict(test_df: pd.DataFrame) -> pd.DataFrame:
            receiver = test_df.loc[:, predictors].reset_index(drop=True)
            matched = nnd_hotdeck(
                data_rec=receiver,
                data_don=donor,
                match_vars=predictors,
                **({} if vector is None else {"don_weights": vector}),
            )
            fused = create_fused(
                data_rec=receiver,
                data_don=donor,
                mtc_ids=matched["mtc.ids"],
                z_vars=targets,
            )
            frame = fused.loc[:, targets].copy()
            frame.index = test_df.index
            return frame

        return predict

    return fit


# ---------------------------------------------------------------------------
# Harness baseline: a real trivial weighted-marginal draw (used by CI/demo)
# ---------------------------------------------------------------------------


def _weighted_marginal_constructor() -> FitFn:
    """A real, trivial baseline: per-row draw from the weighted target marginal.

    Unlike the other constructors this one needs no heavy dependency, so
    ``imp demo`` and the CI tests can push a real method through the real
    metrics. It ignores predictors entirely -- it is a lower bound a serious
    conditional method must beat, not a contender.
    """

    def fit(
        train_df: pd.DataFrame,
        predictors: Sequence[str],
        targets: Sequence[str],
        weights: str | np.ndarray | None,
        *,
        seed: int = 0,
    ) -> PredictFn:
        from imputation_paper.smoke import weighted_empirical_quantile_draw

        train = train_df.reset_index(drop=True)

        def predict(test_df: pd.DataFrame) -> pd.DataFrame:
            return weighted_empirical_quantile_draw(
                train, test_df, list(predictors), list(targets), weights, seed=seed
            )

        return predict

    return fit


#: The method surface. Ordered candidate -> ablations -> baselines.
REGISTRY: dict[str, Method] = {
    # --- Candidate ---
    "populace_fit": Method(
        key="populace_fit",
        category=MethodCategory.CANDIDATE,
        description=(
            "populace-fit: regime-gated, sequentially-chained, weighted-bootstrap "
            "quantile-regression-forest, weight-aware by construction."
        ),
        citation_key="populace2026",
        constructor=lambda: _populace_fit_constructor(
            weights_mode="design", chained=True
        ),
    ),
    # --- Ablations (attribute the gain to each design choice) ---
    "populace_fit_unweighted": Method(
        key="populace_fit_unweighted",
        category=MethodCategory.ABLATION,
        description=(
            "populace-fit with weights='none' (the only unweighted path): isolates "
            "the weighted-bootstrap contribution."
        ),
        citation_key="populace2026",
        constructor=lambda: _populace_fit_constructor(
            weights_mode="none", chained=True
        ),
    ),
    "populace_fit_unchained": Method(
        key="populace_fit_unchained",
        category=MethodCategory.ABLATION,
        description=(
            "populace-fit fitting each target independently (no sequential "
            "chaining): isolates the chaining contribution."
        ),
        citation_key="populace2026",
        constructor=lambda: _populace_fit_constructor(
            weights_mode="design", chained=False
        ),
    ),
    "plain_qrf": Method(
        key="plain_qrf",
        category=MethodCategory.ABLATION,
        description=(
            "Single ungated, unchained quantile-regression-forest (microimpute QRF) "
            "at matched hyperparameters: the gates-off, chaining-off comparison."
        ),
        citation_key="meinshausen2006quantile",
        constructor=lambda: _microimpute_constructor("qrf"),
    ),
    # --- Baselines (standard survey-imputation methods) ---
    "microimpute_qrf": Method(
        key="microimpute_qrf",
        category=MethodCategory.BASELINE,
        description="microimpute quantile-regression-forest imputer.",
        citation_key="meinshausen2006quantile",
        constructor=lambda: _microimpute_constructor("qrf"),
    ),
    "microimpute_ols": Method(
        key="microimpute_ols",
        category=MethodCategory.BASELINE,
        description="microimpute OLS-with-normal-residuals imputer.",
        citation_key="vonhippel2007should",
        constructor=lambda: _microimpute_constructor("ols"),
    ),
    "microimpute_quantreg": Method(
        key="microimpute_quantreg",
        category=MethodCategory.BASELINE,
        description="microimpute quantile-regression imputer.",
        citation_key="koenker1978regression",
        constructor=lambda: _microimpute_constructor("quantreg"),
    ),
    "statmatch_hotdeck": Method(
        key="statmatch_hotdeck",
        category=MethodCategory.BASELINE,
        description=(
            "py-statmatch nearest-neighbour-distance hot-deck statistical matching "
            "(a Python port of R's StatMatch)."
        ),
        citation_key="dorazio2021statistical",
        constructor=_statmatch_hotdeck_constructor,
    ),
    # --- Harness baseline (real, dependency-free; used by demo + CI) ---
    "weighted_marginal": Method(
        key="weighted_marginal",
        category=MethodCategory.BASELINE,
        description=(
            "Trivial per-row draw from the weighted target marginal (ignores "
            "predictors). A dependency-free lower bound used by the demo and CI."
        ),
        citation_key="",
        constructor=_weighted_marginal_constructor,
    ),
}

#: Keys grouped by category, for the sweep and the paper's method list.
CANDIDATE_KEYS: tuple[str, ...] = tuple(
    k for k, m in REGISTRY.items() if m.category == MethodCategory.CANDIDATE
)
ABLATION_KEYS: tuple[str, ...] = tuple(
    k for k, m in REGISTRY.items() if m.category == MethodCategory.ABLATION
)
BASELINE_KEYS: tuple[str, ...] = tuple(
    k for k, m in REGISTRY.items() if m.category == MethodCategory.BASELINE
)


def list_methods(category: str | None = None) -> list[str]:
    """Return registry keys, optionally filtered to one :class:`MethodCategory`."""
    if category is None:
        return list(REGISTRY)
    return [k for k, m in REGISTRY.items() if m.category == category]


def get_method(key: str) -> Method:
    """Return the :class:`Method` for ``key`` (raises ``KeyError`` if unknown)."""
    try:
        return REGISTRY[key]
    except KeyError:
        raise KeyError(
            f"Unknown method {key!r}. Registered methods: {sorted(REGISTRY)}."
        ) from None
