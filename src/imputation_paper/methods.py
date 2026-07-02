"""The method surface: candidate, ablations, and baselines under one registry.

Every method the paper compares is registered here as a :class:`Method`: a key,
a human description, the BibTeX citation key that grounds it, and a lazy
``constructor`` that returns a fitted-imputer factory. The registry is the
single source of truth for "what is compared against what", so the sweep, the
tables, and the paper's method list all read from it.

**Lazy imports are load-bearing.** The method packages (``populace-fit``,
``microimpute``, ``py-statmatch``) live behind the ``methods`` optional extra and
are *not* installed in CI. So no method package may be imported at module import
time: each constructor imports what it needs *inside its own body*. This lets
``list_methods()`` and the registry be imported (and tested) on the base install
alone, while a constructor raises a clear :class:`ModuleNotFoundError` only if
called without its package present.

**Adapter contract.** A constructor returns a ``fit`` callable with the uniform
signature::

    fit(train_df, predictors, targets, weights) -> predict

    predict(test_df) -> pandas.DataFrame   # one column per target, indexed like test_df

where ``weights`` is a column name in ``train_df``, an array aligned to it, or
``None``. The adapters here are deliberately **thin, honest stubs**: each
documents exactly how it would wire its package, and raises
:class:`NotImplementedError` where a faithful adapter still has to be written and
verified against the real API. They are not fake working implementations. The
one exception is :func:`_weighted_marginal_constructor`, a real trivial baseline
used by the toy demo and CI (it lives in :mod:`imputation_paper.smoke`).
"""

from __future__ import annotations

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


class PredictFn(Protocol):
    """A fitted imputer: maps receiver rows to a frame of drawn target columns."""

    def __call__(self, test_df: pd.DataFrame) -> pd.DataFrame: ...


class FitFn(Protocol):
    """A method's fit entry point.

    Fits on donor rows and returns a :class:`PredictFn`. ``weights`` selects the
    donor weights: a column name in ``train_df``, an array aligned to it, or
    ``None`` for an unweighted fit.
    """

    def __call__(
        self,
        train_df: pd.DataFrame,
        predictors: Sequence[str],
        targets: Sequence[str],
        weights: str | np.ndarray | None,
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
            :class:`FitFn`. Imports its package lazily inside the body.
    """

    key: str
    category: str
    description: str
    citation_key: str
    constructor: Callable[[], FitFn]


# ---------------------------------------------------------------------------
# Candidate: populace-fit (the subject of the paper)
# ---------------------------------------------------------------------------


def _populace_fit_constructor(
    *, weights_mode: str = "design", chained: bool = True
) -> FitFn:
    """Adapter for populace-fit's :class:`RegimeGatedQRF` over plain DataFrames.

    populace-fit fits over a :class:`populace.frame.Frame`, and a DataFrame front
    door (fit on a plain DataFrame with an explicit weights array) is landing via
    PR -- that front door is the standalone-use path this benchmark targets.

    Wiring (to finalize against the shipped front door):

    * ``weights_mode="design"`` -> pass the donor weights as the fit weights
      (the weight-aware default the paper defends).
    * ``weights_mode="none"``   -> fit with ``weights="none"`` (the *only*
      unweighted path; used by the ``populace_fit_unweighted`` ablation).
    * ``chained=True``  -> fit all targets in one call so each conditions on the
      targets drawn before it (sequential chaining).
    * ``chained=False`` -> fit each target independently (the chaining-off
      ablation); no target sees another's draw.

    Args:
        weights_mode: ``"design"`` (weighted) or ``"none"`` (unweighted).
        chained: Whether to chain targets sequentially.

    Returns:
        A :class:`FitFn`.
    """

    def fit(
        train_df: pd.DataFrame,
        predictors: Sequence[str],
        targets: Sequence[str],
        weights: str | np.ndarray | None,
    ) -> PredictFn:
        # Lazy import: populace-fit is only present with the `methods` extra.
        from populace.fit import RegimeGatedQRF  # noqa: F401

        raise NotImplementedError(
            "populace-fit adapter pending the DataFrame front door (fit on a "
            "plain DataFrame with explicit weights), which is landing via PR. "
            f"Intended config: weights_mode={weights_mode!r}, chained={chained}. "
            "Wire RegimeGatedQRF.fit over the front door, mapping donor weights "
            "to the fit weights (weights='none' for the unweighted ablation) and "
            "fitting targets jointly (chained) or one-by-one (unchained)."
        )

    return fit


# ---------------------------------------------------------------------------
# Baselines: microimpute (plain QRF, OLS, quantile regression, matching)
# ---------------------------------------------------------------------------


def _microimpute_constructor(model_name: str) -> FitFn:
    """Adapter for a :mod:`microimpute` imputer (QRF / OLS / QuantReg / Matching).

    microimpute exposes each method as an ``Imputer`` with ``fit`` and
    ``predict`` and integrates survey weights by *sampling* the training data in
    proportion to weight (its documented weight path). ``plain_qrf`` uses the
    same underlying quantile-forest as the candidate but *ungated and unchained*,
    so it isolates the effect of populace-fit's structural machinery at a matched
    forest.

    Wiring (to finalize against the installed microimpute API):

    * ``"qrf"``      -> ``microimpute.models.QRF``.
    * ``"ols"``      -> ``microimpute.models.OLS``.
    * ``"quantreg"`` -> ``microimpute.models.QuantReg``.
    * ``"matching"`` -> ``microimpute.models.Matching`` (predictive-mean / hot
      deck matching within microimpute).

    Args:
        model_name: Which microimpute model to wrap.

    Returns:
        A :class:`FitFn`.
    """

    def fit(
        train_df: pd.DataFrame,
        predictors: Sequence[str],
        targets: Sequence[str],
        weights: str | np.ndarray | None,
    ) -> PredictFn:
        # Lazy import: microimpute is only present with the `methods` extra.
        import microimpute.models as _models  # noqa: F401

        raise NotImplementedError(
            f"microimpute {model_name!r} adapter not implemented yet. Wire the "
            "corresponding microimpute Imputer: fit on the donor predictors/"
            "targets passing the donor weights through microimpute's "
            "weighted-sampling path, then predict draws for the receiver rows "
            "and return a DataFrame of target columns indexed like test_df."
        )

    return fit


# ---------------------------------------------------------------------------
# Baseline: py-statmatch nearest-neighbour-distance hot deck
# ---------------------------------------------------------------------------


def _statmatch_hotdeck_constructor() -> FitFn:
    """Adapter for :func:`statmatch.nnd_hotdeck` (classical hot-deck matching).

    py-statmatch is a Python port of R's StatMatch; ``nnd_hotdeck`` performs
    unconstrained nearest-neighbour-distance hot-deck matching, donating an
    observed donor value to each receiver record. This is the traditional
    survey-fusion baseline (the family EUROMOD uses).

    Wiring (to finalize against the installed py-statmatch API):

    * Call ``statmatch.nnd_hotdeck(recipient=test_df, donor=train_df,
      match_vars=list(predictors), ...)`` to get the receiver->donor match,
      optionally weighting by the donor weights.
    * Use ``statmatch.create_fused`` to attach the matched donor ``targets`` to
      the receiver rows, and return them as a DataFrame indexed like test_df.

    Returns:
        A :class:`FitFn`.
    """

    def fit(
        train_df: pd.DataFrame,
        predictors: Sequence[str],
        targets: Sequence[str],
        weights: str | np.ndarray | None,
    ) -> PredictFn:
        # Lazy import: py-statmatch is only present with the `methods` extra.
        from statmatch import create_fused, nnd_hotdeck  # noqa: F401

        raise NotImplementedError(
            "py-statmatch nnd_hotdeck adapter not implemented yet. Match the "
            "receiver to the donor on the predictors with nnd_hotdeck (weighting "
            "by donor weights where supported), fuse the matched donor targets "
            "onto the receiver with create_fused, and return them indexed like "
            "test_df."
        )

    return fit


# ---------------------------------------------------------------------------
# Harness baseline: a real trivial weighted-marginal draw (used by CI/demo)
# ---------------------------------------------------------------------------


def _weighted_marginal_constructor() -> FitFn:
    """A real, trivial baseline: per-row draw from the weighted target marginal.

    Unlike the other constructors this one is fully implemented (it needs no
    heavy dependency), so ``imp demo`` and the CI tests can push a real method
    through the real metrics. It ignores predictors entirely -- it is a lower
    bound a serious conditional method must beat, not a contender.
    """

    def fit(
        train_df: pd.DataFrame,
        predictors: Sequence[str],
        targets: Sequence[str],
        weights: str | np.ndarray | None,
    ) -> PredictFn:
        from imputation_paper.smoke import weighted_empirical_quantile_draw

        train = train_df.reset_index(drop=True)

        def predict(test_df: pd.DataFrame) -> pd.DataFrame:
            return weighted_empirical_quantile_draw(
                train, test_df, list(predictors), list(targets), weights
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
    "microimpute_matching": Method(
        key="microimpute_matching",
        category=MethodCategory.BASELINE,
        description="microimpute predictive-mean / hot-deck matching imputer.",
        citation_key="andridge2010review",
        constructor=lambda: _microimpute_constructor("matching"),
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
