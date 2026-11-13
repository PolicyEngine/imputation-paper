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
    "VARIANT_KEYS",
    "BASELINE_KEYS",
    "ZERO_ATOL",
    "SCF_NET_WORTH_COMPONENTS",
    "single_sign_shift",
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
    #: gain: weighting off, chaining off, gates off. populace-fit exposes no
    #: gates-off switch; the gates-off arms reach its own gate-free path by
    #: giving every target a single sign class (see :func:`single_sign_shift`).
    ABLATION = "ablation"
    #: The candidate re-specified for one task rather than knocked out: for
    #: example, a target set by an accounting identity instead of imputed.
    VARIANT = "variant"
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


#: Magnitudes at or below this count as zero when a target's sign classes are
#: counted. Equal to populace-fit's ``DEFAULT_ZERO_ATOL``, which is what its
#: own regime detection uses (a test pins the two together).
ZERO_ATOL = 1e-6


def single_sign_shift(y: np.ndarray, *, zero_atol: float = ZERO_ATOL) -> float:
    """The shift that leaves ``y`` with one sign class; ``0.0`` if it has one.

    populace-fit builds a sign gate only for a target whose training values
    fall in more than one of {negative, zero, positive}. A target with a
    single class is fitted with one forest and no gate. Adding the returned
    shift makes every value at least 1, so a shifted target takes that
    gate-free path; subtracting it from the draws returns them to the
    original scale. Trees split on the target only through differences, so
    the shift changes what is gated and nothing else.

    Args:
        y: Training target values.
        zero_atol: Magnitudes at or below this are zeros.

    Returns:
        ``1 - min(0, min(y))`` when ``y`` has more than one sign class, else
        ``0.0``.
    """
    y = np.asarray(y, dtype=np.float64)
    if y.size == 0:
        return 0.0
    classes = (
        int((np.abs(y) <= zero_atol).any())
        + int((y > zero_atol).any())
        + int((y < -zero_atol).any())
    )
    if classes <= 1:
        return 0.0
    return 1.0 - min(0.0, float(y.min()))


def _fit_populace(
    train_df: pd.DataFrame,
    predictors: list[str],
    targets: list[str],
    weights: str | np.ndarray | None,
    *,
    seed: int,
    weights_mode: str,
    chained: bool,
    gated: bool,
) -> tuple[list, dict[str, float]]:
    """Fit populace-fit's estimator; return the fitted models and target shifts.

    The shifts are all zero when ``gated``. Otherwise each target with more
    than one sign class is moved by :func:`single_sign_shift` before the fit,
    and the fit is checked to have built no gate.
    """
    from populace.fit import Regime, RegimeGatedQRF

    if weights_mode == "none":
        spec: np.ndarray | str = "none"
    else:
        vector = _resolve_weight_vector(train_df, weights)
        spec = "none" if vector is None else vector
    table = train_df.loc[:, [*predictors, *targets]]

    shifts = dict.fromkeys(targets, 0.0)
    if not gated:
        shifts = {
            target: single_sign_shift(table[target].to_numpy(dtype=np.float64))
            for target in targets
        }
        table = table.assign(
            **{target: table[target] + shift for target, shift in shifts.items()}
        )

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

    if not gated:
        ungated = {Regime.POSITIVE_ONLY, Regime.NEGATIVE_ONLY, Regime.DEGENERATE_ZERO}
        built = {
            target: regime
            for model in fitted
            for target, regime in model.regimes().items()
            if regime not in ungated
        }
        if built:
            raise RuntimeError(
                f"The gates-off arm fitted a gated regime for {built}; the "
                "single-sign shift did not take."
            )
    return fitted, shifts


def _ungated_regimes(
    train_df: pd.DataFrame,
    predictors: Sequence[str],
    targets: Sequence[str],
    weights: str | np.ndarray | None,
    *,
    seed: int = 0,
) -> dict[str, str]:
    """The regime populace-fit detects for each target in the gates-off arm."""
    fitted, _ = _fit_populace(
        train_df,
        list(predictors),
        list(targets),
        weights,
        seed=seed,
        weights_mode="design",
        chained=True,
        gated=False,
    )
    return {
        target: regime for model in fitted for target, regime in model.regimes().items()
    }


def _populace_fit_constructor(
    *, weights_mode: str = "design", chained: bool = True, gated: bool = True
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
        gated: ``True`` leaves the estimator's sign gates on. ``False`` is the
            gates-off ablation: each target with more than one sign class is
            shifted to be strictly positive before the fit, which sends it
            down the estimator's own gate-free path (one forest over all
            rows, same weighted bootstrap, same draw), and the draws are
            shifted back. A target that already has one sign class is left
            alone, so there the ablation is the full estimator.

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
        predictors = list(predictors)
        targets = list(targets)
        fitted, shifts = _fit_populace(
            train_df,
            predictors,
            targets,
            weights,
            seed=seed,
            weights_mode=weights_mode,
            chained=chained,
            gated=gated,
        )

        def predict(test_df: pd.DataFrame) -> pd.DataFrame:
            features = test_df.loc[:, predictors]
            drawn = pd.concat([model.predict(features) for model in fitted], axis=1)
            drawn = drawn.loc[:, targets]
            for target, shift in shifts.items():
                if shift:
                    drawn[target] = drawn[target] - shift
            return drawn

        return predict

    return fit


#: The SCF balance-sheet identity, as the signed components of net worth:
#: ``networth = fin + nfin - debt``. It holds exactly in the summary extract.
SCF_NET_WORTH_COMPONENTS: tuple[tuple[str, float], ...] = (
    ("fin", 1.0),
    ("nfin", 1.0),
    ("debt", -1.0),
)


def _derived_target_constructor(
    base: Callable[[], FitFn],
    *,
    derived: str,
    components: tuple[tuple[str, float], ...],
) -> FitFn:
    """Model the components with ``base`` and set ``derived`` by an identity.

    Where a target is an exact signed sum of other targets, a practitioner
    would impute the components and compute the sum. This wrapper does that:
    ``base`` is fitted on every target except ``derived``, and each drawn row
    gets ``derived = sum(sign * component)``, so the identity holds on every
    row by construction.

    Args:
        base: Constructor of the method that models the components.
        derived: The target set by the identity.
        components: ``(component, sign)`` pairs whose signed sum is
            ``derived``.

    Returns:
        A :class:`FitFn`. Its fit raises :class:`NotImplementedError` on a
        task whose targets lack ``derived`` or a component (the sweep records
        that as a skip), and :class:`ValueError` if the donor rows do not
        satisfy the identity.
    """
    names = [name for name, _ in components]

    def fit(
        train_df: pd.DataFrame,
        predictors: Sequence[str],
        targets: Sequence[str],
        weights: str | np.ndarray | None,
        *,
        seed: int = 0,
    ) -> PredictFn:
        targets = list(targets)
        missing = [name for name in (derived, *names) if name not in targets]
        if missing:
            raise NotImplementedError(
                f"The derived-target arm sets {derived!r} from {names}; this "
                f"task's targets {targets} lack {missing}."
            )
        implied = sum(sign * train_df[name] for name, sign in components)
        gap = float((train_df[derived] - implied).abs().max())
        scale = max(1.0, float(train_df[derived].abs().max()))
        if gap > 1e-9 * scale:
            raise ValueError(
                f"The donor rows do not satisfy the identity {derived} = "
                f"sum of signed {names}: the largest gap is {gap:g}."
            )
        modeled = [target for target in targets if target != derived]
        predict_modeled = base()(train_df, predictors, modeled, weights, seed=seed)

        def predict(test_df: pd.DataFrame) -> pd.DataFrame:
            drawn = predict_modeled(test_df).copy()
            drawn[derived] = sum(sign * drawn[name] for name, sign in components)
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
# Baselines: py-statmatch hot decks (nearest, weighted random, constrained)
# ---------------------------------------------------------------------------


def _donor_use_cap(n_receivers: int, n_donors: int) -> int:
    """The tightest cap on donor use that still gives every receiver a donor."""
    return max(1, -(-n_receivers // n_donors))


#: Largest cost matrix, in cells, the constrained hot deck is asked to solve.
#: py-statmatch builds a dense ``n_receivers x (n_donors x cap)`` matrix and
#: solves the assignment exactly. Every within-survey task is under this (the
#: largest is 4,000 x 16,000). The population view is not: 20,000 receivers
#: against 3,676 donors used up to six times is 441 million cells. On the
#: minimal profile's real matching variables, with their many ties, one split
#: took 13 minutes and 6 GB (against seconds for every other method), so ten
#: splits of both profiles would take about four hours. Above the limit the
#: arm is skipped and the sweep records why.
CONSTRAINED_MAX_CELLS = 100_000_000


def _statmatch_hotdeck_constructor(selection: str = "nearest") -> FitFn:
    """Adapter for py-statmatch's distance hot decks (statistical matching).

    py-statmatch is a Python port of R's StatMatch. Each receiver is matched
    to a donor on the matching variables and ``create_fused`` donates that
    donor's target values, so zero masses and cross-target structure come
    over whole. The three selections share one distance: Euclidean on the
    matching variables, standardized by pooled weighted moments when the donor
    weights are not all equal (what ``nnd_hotdeck`` does itself). They differ
    only in which donor is taken.

    Args:
        selection: How the donor is chosen.

            * ``"nearest"`` -- ``nnd_hotdeck``: the nearest donor. Donor
              weights enter the standardization only. Deterministic given the
              donor pool, so ``seed`` only varies with the split.
            * ``"weighted_random"`` -- ``rand_hotdeck`` with ``weight_don``
              (StatMatch's RANDwNND): one donor among the ``ceil(sqrt(n))``
              nearest, drawn with probability proportional to its weight.
            * ``"constrained"`` -- ``nnd_hotdeck`` with ``constr_alg``: the
              assignment of least total distance in which no donor serves
              more than ``ceil(n_receivers / n_donors)`` receivers. The cap
              counts receivers; it does not use the weights. Its predict
              raises :class:`NotImplementedError` above
              :data:`CONSTRAINED_MAX_CELLS`, which the sweep records as a
              skip.

    Returns:
        A :class:`FitFn`.
    """
    if selection not in {"nearest", "weighted_random", "constrained"}:
        raise ValueError(f"Unknown hot-deck selection {selection!r}.")

    def fit(
        train_df: pd.DataFrame,
        predictors: Sequence[str],
        targets: Sequence[str],
        weights: str | np.ndarray | None,
        *,
        seed: int = 0,
    ) -> PredictFn:
        from statmatch import create_fused, nnd_hotdeck, rand_hotdeck
        from statmatch.nnd_hotdeck import _weighted_standardize

        predictors = list(predictors)
        targets = list(targets)
        donor = train_df.loc[:, [*predictors, *targets]].reset_index(drop=True)
        vector = _resolve_weight_vector(train_df, weights)

        def match(receiver: pd.DataFrame) -> pd.DataFrame:
            weighted = {} if vector is None else {"don_weights": vector}
            if selection == "nearest":
                return nnd_hotdeck(
                    data_rec=receiver, data_don=donor, match_vars=predictors, **weighted
                )["mtc.ids"]
            if selection == "constrained":
                cap = _donor_use_cap(len(receiver), len(donor))
                cells = len(receiver) * len(donor) * cap
                if cells > CONSTRAINED_MAX_CELLS:
                    raise NotImplementedError(
                        f"The constrained hot deck is not run here: {len(receiver):,} "
                        f"receivers against {len(donor):,} donors used up to {cap} "
                        f"times is a dense assignment problem of {cells:,} cells, "
                        f"over the {CONSTRAINED_MAX_CELLS:,}-cell limit."
                    )
                return nnd_hotdeck(
                    data_rec=receiver,
                    data_don=donor,
                    match_vars=predictors,
                    k=cap,
                    constr_alg="hungarian",
                    **weighted,
                )["mtc.ids"]

            # rand_hotdeck measures distance on the columns as given, so the
            # standardization nnd_hotdeck applies internally is applied here.
            rec = receiver.to_numpy(dtype=np.float64)
            don = donor.loc[:, predictors].to_numpy(dtype=np.float64)
            if vector is not None and not np.allclose(vector, vector[0]):
                rec, don = _weighted_standardize(rec, don, np.ones(len(rec)), vector)
            standardized = pd.DataFrame(don, columns=predictors)
            weight_column = None
            if vector is not None:
                weight_column = "__donor_weight__"
                standardized[weight_column] = vector
            # rand_hotdeck draws from numpy's global generator. Seed it for
            # this call and put the caller's state back afterwards.
            state = np.random.get_state()
            np.random.seed(seed)
            try:
                return rand_hotdeck(
                    data_rec=pd.DataFrame(rec, columns=predictors),
                    data_don=standardized,
                    match_vars=predictors,
                    dist_fun="euclidean",
                    cut_don="rot",
                    weight_don=weight_column,
                )["mtc.ids"]
            finally:
                np.random.set_state(state)

        def predict(test_df: pd.DataFrame) -> pd.DataFrame:
            receiver = test_df.loc[:, predictors].reset_index(drop=True)
            fused = create_fused(
                data_rec=receiver,
                data_don=donor,
                mtc_ids=match(receiver),
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


#: The method surface. Ordered candidate -> ablations -> variants -> baselines.
REGISTRY: dict[str, Method] = {
    # --- Candidate ---
    "populace_fit": Method(
        key="populace_fit",
        category=MethodCategory.CANDIDATE,
        description=(
            "populace-fit: regime-gated, sequentially-chained, weighted-bootstrap "
            "quantile-regression-forest, weight-aware by construction."
        ),
        citation_key="microcosm2026",
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
        citation_key="microcosm2026",
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
        citation_key="microcosm2026",
        constructor=lambda: _populace_fit_constructor(
            weights_mode="design", chained=False
        ),
    ),
    "populace_fit_ungated": Method(
        key="populace_fit_ungated",
        category=MethodCategory.ABLATION,
        description=(
            "populace-fit with its sign gates off: every target is fitted "
            "through the estimator's own single-sign path (one forest over all "
            "rows). Isolates the gating contribution."
        ),
        citation_key="microcosm2026",
        constructor=lambda: _populace_fit_constructor(
            weights_mode="design", chained=True, gated=False
        ),
    ),
    "populace_fit_ungated_unchained": Method(
        key="populace_fit_ungated_unchained",
        category=MethodCategory.ABLATION,
        description=(
            "populace-fit with gates and chaining both off: one weighted "
            "forest per target on the predictors alone. The fourth cell of "
            "the gated-by-chained design."
        ),
        citation_key="microcosm2026",
        constructor=lambda: _populace_fit_constructor(
            weights_mode="design", chained=False, gated=False
        ),
    ),
    # --- Variants (the candidate re-specified for one task) ---
    "populace_fit_derived": Method(
        key="populace_fit_derived",
        category=MethodCategory.VARIANT,
        description=(
            "populace-fit imputing the three balance-sheet components, chained, "
            "with net worth set by the identity networth = fin + nfin - debt. "
            "Runs only where the task has all four columns."
        ),
        citation_key="microcosm2026",
        constructor=lambda: _derived_target_constructor(
            lambda: _populace_fit_constructor(weights_mode="design", chained=True),
            derived="networth",
            components=SCF_NET_WORTH_COMPONENTS,
        ),
    ),
    "populace_fit_unchained_derived": Method(
        key="populace_fit_unchained_derived",
        category=MethodCategory.VARIANT,
        description=(
            "As populace_fit_derived with the three components fitted "
            "independently: what chaining adds once the identity is imposed."
        ),
        citation_key="microcosm2026",
        constructor=lambda: _derived_target_constructor(
            lambda: _populace_fit_constructor(weights_mode="design", chained=False),
            derived="networth",
            components=SCF_NET_WORTH_COMPONENTS,
        ),
    ),
    # --- Baselines (standard survey-imputation methods) ---
    # `plain_qrf` is `microimpute_qrf` under a second key. It was first
    # registered as a gates-off, chaining-off ablation endpoint, which it is
    # not: microimpute's QRF conditions each target on the targets already
    # imputed, and it draws through its own sampler. The key stays because the
    # committed run artifacts carry rows under it.
    "plain_qrf": Method(
        key="plain_qrf",
        category=MethodCategory.BASELINE,
        description=(
            "microimpute quantile-regression-forest imputer as shipped: the same "
            "estimator and configuration as `microimpute_qrf`, under the key the "
            "committed runs also carry. Ungated, but sequentially chained and "
            "drawn through its own sampler, so not a knock-out of the candidate."
        ),
        citation_key="meinshausen2006quantile",
        constructor=lambda: _microimpute_constructor("qrf"),
    ),
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
        citation_key="vonhippel2013should",
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
        citation_key="dorazio2025statmatch",
        constructor=_statmatch_hotdeck_constructor,
    ),
    "statmatch_rand_hotdeck": Method(
        key="statmatch_rand_hotdeck",
        category=MethodCategory.BASELINE,
        description=(
            "py-statmatch random hot deck with donor weights (StatMatch's "
            "RANDwNND): one donor among the nearest, drawn with probability "
            "proportional to its weight."
        ),
        citation_key="dorazio2025statmatch",
        constructor=lambda: _statmatch_hotdeck_constructor("weighted_random"),
    ),
    "statmatch_constrained_hotdeck": Method(
        key="statmatch_constrained_hotdeck",
        category=MethodCategory.BASELINE,
        description=(
            "py-statmatch constrained nearest-neighbour hot deck: the least-"
            "distance assignment in which no donor serves more than "
            "ceil(receivers / donors) receivers."
        ),
        citation_key="dorazio2025statmatch",
        constructor=lambda: _statmatch_hotdeck_constructor("constrained"),
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
VARIANT_KEYS: tuple[str, ...] = tuple(
    k for k, m in REGISTRY.items() if m.category == MethodCategory.VARIANT
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
