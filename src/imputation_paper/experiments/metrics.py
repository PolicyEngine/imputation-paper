"""Weighted evaluation metrics for imputed target distributions.

An imputation is a draw from an estimated conditional distribution, so the
metrics score *distributional* fidelity, not point accuracy, and they do so
under survey weights -- the property the paper argues common practice neglects.
Three metrics are implemented in full because they are cheap and testable:

* :func:`weighted_pinball_loss` -- weighted quantile (pinball) loss over a
  quantile grid, the primary distributional-calibration metric.
* :func:`weighted_wasserstein1` -- weighted Wasserstein-1 distance from the
  imputed sample to a donor sample, a single-number marginal-fit summary.
* :func:`zero_share_error` -- absolute error in the weighted share of exact
  zeros, the diagnostic for zero-inflated targets.

Two more are stubs with a precise contract but no implementation yet:

* :func:`prdc_coverage` -- Precision/Density/Coverage (PRDC) coverage.
* :func:`reweight_fragility` -- the adversarial-reweighting "landmine"
  diagnostic.

All weighted reductions accept ``weights=None`` for the unweighted case, so the
same code paths serve the weighted/unweighted ablation.
"""

from __future__ import annotations

import numpy as np

#: Default quantile grid for :func:`weighted_pinball_loss`. Deciles avoid the
#: exact 0/1 endpoints (where pinball loss is dominated by a single tail order
#: statistic) while spanning the distribution.
DEFAULT_QUANTILE_GRID: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


def _as_weights(weights: np.ndarray | None, n: int) -> np.ndarray:
    """Return a validated non-negative weight vector of length ``n``.

    ``None`` becomes uniform ones (the unweighted case). Raises on a length
    mismatch, negative weights, or an all-zero vector, so a metric never
    silently divides by a zero total.
    """
    if weights is None:
        return np.ones(n, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if w.shape != (n,):
        raise ValueError(f"weights must have shape ({n},), got {w.shape}.")
    if np.any(w < 0):
        raise ValueError("weights must be non-negative.")
    total = float(w.sum())
    if total <= 0:
        raise ValueError("weights must not sum to zero.")
    return w


def _weighted_quantile(
    values: np.ndarray, weights: np.ndarray, q: np.ndarray
) -> np.ndarray:
    """Weighted quantiles of ``values`` at levels ``q``.

    Uses the standard cumulative-weight interpolation: sort by value, form the
    normalized cumulative weight at the midpoint of each atom, and linearly
    interpolate the value at each requested level. Reduces to the usual linear
    (``numpy.quantile`` "linear") quantile when weights are equal.
    """
    order = np.argsort(values)
    v = values[order]
    w = weights[order]
    cumulative = np.cumsum(w)
    total = cumulative[-1]
    # Midpoint (type-7-like) plotting positions on the weighted CDF.
    positions = (cumulative - 0.5 * w) / total
    return np.interp(q, positions, v)


def weighted_pinball_loss(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    weights: np.ndarray | None = None,
    quantiles: tuple[float, ...] = DEFAULT_QUANTILE_GRID,
) -> float:
    """Weighted pinball (quantile) loss, averaged over a quantile grid.

    For each level ``tau`` the loss compares the ``tau``-quantile of the
    *predicted* (imputed) sample against every held-out truth value with the
    asymmetric check function ``rho_tau(u) = u * (tau - 1{u < 0})``, weighted by
    the receiver weights, then averages the per-level weighted means over the
    grid. Lower is better.

    This scores the imputed *distribution*: a method that recovers the receiver
    population's conditional quantiles scores low even if no single row is
    predicted exactly. Because a single stochastic draw does not itself carry
    quantiles, the predicted quantiles are read from the pooled draw here; the
    per-method sweep can instead pass a method's own predicted-quantile columns.

    Args:
        y_true: Held-out truth values (receiver), shape ``(n,)``.
        y_pred: Imputed values whose quantiles are evaluated, shape ``(m,)``.
        weights: Receiver weights aligned to ``y_true`` (``None`` for uniform).
        quantiles: Quantile levels to score.

    Returns:
        The grid-averaged weighted pinball loss.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    w = _as_weights(weights, y_true.shape[0])
    q = np.asarray(quantiles, dtype=np.float64)
    pred_quantiles = np.quantile(y_pred, q)

    total = w.sum()
    losses = np.empty(q.shape[0], dtype=np.float64)
    for i, (tau, q_hat) in enumerate(zip(q, pred_quantiles, strict=True)):
        residual = y_true - q_hat
        check = residual * (tau - (residual < 0.0))
        losses[i] = float(np.dot(w, check) / total)
    return float(losses.mean())


def weighted_wasserstein1(
    imputed: np.ndarray,
    donor: np.ndarray,
    *,
    imputed_weights: np.ndarray | None = None,
    donor_weights: np.ndarray | None = None,
    n_grid: int = 1000,
) -> float:
    """Weighted Wasserstein-1 (earth-mover) distance between two 1-D samples.

    Computes ``W_1(P, Q) = integral over u in (0,1) of |F_P^{-1}(u) -
    F_Q^{-1}(u)| du`` by evaluating both weighted quantile functions on a shared
    grid of ``u`` levels and integrating their absolute difference (trapezoidal).
    This is the weighted analogue of the L1 distance between inverse CDFs and
    reduces to :func:`scipy.stats.wasserstein_distance` when both weight vectors
    are uniform. Lower is better.

    Args:
        imputed: Imputed target values.
        donor: Reference (donor) target values.
        imputed_weights: Weights for ``imputed`` (``None`` for uniform).
        donor_weights: Weights for ``donor`` (``None`` for uniform).
        n_grid: Number of interior ``u`` levels to integrate over.

    Returns:
        The weighted Wasserstein-1 distance.
    """
    imputed = np.asarray(imputed, dtype=np.float64)
    donor = np.asarray(donor, dtype=np.float64)
    wi = _as_weights(imputed_weights, imputed.shape[0])
    wd = _as_weights(donor_weights, donor.shape[0])
    # Interior grid avoids the degenerate exact-0/1 endpoints of the inverse CDF.
    u = (np.arange(n_grid, dtype=np.float64) + 0.5) / n_grid
    qi = _weighted_quantile(imputed, wi, u)
    qd = _weighted_quantile(donor, wd, u)
    return float(np.trapezoid(np.abs(qi - qd), u))


def zero_share_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    true_weights: np.ndarray | None = None,
    pred_weights: np.ndarray | None = None,
    zero_atol: float = 1e-6,
) -> float:
    """Absolute error in the weighted share of exact zeros.

    Zero-inflated targets (income components, credits, gains) carry a mass at
    zero that a method must reproduce: too few zeros inflates a program's
    caseload, too many erases it. This returns ``|zero_share(pred) -
    zero_share(true)|`` under weights, where a value is a zero when its magnitude
    is at or below ``zero_atol``. Lower is better; ``0`` means the imputed and
    true zero masses match.

    Args:
        y_true: Held-out truth values.
        y_pred: Imputed values.
        true_weights: Weights for ``y_true`` (``None`` for uniform).
        pred_weights: Weights for ``y_pred`` (``None`` for uniform).
        zero_atol: Magnitudes at or below this count as zeros.

    Returns:
        The absolute weighted zero-share error, in ``[0, 1]``.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    wt = _as_weights(true_weights, y_true.shape[0])
    wp = _as_weights(pred_weights, y_pred.shape[0])
    true_share = float(np.dot(wt, np.abs(y_true) <= zero_atol) / wt.sum())
    pred_share = float(np.dot(wp, np.abs(y_pred) <= zero_atol) / wp.sum())
    return abs(pred_share - true_share)


def prdc_coverage(
    real: np.ndarray,
    synthetic: np.ndarray,
    *,
    k: int = 5,
) -> float:
    """Precision/Density/Coverage (PRDC) coverage. **TODO: not implemented.**

    Coverage (Naeem et al. 2020) is the fraction of real (held-out) points that
    have at least one synthetic (imputed) neighbour within their k-th-nearest-
    neighbour radius in the real manifold, computed in standardized feature
    space. It is the paper's cross-survey generalization metric: it rewards an
    imputer for populating the whole real manifold, not just its dense interior,
    and -- unlike marginal distances -- it is defined on the joint of all imputed
    columns.

    TODO: implement the k-NN radius construction and the coverage count over the
    joint imputed-column space, matching the PRDC definition; add a weighted
    variant (real points weighted by receiver weights). Until then this raises so
    no caller silently scores against an unimplemented metric.

    Args:
        real: Real (held-out) points, shape ``(n_real, d)``.
        synthetic: Synthetic (imputed) points, shape ``(n_syn, d)``.
        k: Neighbour rank defining each real point's local radius.

    Raises:
        NotImplementedError: Always, until PRDC coverage is implemented.
    """
    raise NotImplementedError(
        "PRDC coverage (Naeem et al. 2020) is not implemented yet; see the "
        "docstring for the intended k-NN-radius coverage definition."
    )


def reweight_fragility(
    imputed: np.ndarray,
    weights: np.ndarray,  # noqa: ARG001 - contract documented; body is a stub
    *,
    aggregate: str = "sum",
) -> float:
    """Adversarial-reweighting fragility -- the "landmine" diagnostic. **TODO.**

    After imputation, a downstream user re-weights the file (a new calibration,
    a subpopulation zoom, a stress scenario). If a single imputed record carries
    an extreme value, an adversarial or merely aggressive reweighting can make
    that one record dominate a population aggregate -- a "landmine" that detonates
    only under a reweighting the imputer never saw. This diagnostic measures how
    exposed a file is: apply an extreme admissible reweighting and report the
    maximum single-record share of the resulting aggregate (the larger, the more
    fragile).

    The design (to implement):

    * Fix an admissible reweighting family -- e.g. reweightings that hold a set
      of calibration margins but are otherwise free, or a bounded multiplicative
      perturbation ``w_i -> w_i * exp(t * g_i)`` for adversarial direction
      ``g`` and budget ``t``.
    * Maximize, over that family, the largest single-record contribution
      ``max_i (w_i' * a_i) / sum_j (w_j' * a_j)`` to the target ``aggregate`` of
      the imputed quantity ``a`` (``"sum"`` for a total; extendable to a mean or
      a tail share).
    * Return that worst-case share as the fragility score.

    Prior internal eCPS measurements of this quantity exist, but they MUST be
    re-measured under this paper's protocol -- do NOT embed any prior number
    here or in the manuscript. This stub fixes the contract; the sweep will call
    it once implemented.

    Args:
        imputed: Imputed values of the aggregated quantity, shape ``(n,)``.
        weights: Baseline weights aligned to ``imputed``.
        aggregate: Which aggregate the worst-case share is taken of
            (``"sum"`` initially).

    Raises:
        NotImplementedError: Always, until the adversarial-reweighting search is
            implemented.
    """
    raise NotImplementedError(
        "reweight_fragility (the adversarial-reweighting landmine diagnostic) is "
        "not implemented yet; see the docstring for the intended worst-case "
        "single-record-share design. Prior eCPS numbers must be re-measured, not "
        "embedded."
    )
