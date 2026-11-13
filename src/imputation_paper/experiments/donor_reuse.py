"""Donor reuse in a matched file: how much receiver weight leans on one donor.

Hot-deck matching donates whole observed values, so a donor that is the
nearest neighbour of many receivers is copied many times. That is harmless in
the middle of a distribution and decisive in its tail. Under the harness's
interpolating weighted quantile, a donor whose receivers carry a weight share
above ``(1 - q)`` *plus the largest single receiver's share* puts the matched
file's ``q``-quantile at or above that donor's value. (Without the
one-receiver margin the interpolation can land just below it.) So one such
donor above the holdout's q99 is enough to inflate the q99 ratio.
:func:`donor_reuse_summary` measures the shares that condition needs, in
aggregate terms only (weight shares and counts; no record's values).
"""

from __future__ import annotations

import numpy as np

__all__ = ["donor_reuse_summary"]


def donor_reuse_summary(
    donor_index: np.ndarray,
    receiver_weights: np.ndarray,
    donated: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    """Summarize how concentrated a matched file is on single donors.

    Args:
        donor_index: For each receiver row, the position of its matched donor.
        receiver_weights: Receiver weights, aligned with ``donor_index``.
        donated: The value each receiver row was given (the matched donor's),
            aligned with ``donor_index``.
        threshold: A reference level of ``donated`` (for example the holdout's
            99th percentile); "the tail" is receiver weight given a value
            strictly above it.

    Returns:
        ``donors_used`` (distinct donors matched), ``top_donor_weight_share``
        (largest share of receiver weight matched to one donor),
        ``tail_weight_share`` (share of receiver weight in the tail),
        ``tail_donors`` (distinct donors in the tail), and
        ``top_tail_donor_weight_share`` (largest share of *all* receiver
        weight that one tail donor accounts for), which never exceeds
        ``top_donor_weight_share`` or ``tail_weight_share``; and
        ``max_receiver_weight_share`` (the largest single receiver's share,
        the margin in the sufficient condition above).

    Raises:
        ValueError: If the inputs are not aligned 1-D arrays, or the weights
            are negative or sum to zero.
    """
    donor_index = np.asarray(donor_index)
    weights = np.asarray(receiver_weights, dtype=np.float64)
    donated = np.asarray(donated, dtype=np.float64)
    if not (donor_index.shape == weights.shape == donated.shape and weights.ndim == 1):
        raise ValueError("donor_index, receiver_weights, and donated must align (1-D).")
    total = float(weights.sum())
    if (weights < 0).any() or total <= 0:
        raise ValueError("receiver weights must be non-negative and sum above zero.")

    def by_donor(mask: np.ndarray) -> np.ndarray:
        """Receiver weight per distinct donor among the masked rows."""
        if not mask.any():
            return np.zeros(0)
        _, inverse = np.unique(donor_index[mask], return_inverse=True)
        return np.bincount(inverse, weights=weights[mask])

    everyone = by_donor(np.ones(len(weights), dtype=bool))
    tail = donated > threshold
    tail_by_donor = by_donor(tail)
    return {
        "donors_used": int(everyone.size),
        "top_donor_weight_share": float(everyone.max() / total),
        "tail_weight_share": float(weights[tail].sum() / total),
        "tail_donors": int(tail_by_donor.size),
        "top_tail_donor_weight_share": float(tail_by_donor.max() / total)
        if tail_by_donor.size
        else 0.0,
        "max_receiver_weight_share": float(weights.max() / total),
    }
