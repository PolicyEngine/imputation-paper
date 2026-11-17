"""Population checks on the SCF extract: weights, the identity, published tables.

Three properties the paper's SCF experiments lean on, checked on any wave:

* **Weight total.** ``wgt`` summed over all five implicates is the number of
  families the survey represents.
* **Balance-sheet identity.** ``networth == fin + nfin - debt`` (and
  ``asset == fin + nfin``) on every row. The populace-scale chaining result is
  interpretable only because this holds exactly in the donor.
* **Agreement with the Fed's published tables.** All-family median and mean
  net worth, pooled over implicates, against Table 2 of the Fed's report on
  the 2025 survey, and the weight total against its Table B.1.

The last is a *differential* check with a known, explained gap. The Fed
computes its tables from the internal data set. Per each wave's codebook
("Disclosure review"), the public extract rounds continuous variables, blurs
and re-imputes some values, and removes the families whose net worth reaches
the minimum for the Forbes 400 list: seven in each wave (4,595 of 4,602 in
2022; 4,360 of 4,367 in 2025). The size of the gap is therefore calibrated
rather than assumed: the tolerance on the 2025 wave is set from how far the
2022 public file sits from the published 2022 figures (see
:func:`published_gaps`).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from imputation_paper.data.scf import load_scf_implicates

__all__ = [
    "PUBLISHED_DOLLAR_YEAR",
    "PUBLISHED_FAMILIES_MILLIONS",
    "PUBLISHED_FAMILIES_INTERVIEWED",
    "PUBLISHED_NET_WORTH_THOUSANDS",
    "PUBLISHED_SOURCE",
    "population_summary",
    "published_gaps",
    "summarize_implicates",
    "weighted_median",
]

#: Where the published figures below come from.
PUBLISHED_SOURCE = (
    "Board of Governors of the Federal Reserve System, 'Changes in U.S. Family "
    "Finances from 2022 to 2025' "
    "(https://www.federalreserve.gov/publications/files/scf26.pdf): Table 2 "
    "(net worth, all families, thousands of 2025 dollars), Table B.1 (families "
    "represented, millions), and Appendix A (families interviewed)."
)

#: Dollar basis of the published net-worth figures.
PUBLISHED_DOLLAR_YEAR = 2025

#: Table 2, "All families": median and mean net worth, thousands of 2025 dollars.
PUBLISHED_NET_WORTH_THOUSANDS: dict[int, dict[str, float]] = {
    2022: {"median": 211.1, "mean": 1164.1},
    2025: {"median": 215.9, "mean": 1241.5},
}

#: Table B.1: number of families represented, millions.
PUBLISHED_FAMILIES_MILLIONS: dict[int, float] = {2022: 131.3, 2025: 135.0}

#: Appendix A: families interviewed (the internal data set's size).
PUBLISHED_FAMILIES_INTERVIEWED: dict[int, int] = {2022: 4602, 2025: 4367}

#: Floors on the calibrated tolerances: the net-worth figures get at least one
#: percent; the weight total, published to the nearest 0.1 million, a tenth of
#: a percent.
_MIN_NET_WORTH_TOLERANCE = 0.01
_MIN_WEIGHT_TOLERANCE = 0.001


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """The weighted median: the smallest value holding half the weight below it.

    Sorts by value and returns the first value whose cumulative weight reaches
    half the total. Invariant to rescaling the weights and to row order.
    """
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if values.shape != weights.shape or values.ndim != 1 or values.size == 0:
        raise ValueError("values and weights must be non-empty 1-D arrays alike.")
    if (weights < 0).any() or weights.sum() <= 0:
        raise ValueError("weights must be non-negative and sum to more than zero.")
    order = np.argsort(values, kind="stable")
    cumulative = np.cumsum(weights[order])
    index = int(np.searchsorted(cumulative, 0.5 * cumulative[-1], side="left"))
    return float(values[order][index])


def summarize_implicates(frame: pd.DataFrame) -> dict[str, float | int]:
    """Summarize one wave's all-implicate table (see ``load_scf_implicates``).

    Args:
        frame: One row per household and implicate, carrying ``yy1``,
            ``implicate``, ``wgt``, ``fin``, ``nfin``, ``debt``, ``asset``, and
            ``networth``.

    Returns:
        Row, family, and implicate counts; the weight total; the largest
        absolute violations of the two accounting identities, in dollars; and
        pooled all-family median and mean net worth.
    """
    weights = frame["wgt"].to_numpy(dtype=np.float64)
    net_worth = frame["networth"].to_numpy(dtype=np.float64)
    identity = frame["networth"] - (frame["fin"] + frame["nfin"] - frame["debt"])
    assets = frame["asset"] - (frame["fin"] + frame["nfin"])
    return {
        "rows": int(len(frame)),
        "families": int(frame["yy1"].nunique()),
        "implicates": int(frame["implicate"].nunique()),
        "weight_total": float(weights.sum()),
        "net_worth_identity_max_abs_error": float(identity.abs().max()),
        "asset_identity_max_abs_error": float(assets.abs().max()),
        "net_worth_median": weighted_median(net_worth, weights),
        "net_worth_mean": float(np.average(net_worth, weights=weights)),
    }


def population_summary(year: int) -> dict[str, float | int]:
    """Load a wave on the published dollar basis and summarize it."""
    frame = load_scf_implicates(year, dollar_year=PUBLISHED_DOLLAR_YEAR)
    summary = summarize_implicates(frame)
    provenance = frame.attrs["provenance"]
    return {
        **summary,
        "scf_year": year,
        "dollar_year": PUBLISHED_DOLLAR_YEAR,
        "zip_sha256": str(provenance["zip_sha256"]),
        "file_dollar_year": int(provenance["file_dollar_year"]),
    }


def published_gaps(
    calibration: dict[str, float | int],
    replication: dict[str, float | int],
    *,
    calibration_year: int = 2022,
    replication_year: int = 2025,
) -> dict[str, dict[str, float | bool]]:
    """Compare two waves with the Fed's published figures.

    For each statistic the relative gap is ``public / published - 1``. The
    tolerance applied to the replication wave is the larger of a fixed floor
    (one percent for the net-worth figures, a tenth of a percent for the
    weight total) and twice the calibration wave's absolute gap, so it is
    fixed by the 2022 wave before the 2025 values are looked at.

    Args:
        calibration: :func:`population_summary` of the calibration wave.
        replication: :func:`population_summary` of the replication wave.
        calibration_year: Wave the tolerance is calibrated on.
        replication_year: Wave the tolerance is applied to.

    Returns:
        Statistic -> published and public values for both waves, both relative
        gaps, the tolerance, and whether the replication wave is within it.
    """

    def published(statistic: str, year: int) -> float:
        if statistic == "weight_total":
            return PUBLISHED_FAMILIES_MILLIONS[year] * 1e6
        key = statistic.removeprefix("net_worth_")
        return PUBLISHED_NET_WORTH_THOUSANDS[year][key] * 1e3

    result: dict[str, dict[str, float | bool]] = {}
    for statistic, floor in (
        ("weight_total", _MIN_WEIGHT_TOLERANCE),
        ("net_worth_median", _MIN_NET_WORTH_TOLERANCE),
        ("net_worth_mean", _MIN_NET_WORTH_TOLERANCE),
    ):
        calibration_gap = (
            float(calibration[statistic]) / published(statistic, calibration_year) - 1
        )
        replication_gap = (
            float(replication[statistic]) / published(statistic, replication_year) - 1
        )
        tolerance = max(floor, 2 * abs(calibration_gap))
        result[statistic] = {
            "published_calibration": published(statistic, calibration_year),
            "public_calibration": float(calibration[statistic]),
            "gap_calibration": calibration_gap,
            "published_replication": published(statistic, replication_year),
            "public_replication": float(replication[statistic]),
            "gap_replication": replication_gap,
            "tolerance": tolerance,
            "within_tolerance": abs(replication_gap) <= tolerance,
        }
    return result
