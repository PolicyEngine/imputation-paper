"""SIPP task (optional): zero-inflated asset/income components from SIPP 2022.

This loader reads a *local* SIPP 2022 public-use file. It is optional: as of
writing the expected path is a dangling symlink with no data behind it, so
:func:`load_sipp` raises :class:`FileNotFoundError` when called. It is kept (and
wired for the real file's column layout) so the task activates the moment the
file is present -- but it never fabricates data and never silently substitutes a
synthetic table.

The SIPP PU is a monthly person file; this loader keeps one record per person
(the December reference month, ``MONTHCODE == 12``) to avoid counting each
person twelve times.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from imputation_paper.data.base import TaskFrame

__all__ = ["load_sipp", "SIPP_PATH"]

#: Local SIPP 2022 public-use CSV. As of writing this is a dangling symlink.
SIPP_PATH = Path("/Users/maxghenis/CosilicoAI/sipp/data/pu2022.csv")

#: Identifiers and the reference-month selector.
_SSUID = "SSUID"
_PNUM = "PNUM"
_MONTHCODE = "MONTHCODE"
_DECEMBER = 12

#: Predictors: age, sex, and total person income.
_PREDICTORS: tuple[str, ...] = ("TAGE", "ESEX", "TPTOTINC")
#: Targets: two zero-inflated income components (interest and dividends).
_TARGETS: tuple[str, ...] = ("TINTINC", "TDIVINC")
#: SIPP final person weight.
_WEIGHT_COLUMN = "WPFINWGT"

#: The columns actually read (keeps the multi-GB file's memory bounded).
_USECOLS: tuple[str, ...] = (
    _SSUID,
    _PNUM,
    _MONTHCODE,
    *_PREDICTORS,
    *_TARGETS,
    _WEIGHT_COLUMN,
)


def load_sipp() -> TaskFrame:
    """Load the SIPP component task from the local 2022 public-use file.

    Reads only :data:`_USECOLS` (the file is large), keeps the December
    reference month per person, coerces used columns to float64, drops NaN rows,
    and returns the ``sipp_components`` task.

    Returns:
        A :class:`~imputation_paper.data.base.TaskFrame` named
        ``"sipp_components"``.

    Raises:
        FileNotFoundError: If :data:`SIPP_PATH` does not resolve to a real file
            (currently the case -- the path is a dangling symlink). No synthetic
            fallback is ever substituted.
        KeyError: If an expected SIPP column is absent from the file.
    """
    if not SIPP_PATH.exists():
        raise FileNotFoundError(
            f"SIPP file {SIPP_PATH} is not present (the path is a dangling "
            "symlink with no data behind it). The SIPP task is skipped until a "
            "real SIPP 2022 public-use file is available; no synthetic data is "
            "substituted."
        )

    frame = pd.read_csv(
        SIPP_PATH,
        sep="|",
        usecols=list(_USECOLS),
    )
    missing = [c for c in _USECOLS if c not in frame.columns]
    if missing:
        raise KeyError(f"SIPP file is missing column(s) {missing}.")

    december = frame.loc[frame[_MONTHCODE] == _DECEMBER]
    # Defensive dedupe on (SSUID, PNUM) in case a person appears twice.
    december = december.drop_duplicates(subset=[_SSUID, _PNUM], keep="first")

    used = [*_PREDICTORS, *_TARGETS, _WEIGHT_COLUMN]
    reduced = december.loc[:, used].astype(np.float64).dropna().reset_index(drop=True)

    notes = (
        f"Source: local SIPP 2022 public-use file {SIPP_PATH}.",
        "Kept the December reference month (MONTHCODE == 12), one record per person.",
        "Targets: TINTINC (interest) and TDIVINC (dividends), zero-inflated.",
    )
    return TaskFrame(
        name="sipp_components",
        frame=reduced,
        predictors=_PREDICTORS,
        targets=_TARGETS,
        weight_column=_WEIGHT_COLUMN,
        notes=notes,
    )
