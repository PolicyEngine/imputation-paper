"""SCF wealth task: net worth and debt from the Survey of Consumer Finances.

Loads the Federal Reserve's *Summary Extract Public Data* for the 2022 SCF and
reduces it to the ``scf_wealth`` :class:`~imputation_paper.data.base.TaskFrame`:
household demographics and income as predictors, ``debt`` and ``networth`` as
targets. The two targets are chosen deliberately for the regime-gate story --
``debt`` is zero-inflated and nonnegative, ``networth`` is sign-mixed (a
meaningful share of households are underwater) -- so a method that ignores those
regimes is visibly penalised.

The summary extract stacks the SCF's five multiply-imputed *implicates* per
household; this loader keeps implicate 1 only. See :func:`load_scf` for the
overflow-safe implicate filter.
"""

from __future__ import annotations

import io
import zipfile

import numpy as np
import pandas as pd

from imputation_paper.data.base import TaskFrame, download

__all__ = ["load_scf"]

#: The Fed summary-extract public-data Stata zip, by survey year.
_SCF_ZIP_URL = "https://www.federalreserve.gov/econres/files/scfp{year}s.zip"

#: Minimal-profile predictors: household head demographics plus income (all
#: standard SCFP summary-extract variable names, lowercase in the extract).
_PREDICTORS: tuple[str, ...] = (
    "age",  # age of the household head/reference person
    "hhsex",  # sex of the head (1 = male, 2 = female)
    "edcl",  # education class of the head (1..4)
    "married",  # 1 = married/living with partner, 2 = otherwise
    "kids",  # number of children in the household
    "income",  # total household income
    "wageinc",  # household wage and salary income
)

#: Rich-profile additions (all mappable onto Census ASEC variables, so the
#: SCF->CPS shared set keeps them): race class, homeownership, labor force.
_RICH_EXTRA_PREDICTORS: tuple[str, ...] = (
    "race",  # 1 white non-Hisp, 2 Black, 3 Hispanic, 4 Asian, 5 other
    "housecl",  # 1 = owns (incl. mortgaged), 2 = otherwise
    "lf",  # 1 = working in some way, 0 = not working
)

#: Minimal-profile targets: the sign/zero regimes the paper's gates capture.
_TARGETS: tuple[str, ...] = (
    "debt",  # total debt: zero-inflated, nonnegative
    "networth",  # net worth: sign-mixed (negative for underwater households)
)

#: Rich-profile targets, in chain order. The extract satisfies
#: networth == fin + nfin - debt exactly, so a chained imputer can in
#: principle draw a self-consistent balance sheet while independent
#: imputation cannot -- the populace-scale chaining test.
_RICH_TARGETS: tuple[str, ...] = (
    "fin",  # financial assets: nonnegative, mildly zero-inflated
    "nfin",  # nonfinancial assets: nonnegative, zero for non-owners
    "debt",  # total debt: zero-inflated, nonnegative
    "networth",  # net worth: sign-mixed; identically fin + nfin - debt
)

#: SCF weight column in the summary extract.
_WEIGHT_COLUMN = "wgt"

#: Case-and-implicate id (``yy1 * 10 + implicate``) and case id, respectively.
_IMPLICATE_ID = "y1"
_CASE_ID = "yy1"


def load_scf(year: int = 2022, *, profile: str = "minimal") -> TaskFrame:
    """Load the SCF wealth task (implicate 1 of the summary extract).

    Downloads and caches the Fed summary-extract Stata zip, reads the single
    ``.dta`` it contains, keeps implicate 1, and reduces to the predictor/target
    columns.

    **Implicate filter.** The extract encodes ``y1 = yy1 * 10 + implicate``. The
    raw columns are narrow integer types (``y1`` int32, ``yy1`` int16), so for
    households with ``yy1 >= 6554`` the product ``yy1 * 10`` overflows int16 and
    the naive ``y1 - yy1 * 10`` wraps around -- silently dropping ~29% of
    households. This loader casts both ids to int64 *before* subtracting, so the
    filter ``y1 - yy1 * 10 == 1`` selects all first implicates.

    Args:
        year: SCF survey year. Only 2022 is exercised by the paper; other years
            share the summary-extract schema but are not tested here.
        profile: ``"minimal"`` (seven predictors, two targets) or
            ``"populace-scale"`` (ten predictors, four chain-ordered targets
            whose balance-sheet identity networth == fin + nfin - debt holds
            exactly in the extract).

    Returns:
        A :class:`~imputation_paper.data.base.TaskFrame` named ``"scf_wealth"``.

    Raises:
        KeyError: If an expected summary-extract column is absent.
        ValueError: If the zip does not contain exactly one ``.dta``.
    """
    if profile == "minimal":
        predictors, targets = _PREDICTORS, _TARGETS
    elif profile == "populace-scale":
        predictors = (*_PREDICTORS, *_RICH_EXTRA_PREDICTORS)
        targets = _RICH_TARGETS
    else:
        raise ValueError(
            f"Unknown profile {profile!r}; expected 'minimal' or 'populace-scale'."
        )

    url = _SCF_ZIP_URL.format(year=year)
    zip_path = download(url, f"scfp{year}s.zip")

    with zipfile.ZipFile(zip_path) as archive:
        dta_names = [n for n in archive.namelist() if n.lower().endswith(".dta")]
        if len(dta_names) != 1:
            raise ValueError(
                f"Expected exactly one .dta in {zip_path.name}, found {dta_names}."
            )
        with archive.open(dta_names[0]) as member:
            raw = pd.read_stata(io.BytesIO(member.read()), convert_categoricals=False)

    needed = (_IMPLICATE_ID, _CASE_ID, _WEIGHT_COLUMN, *predictors, *targets)
    missing = [c for c in needed if c not in raw.columns]
    if missing:
        raise KeyError(
            f"SCF summary extract for {year} is missing column(s) {missing}; "
            f"it has {len(raw.columns)} columns."
        )

    # int64 cast defuses the int16 overflow in yy1 * 10 (see docstring).
    implicate = raw[_IMPLICATE_ID].astype("int64") - raw[_CASE_ID].astype("int64") * 10
    first_implicate = raw.loc[implicate == 1]

    used = [*predictors, *targets, _WEIGHT_COLUMN]
    frame = first_implicate.loc[:, used].astype(np.float64)
    frame = frame.dropna().reset_index(drop=True)

    notes = (
        f"Source: Federal Reserve SCF {year} Summary Extract Public Data "
        f"({url}), file {dta_names[0]}.",
        "Kept implicate 1 via int64-cast filter y1 - yy1*10 == 1 (guards the "
        "int16 overflow in yy1*10 for yy1 >= 6554).",
        "Targets chosen for the regime-gate story: debt is zero-inflated and "
        "nonnegative; networth is sign-mixed.",
    )
    return TaskFrame(
        name="scf_wealth" if profile == "minimal" else "scf_wealth_rich",
        frame=frame,
        predictors=predictors,
        targets=targets,
        weight_column=_WEIGHT_COLUMN,
        notes=(*notes, f"Profile: {profile}."),
    )
