"""SCF wealth task: net worth and debt from the Survey of Consumer Finances.

Loads the Federal Reserve's *Summary Extract Public Data* for an SCF wave and
reduces it to the ``scf_wealth`` :class:`~imputation_paper.data.base.TaskFrame`:
household demographics and income as predictors, ``debt`` and ``networth`` as
targets. The two targets are chosen deliberately for the regime-gate story --
``debt`` is zero-inflated and nonnegative, ``networth`` is sign-mixed (a
meaningful share of households are underwater) -- so a method that ignores those
regimes is visibly penalised.

The summary extract stacks the SCF's five multiply-imputed *implicates* per
household; :func:`load_scf` keeps implicate 1 only (see its docstring for the
overflow-safe implicate filter) and :func:`load_scf_implicates` returns all
five.

**Two waves, pinned files, a stated dollar basis.** The paper's experiments run
on the 2022 wave; the 2025 wave is the replication of
``docs/scf-2025-replication-criteria.md``. The Fed serves one file per wave at
a fixed URL and *replaces* it when a new wave comes out, re-basing every dollar
column to the new wave's dollars: on 2026-10-09 ``scfp2022s.zip`` changed from
2022 dollars to 2025 dollars. A loader that trusts the URL therefore returns
different numbers depending on when its cache was filled. This one does not:
every file it accepts is registered in :data:`KNOWN_EXTRACTS` by SHA-256 with
its dollar basis, and the returned frame is in the dollars the caller asks for
(the survey year's by default, which is what the paper's 2022 runs used),
re-based with the Fed's own CPI-U-RS factors when the cached file is on a
different basis.
"""

from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from imputation_paper.data.base import TaskFrame, download, sha256_of_file

__all__ = [
    "CPI_U_RS_SEPTEMBER",
    "DOLLAR_COLUMNS",
    "KNOWN_EXTRACTS",
    "ExtractVintage",
    "load_scf",
    "load_scf_implicates",
    "rebase_dollars",
    "rebase_factor",
]

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
#: ``race`` is the first response to the general race question in both waves;
#: from 2025 the Fed folds the Middle Eastern or North African response into
#: class 1.
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

#: SCF weight column in the summary extract. The Fed defines it as the
#: analysis weight divided by five, so statistics pooled over the five
#: implicates are population-scaled.
_WEIGHT_COLUMN = "wgt"

#: Case-and-implicate id (``yy1 * 10 + implicate``) and case id, respectively.
_IMPLICATE_ID = "y1"
_CASE_ID = "yy1"

#: Total assets; identically ``fin + nfin``. Carried by
#: :func:`load_scf_implicates` so the balance-sheet identity can be checked.
_ASSET_COLUMN = "asset"

#: The dollar-denominated columns among those this module returns. These, and
#: only these, move when an extract is re-based.
DOLLAR_COLUMNS: tuple[str, ...] = (
    "income",
    "wageinc",
    "fin",
    "nfin",
    "debt",
    "networth",
    _ASSET_COLUMN,
)

#: CPI-U-RS (times 100) for September of each survey year, as set in the Fed's
#: ``bulletin.macro.txt`` (``CPIADJ = CPIBASE / value``; the 2026-10-09 release
#: calls the macro with ``CPIBASE=4789``). The Fed re-bases every extract to
#: the newest wave's dollars with the ratio of two of these.
CPI_U_RS_SEPTEMBER: dict[int, int] = {2022: 4376, 2025: 4789}


@dataclass(frozen=True)
class ExtractVintage:
    """One published summary-extract file.

    Attributes:
        survey_year: The SCF wave the file holds.
        dollar_year: The year whose dollars its dollar columns are in.
        description: When the Fed served it, for error messages and manifests.
    """

    survey_year: int
    dollar_year: int
    description: str


#: Every summary-extract zip this loader accepts, by SHA-256. A file that is
#: not here is refused: its dollar basis cannot be read off the file.
KNOWN_EXTRACTS: dict[str, ExtractVintage] = {
    "3bb4d890ae2463ff6039ec7692e375f544dd98a55a37ca2cb2340354b9cc9d80": ExtractVintage(
        2022,
        2022,
        "SCF 2022 in 2022 dollars (rscfp2022.dta dated 2024-03-29), served "
        "until 2026-10-09; the input of the paper's committed 2022 runs",
    ),
    "d4e139c69f6acf72c47512c7a204ae9a38d263781a9f0f51cdaecdf101d2d657": ExtractVintage(
        2022,
        2025,
        "SCF 2022 re-based to 2025 dollars (rscfp2022.dta dated 2026-10-05), "
        "served from 2026-10-09",
    ),
    "32dcf86be48b20fb5fd9e659f79fa22a4d9cccf456e44b6ca9b1329e9869bbeb": ExtractVintage(
        2025,
        2025,
        "SCF 2025 in 2025 dollars (rscfp2025.dta dated 2026-10-05), first "
        "release, served from 2026-10-09",
    ),
}


def rebase_factor(from_year: int, to_year: int) -> float:
    """The Fed's multiplier taking ``from_year`` dollars to ``to_year`` dollars.

    Args:
        from_year: Dollar basis of the values in hand.
        to_year: Dollar basis wanted.

    Returns:
        ``CPI_U_RS_SEPTEMBER[to_year] / CPI_U_RS_SEPTEMBER[from_year]``, and
        exactly ``1.0`` when the two years are equal.

    Raises:
        ValueError: If either year has no registered index value.
    """
    unknown = sorted({from_year, to_year} - set(CPI_U_RS_SEPTEMBER))
    if unknown:
        raise ValueError(
            f"No CPI-U-RS value registered for {unknown}; registered years: "
            f"{sorted(CPI_U_RS_SEPTEMBER)}."
        )
    if from_year == to_year:
        return 1.0
    return CPI_U_RS_SEPTEMBER[to_year] / CPI_U_RS_SEPTEMBER[from_year]


def rebase_dollars(frame: pd.DataFrame, from_year: int, to_year: int) -> pd.DataFrame:
    """Re-base the dollar columns of ``frame`` from one year's dollars to another.

    Only columns named in :data:`DOLLAR_COLUMNS` are touched; weights, ids, and
    categorical columns are returned as they came. With equal years the frame
    is returned unmultiplied, so values are bit-for-bit the file's.

    Args:
        frame: Any frame carrying some of :data:`DOLLAR_COLUMNS`.
        from_year: Dollar basis of ``frame``.
        to_year: Dollar basis wanted.

    Returns:
        A new frame on the requested basis.
    """
    factor = rebase_factor(from_year, to_year)
    result = frame.copy()
    if factor == 1.0:
        return result
    for column in DOLLAR_COLUMNS:
        if column in result.columns:
            result[column] = result[column].astype(np.float64) * factor
    return result


def _read_extract(zip_path: Path, year: int) -> tuple[pd.DataFrame, dict[str, object]]:
    """Read one registered summary-extract zip; return its table and provenance.

    Raises:
        ValueError: If the file's SHA-256 is not in :data:`KNOWN_EXTRACTS`, if
            it holds a different wave than ``year``, or if the zip does not
            contain exactly one ``.dta``.
    """
    digest = sha256_of_file(zip_path)
    vintage = KNOWN_EXTRACTS.get(digest)
    if vintage is None:
        raise ValueError(
            f"{zip_path} has SHA-256 {digest}, which is not a registered SCF "
            "summary extract. The Fed replaces each wave's file when a new "
            "wave comes out, re-basing its dollar columns, so the dollar basis "
            "of an unregistered file is unknown. Check the file against the "
            "Fed's bulletin.macro.txt and register it in "
            "imputation_paper.data.scf.KNOWN_EXTRACTS."
        )
    if vintage.survey_year != year:
        raise ValueError(
            f"{zip_path} is the {vintage.survey_year} wave ({vintage.description}), "
            f"not the {year} wave that was asked for."
        )
    with zipfile.ZipFile(zip_path) as archive:
        dta_names = [n for n in archive.namelist() if n.lower().endswith(".dta")]
        if len(dta_names) != 1:
            raise ValueError(
                f"Expected exactly one .dta in {zip_path.name}, found {dta_names}."
            )
        with archive.open(dta_names[0]) as member:
            raw = pd.read_stata(io.BytesIO(member.read()), convert_categoricals=False)
    provenance: dict[str, object] = {
        "scf_year": year,
        "file": dta_names[0],
        "zip_sha256": digest,
        "file_dollar_year": vintage.dollar_year,
        "file_description": vintage.description,
    }
    return raw, provenance


def _implicates_from_zip(
    zip_path: Path, year: int, dollar_year: int | None
) -> tuple[pd.DataFrame, dict[str, object]]:
    """All implicates of the used columns from one zip, on ``dollar_year`` basis."""
    raw, provenance = _read_extract(zip_path, year)
    needed = (
        _IMPLICATE_ID,
        _CASE_ID,
        _WEIGHT_COLUMN,
        *_PREDICTORS,
        *_RICH_EXTRA_PREDICTORS,
        *_RICH_TARGETS,
        _ASSET_COLUMN,
    )
    missing = [c for c in needed if c not in raw.columns]
    if missing:
        raise KeyError(
            f"SCF summary extract for {year} is missing column(s) {missing}; "
            f"it has {len(raw.columns)} columns."
        )
    wanted = year if dollar_year is None else dollar_year
    file_dollar_year = int(provenance["file_dollar_year"])  # type: ignore[call-overload]
    # int64 cast defuses the int16 overflow in yy1 * 10 (see load_scf).
    implicate = raw[_IMPLICATE_ID].astype("int64") - raw[_CASE_ID].astype("int64") * 10
    frame = raw.loc[:, list(needed)].copy()
    frame.insert(2, "implicate", implicate)
    frame = rebase_dollars(frame, file_dollar_year, wanted)
    provenance = {
        **provenance,
        "dollar_year": wanted,
        "rebase_factor": rebase_factor(file_dollar_year, wanted),
    }
    return frame.reset_index(drop=True), provenance


def load_scf_implicates(
    year: int = 2022, *, dollar_year: int | None = None
) -> pd.DataFrame:
    """All five implicates of the columns the paper uses, for one SCF wave.

    This is the table the weight-total, balance-sheet-identity, and
    published-table checks run on. Statistics pooled over all rows with ``wgt``
    are population-scaled (``wgt`` is the analysis weight divided by five).

    Args:
        year: SCF survey year (2022 or 2025).
        dollar_year: Dollar basis of the returned dollar columns; ``None``
            means the survey year's own dollars.

    Returns:
        One row per household and implicate with ``y1``, ``yy1``,
        ``implicate`` (1..5), ``wgt``, the ten predictors, the four
        balance-sheet targets, and ``asset``. Provenance (file hash, dollar
        basis, re-basing factor) is in ``frame.attrs["provenance"]``.

    Raises:
        KeyError: If an expected summary-extract column is absent.
        ValueError: If the cached file is not a registered extract for ``year``.
    """
    url = _SCF_ZIP_URL.format(year=year)
    zip_path = download(url, f"scfp{year}s.zip")
    frame, provenance = _implicates_from_zip(zip_path, year, dollar_year)
    frame.attrs["provenance"] = {**provenance, "url": url}
    return frame


def load_scf(
    year: int = 2022,
    *,
    profile: str = "minimal",
    dollar_year: int | None = None,
) -> TaskFrame:
    """Load the SCF wealth task (implicate 1 of the summary extract).

    Downloads and caches the Fed summary-extract Stata zip, verifies it is a
    registered file, reads the single ``.dta`` it contains, keeps implicate 1,
    and reduces to the predictor/target columns on the requested dollar basis.

    **Implicate filter.** The extract encodes ``y1 = yy1 * 10 + implicate``. The
    raw columns are narrow integer types (``y1`` int32, ``yy1`` int16), so for
    households with ``yy1 >= 6554`` the product ``yy1 * 10`` overflows int16 and
    the naive ``y1 - yy1 * 10`` wraps around -- silently dropping ~29% of
    households. This loader casts both ids to int64 *before* subtracting, so the
    filter ``y1 - yy1 * 10 == 1`` selects all first implicates.

    Args:
        year: SCF survey year. 2022 is the wave the paper's experiments run
            on; 2025 is the replication wave. Other years are refused until
            their files are registered.
        profile: ``"minimal"`` (seven predictors, two targets) or
            ``"populace-scale"`` (ten predictors, four chain-ordered targets
            whose balance-sheet identity networth == fin + nfin - debt holds
            exactly in the extract).
        dollar_year: Dollar basis of the returned dollar columns. ``None``
            (the default) means the survey year's own dollars -- for 2022, the
            basis of the paper's committed runs, whichever file is cached.

    Returns:
        A :class:`~imputation_paper.data.base.TaskFrame` named ``"scf_wealth"``
        (minimal) or ``"scf_wealth_rich"`` (populace-scale), with the input
        file's hash and dollar basis in ``provenance``.

    Raises:
        KeyError: If an expected summary-extract column is absent.
        ValueError: If the profile is unknown, the cached file is not a
            registered extract for ``year``, or the zip does not contain
            exactly one ``.dta``.
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

    implicates = load_scf_implicates(year, dollar_year=dollar_year)
    provenance = dict(implicates.attrs["provenance"])
    first_implicate = implicates.loc[implicates["implicate"] == 1]

    used = [*predictors, *targets, _WEIGHT_COLUMN]
    frame = first_implicate.loc[:, used].astype(np.float64)
    frame = frame.dropna().reset_index(drop=True)

    notes = (
        f"Source: Federal Reserve SCF {year} Summary Extract Public Data "
        f"({provenance['url']}), file {provenance['file']}, SHA-256 "
        f"{provenance['zip_sha256']}.",
        f"Dollar columns in {provenance['dollar_year']} dollars (the file is in "
        f"{provenance['file_dollar_year']} dollars; re-basing factor "
        f"{provenance['rebase_factor']:.6f}).",
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
        provenance={**provenance, "profile": profile},
    )
