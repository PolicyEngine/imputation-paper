"""CPS tasks from the Census Bureau's ASEC public-use files, directly.

Both loaders read the CPS Annual Social and Economic Supplement CSVs straight
from ``census.gov`` (``asecpub{yy}csv.zip``) rather than any processed
artifact. Two rules motivate the direct source:

* **Independence.** The paper evaluates PolicyEngine-adjacent methods, so its
  task inputs must not flow through PolicyEngine's own data processing; the
  Census files are the neutral origin every pipeline shares.
* **No enhanced files, ever.** Derived products such as the enhanced CPS embed
  the very QRF imputations this paper evaluates -- scoring against them would
  be self-referential.

Two loaders are exposed:

* :func:`load_cps` -- person-level ``cps_components`` task. Adults only,
  predictors ``age, is_female, employment_income``, targets
  ``interest_income`` and ``dividend_income``. These are the paper's
  zero-inflated component surface, where gating and weighting matter most.
* :func:`load_cps_households` -- one row per household, the *receiver* table
  for the SCF->CPS wealth harness. Its columns mirror the SCF predictor
  semantics as closely as the ASEC allows.

**File conventions** (verified against ASEC 2024 and 2025): the person file
(``pppub{yy}.csv``) and household file (``hhpub{yy}.csv``) join on
``PH_SEQ``/``H_SEQ``; weights (``MARSUPWT``, ``HSUP_WGT``) carry two implied
decimals, so raw sums of roughly 33.2e9 persons and 13.2e9 households divide
by 100 to the actual ~332M persons and ~132M households; the household file
includes zero-weight shell records, which are dropped. ASEC year ``N``
reports income for calendar year ``N - 1``.
"""

from __future__ import annotations

import zipfile
from collections.abc import Sequence

import pandas as pd

from imputation_paper.data.base import TaskFrame, download

__all__ = ["load_cps", "load_cps_households"]

#: The Census ASEC public-use CSV bundle for a given ASEC (survey) year.
_URL_TEMPLATE = (
    "https://www2.census.gov/programs-surveys/cps/datasets/{year}/march/"
    "asecpub{yy}csv.zip"
)

#: ASEC public-use weights carry two implied decimal places.
_WEIGHT_DIVISOR = 100.0

# --- Person-file (pppub) variable names -------------------------------------
_P_AGE = "A_AGE"
_P_SEX = "A_SEX"  # 1 = male, 2 = female (same coding as SCF `hhsex`)
_P_MARITAL = "A_MARITL"  # 1-7; 1/2 = married, spouse present (civilian/AF)
_P_REL = "A_EXPRRP"  # 1/2 = reference person (with/without relatives)
_P_WAGE = "WSAL_VAL"  # wage and salary income, prior calendar year
_P_SELF_EMP = "SEMP_VAL"  # self-employment income, prior calendar year
_P_INTEREST = "INT_VAL"  # interest income, prior calendar year
_P_DIVIDEND = "DIV_VAL"  # dividend income, prior calendar year
_P_WEIGHT = "MARSUPWT"  # person supplement weight (two implied decimals)
_P_HH_SEQ = "PH_SEQ"  # household sequence number (joins H_SEQ)

# --- Rich-profile person variables (populace-scale shared predictors) -------
_P_EDUCATION = "A_HGA"  # educational attainment code (31..46 for adults)
_P_HISPANIC = "PEHSPNON"  # 1 = Hispanic, 2 = not
_P_RACE = "PRDTRACE"  # detailed race; 1 white, 2 Black, 4 Asian, else other
_P_LABOR_FORCE = "A_LFSR"  # labor force status recode; 1/2 = employed

# --- Household-file (hhpub) variable names ----------------------------------
_H_SEQ = "H_SEQ"
_H_TOTAL_INCOME = "HTOTVAL"  # total household income, prior calendar year
_H_WEIGHT = "HSUP_WGT"  # household supplement weight (two implied decimals)
_H_TENURE = "H_TENURE"  # 1 = owned, 2 = rented, 3 = occupied w/o payment


def _asec_member(year: int, member_prefix: str, usecols: Sequence[str]) -> pd.DataFrame:
    """Read one CSV member of the cached ASEC bundle for ``year``.

    Args:
        year: ASEC survey year (income reference year is ``year - 1``).
        member_prefix: ``"pppub"`` (person) or ``"hhpub"`` (household).
        usecols: Columns to load (the person file has ~800; never read all).

    Returns:
        The requested columns as a DataFrame.
    """
    yy = f"{year % 100:02d}"
    path = download(_URL_TEMPLATE.format(year=year, yy=yy), f"asecpub{yy}csv.zip")
    with zipfile.ZipFile(path) as bundle:
        with bundle.open(f"{member_prefix}{yy}.csv") as handle:
            return pd.read_csv(handle, usecols=list(usecols))


def load_cps(year: int = 2025) -> TaskFrame:
    """Person-level zero-inflated components task from the ASEC person file.

    Adults (18+) with positive supplement weight; ``employment_income`` is
    wage-and-salary plus self-employment income; interest and dividend income
    are the ASEC amounts for the prior calendar year.

    Args:
        year: ASEC survey year (default 2025, income year 2024).

    Returns:
        The ``cps_components`` :class:`TaskFrame`, weighted by the person
        supplement weight (implied decimals resolved).
    """
    persons = _asec_member(
        year,
        "pppub",
        [_P_AGE, _P_SEX, _P_WAGE, _P_SELF_EMP, _P_INTEREST, _P_DIVIDEND, _P_WEIGHT],
    )
    adults = persons[(persons[_P_AGE] >= 18) & (persons[_P_WEIGHT] > 0)]
    frame = pd.DataFrame(
        {
            "age": adults[_P_AGE],
            "is_female": (adults[_P_SEX] == 2).astype(float),
            "employment_income": adults[_P_WAGE] + adults[_P_SELF_EMP],
            "interest_income": adults[_P_INTEREST],
            "dividend_income": adults[_P_DIVIDEND],
            "person_weight": adults[_P_WEIGHT] / _WEIGHT_DIVISOR,
        }
    ).astype("float64")
    frame = frame.dropna().reset_index(drop=True)
    return TaskFrame(
        name="cps_components",
        frame=frame,
        predictors=("age", "is_female", "employment_income"),
        targets=("interest_income", "dividend_income"),
        weight_column="person_weight",
        notes=(
            f"Census ASEC {year} public-use person file (pppub{year % 100:02d}"
            f".csv) from census.gov; income reference year {year - 1}; adults "
            "18+; MARSUPWT/100 person weights (two implied decimals)."
        ),
    )


def _households_one_year(year: int, profile: str) -> pd.DataFrame:
    """The receiver table for a single ASEC year (see load_cps_households)."""
    person_cols = [_P_HH_SEQ, _P_AGE, _P_SEX, _P_MARITAL, _P_REL, _P_WAGE]
    household_cols = [_H_SEQ, _H_TOTAL_INCOME, _H_WEIGHT]
    if profile == "populace-scale":
        person_cols += [_P_EDUCATION, _P_HISPANIC, _P_RACE, _P_LABOR_FORCE]
        household_cols.append(_H_TENURE)
    persons = _asec_member(year, "pppub", person_cols)
    households = _asec_member(year, "hhpub", household_cols)
    households = households[households[_H_WEIGHT] > 0]

    reference = (
        persons[persons[_P_REL].isin((1, 2))]
        .drop_duplicates(subset=_P_HH_SEQ)
        .set_index(_P_HH_SEQ)
    )
    per_household = persons.groupby(_P_HH_SEQ).agg(
        kids=(_P_AGE, lambda ages: float((ages < 18).sum())),
        wageinc=(_P_WAGE, "sum"),
    )

    reference_cols = [_P_AGE, _P_SEX, _P_MARITAL]
    if profile == "populace-scale":
        reference_cols += [_P_EDUCATION, _P_HISPANIC, _P_RACE, _P_LABOR_FORCE]
    merged = (
        households.set_index(_H_SEQ)
        .join(reference[reference_cols], how="inner")
        .join(per_household, how="left")
    )
    columns = {
        "age": merged[_P_AGE],
        "hhsex": merged[_P_SEX],
        "married": merged[_P_MARITAL].isin((1, 2)).map({True: 1.0, False: 2.0}),
        "kids": merged["kids"].fillna(0.0),
        "income": merged[_H_TOTAL_INCOME],
        "wageinc": merged["wageinc"].fillna(0.0),
        "household_weight": merged[_H_WEIGHT] / _WEIGHT_DIVISOR,
    }
    if profile == "populace-scale":
        # SCF edcl classes: <39 no diploma; 39 HS/GED; 40-42 some college or
        # associate degree; 43+ bachelor's or higher.
        education = merged[_P_EDUCATION]
        columns["edcl"] = (
            1.0 * (education < 39)
            + 2.0 * (education == 39)
            + 3.0 * education.between(40, 42)
            + 4.0 * (education >= 43)
        )
        # SCF race classes: Hispanic overrides (3); white 1, Black 2, Asian 4;
        # everything else (incl. multi-race combinations) -> other (5).
        race = merged[_P_RACE]
        columns["race"] = (
            pd.Series(5.0, index=merged.index)
            .mask(race == 1, 1.0)
            .mask(race == 2, 2.0)
            .mask(race == 4, 4.0)
            .mask(merged[_P_HISPANIC] == 1, 3.0)
        )
        columns["housecl"] = merged[_H_TENURE].eq(1).map({True: 1.0, False: 2.0})
        columns["lf"] = merged[_P_LABOR_FORCE].isin((1, 2)).astype(float)
    frame = pd.DataFrame(columns).astype("float64")
    return frame.dropna().reset_index(drop=True)


def load_cps_households(
    years: int | tuple[int, ...] = 2025, *, profile: str = "minimal"
) -> pd.DataFrame:
    """One row per household: the SCF->CPS receiver, from the ASEC files.

    Columns mirror the SCF summary-extract predictor semantics:

    * ``age``, ``hhsex`` -- the reference person's age and sex (``A_EXPRRP``
      in {1, 2}; sex is 1/2 coded like SCF ``hhsex``).
    * ``married`` -- SCF-style 1/2 from the reference person's ``A_MARITL``:
      married with spouse present (codes 1-2) maps to 1, all else to 2. The
      SCF's code 1 also includes cohabiting partners, which ``A_MARITL``
      cannot see -- a documented approximation of the shared predictor.
    * ``kids`` -- household members under 18.
    * ``income`` -- ``HTOTVAL``, total household income (prior calendar year).
    * ``wageinc`` -- household sum of wage-and-salary income.
    * ``household_weight`` -- ``HSUP_WGT``/100 (zero-weight shell records in
      the household file are dropped).

    The ``"populace-scale"`` profile adds the widened SCF-shared predictors:
    ``edcl`` (from ``A_HGA``), ``race`` (Hispanic overrides; white/Black/Asian
    mapped, combinations to other -- SCF race classes), ``housecl`` (owned vs
    not from ``H_TENURE``), and ``lf`` (employed from ``A_LFSR``).

    Args:
        years: One ASEC survey year, or a tuple of years to pool. Pooling
            mirrors the populace support spine's multi-year design: each
            year's household weights are divided by the number of pooled
            years, so the pooled file still represents one US household
            population.
        profile: ``"minimal"`` or ``"populace-scale"``.

    Returns:
        The receiver DataFrame (float64 columns, no NaNs).
    """
    if profile not in ("minimal", "populace-scale"):
        raise ValueError(
            f"Unknown profile {profile!r}; expected 'minimal' or 'populace-scale'."
        )
    year_tuple = (years,) if isinstance(years, int) else tuple(years)
    frames = [_households_one_year(year, profile) for year in year_tuple]
    pooled = pd.concat(frames, ignore_index=True)
    pooled["household_weight"] = pooled["household_weight"] / len(year_tuple)
    return pooled
