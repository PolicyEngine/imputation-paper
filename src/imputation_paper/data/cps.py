"""CPS tasks: within-CPS zero-inflated components, and the SCF->CPS receiver.

Both loaders read the *raw* CPS file the PolicyEngine data pipeline starts from
(``cps_2023.h5`` on the Hugging Face Hub). This is a hard rule: the loaders must
never read ``enhanced_cps*``, because the enhanced file already embeds the very
QRF imputations this paper evaluates -- scoring against it would be
self-referential.

Two tasks are exposed:

* :func:`load_cps` -- person-level ``cps_components`` task. Adults only,
  predictors ``age, is_female, employment_income``, targets ``interest_income``
  and ``dividend_income`` (each summed from the file's component variables).
  These are the paper's zero-inflated component surface, where gating and
  weighting matter most.
* :func:`load_cps_households` -- one row per household, the *receiver* table for
  the SCF->CPS wealth harness. Its columns mirror the SCF predictor semantics as
  closely as the CPS allows.

**h5 layout.** This file keys datasets by bare PolicyEngine variable name (e.g.
``age``, not ``age/2023``); each key is a 1-D array whose length is its entity
count -- 50863 persons, 20655 households. Variables are joined across entities by
the ``person_*_id`` / ``*_id`` columns. See the module constants for the exact
names used and the derivations documented on each loader.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from imputation_paper.data.base import TaskFrame

if TYPE_CHECKING:  # pragma: no cover - typing only
    import h5py

__all__ = ["load_cps", "load_cps_households"]

#: The raw CPS file on the Hub. NEVER an ``enhanced_cps*`` file (those embed the
#: QRF imputations this paper evaluates; reading them would be self-referential).
_HF_REPO_ID = "policyengine/policyengine-us-data"
_HF_FILENAME = "cps_2023.h5"

# --- Person-level variable names (length 50863) -----------------------------
_V_AGE = "age"
_V_IS_FEMALE = "is_female"
_V_EMPLOYMENT_INCOME = "employment_income"
#: Interest components. In this file ``tax_exempt_interest_income`` is a fixed
#: multiple of ``taxable_interest_income`` on the same support; their sum is a
#: valid "total interest received" concept sharing the components' zero pattern.
_V_INTEREST_PARTS = ("taxable_interest_income", "tax_exempt_interest_income")
#: Dividend components (qualified + non-qualified) -- a genuine split summing to
#: total dividend income, sharing one nonzero support.
_V_DIVIDEND_PARTS = ("qualified_dividend_income", "non_qualified_dividend_income")
_V_IS_HOUSEHOLD_HEAD = "is_household_head"
_V_PERSON_HOUSEHOLD_ID = "person_household_id"
_V_PERSON_MARITAL_UNIT_ID = "person_marital_unit_id"
_V_SELF_EMPLOYMENT_INCOME = "self_employment_income"

# --- Household-level variable names (length 20655) --------------------------
_V_HOUSEHOLD_ID = "household_id"
_V_HOUSEHOLD_WEIGHT = "household_weight"

#: Adults only, matching the paper's component surface.
_ADULT_AGE = 18

#: Derived clean target column names.
_INTEREST_INCOME = "interest_income"
_DIVIDEND_INCOME = "dividend_income"

_CPS_COMPONENTS_PREDICTORS: tuple[str, ...] = (
    _V_AGE,
    _V_IS_FEMALE,
    _V_EMPLOYMENT_INCOME,
)
_CPS_COMPONENTS_TARGETS: tuple[str, ...] = (_INTEREST_INCOME, _DIVIDEND_INCOME)
_PERSON_WEIGHT = "person_weight"

#: SCF->CPS shared predictor list (the intersection of what both surveys carry).
#: ``edcl`` (SCF education class) has no clean analogue in this CPS file, so it
#: is omitted; ``income``/``wageinc`` are household aggregates matching the SCF
#: household-income semantics.
_HOUSEHOLD_SHARED_PREDICTORS: tuple[str, ...] = (
    "age",  # household head's age
    "hhsex",  # head sex, 1 = male / 2 = female (SCF coding)
    "married",  # 1 = head in a 2-person marital unit, 2 = otherwise (SCF coding)
    "kids",  # count of household members under 18
    "income",  # household total income (sum of person incomes)
    "wageinc",  # household wage/employment income (sum of person employment income)
)


def _download_raw_cps() -> str:
    """Return the local path to the raw CPS h5, fetching it if needed.

    Tries the Hub anonymously first (the repo is public). Only on an
    authorization failure does it retry with a token from the environment, so
    the common path needs no credentials.

    Returns:
        The local filesystem path to ``cps_2023.h5``.
    """
    import logging
    import os

    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError

    logger = logging.getLogger(__name__)
    if "enhanced" in _HF_FILENAME:  # pragma: no cover - guards a future edit
        raise ValueError(
            "Refusing to load an enhanced CPS file: it embeds prior QRF "
            "imputations, which would make this benchmark self-referential."
        )
    try:
        path = hf_hub_download(repo_id=_HF_REPO_ID, filename=_HF_FILENAME)
        logger.info("Fetched %s anonymously.", _HF_FILENAME)
        return path
    except (GatedRepoError, RepositoryNotFoundError, PermissionError) as exc:
        token = os.environ.get("HUGGING_FACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")
        if not token:
            raise RuntimeError(
                f"Anonymous download of {_HF_FILENAME} failed ({exc}); set "
                "HUGGING_FACE_HUB_TOKEN to authenticate."
            ) from exc
        path = hf_hub_download(repo_id=_HF_REPO_ID, filename=_HF_FILENAME, token=token)
        logger.info("Fetched %s with HUGGING_FACE_HUB_TOKEN.", _HF_FILENAME)
        return path


def _read_person_columns(store: h5py.File, names: tuple[str, ...]) -> pd.DataFrame:
    """Read named person-level datasets into a float64 frame."""
    return pd.DataFrame({name: store[name][:].astype(np.float64) for name in names})


def _person_household_weight(
    store: h5py.File, person_household_id: np.ndarray
) -> np.ndarray:
    """Map each person to their household's weight.

    The raw CPS file stores no per-person weight -- only ``household_weight``.
    PolicyEngine's convention is that a person inherits their household's weight,
    so this joins ``household_weight`` onto persons by household id. Every person
    resolves to exactly one household weight.

    Args:
        store: The open h5 file.
        person_household_id: Person-level household ids (length = n persons).

    Returns:
        A float64 person-level weight vector.

    Raises:
        KeyError: If a person references a household id absent from the
            household table.
    """
    household_id = store[_V_HOUSEHOLD_ID][:]
    household_weight = store[_V_HOUSEHOLD_WEIGHT][:].astype(np.float64)
    weight_by_household = dict(
        zip(household_id.tolist(), household_weight.tolist(), strict=True)
    )
    try:
        return np.array(
            [weight_by_household[h] for h in person_household_id.tolist()],
            dtype=np.float64,
        )
    except KeyError as exc:  # pragma: no cover - defensive; ids are consistent
        raise KeyError(
            f"Person references household id {exc} absent from the household "
            "table; the CPS file's entity ids are inconsistent."
        ) from None


def _married_from_marital_unit(person_marital_unit_id: np.ndarray) -> np.ndarray:
    """Derive an SCF-style ``married`` flag from marital-unit membership.

    The raw CPS file has no marital-status field; it groups people into marital
    units via ``person_marital_unit_id``. A unit with two members is a
    married/partnered couple; a singleton unit is an unmarried person. This
    returns the SCF ``married`` coding: ``1`` for a person in a 2-member unit,
    ``2`` otherwise -- matching the SCF summary extract's
    ``married in {1: married/living with partner, 2: otherwise}``.

    Args:
        person_marital_unit_id: Person-level marital-unit ids.

    Returns:
        An int-valued (as float64 downstream) array of 1/2 codes, per person.
    """
    unit = pd.Series(person_marital_unit_id)
    unit_size = unit.map(unit.value_counts())
    return np.where(unit_size.to_numpy() >= 2, 1, 2)


def load_cps(year: int = 2023) -> TaskFrame:
    """Load the ``cps_components`` task: zero-inflated interest and dividends.

    Person-level, adults (``age >= 18``) only. Predictors are ``age``,
    ``is_female`` and ``employment_income``; targets are ``interest_income`` (sum
    of the file's taxable and tax-exempt interest components) and
    ``dividend_income`` (sum of qualified and non-qualified dividends). The
    weight column is ``person_weight``, derived from ``household_weight`` (this
    raw file carries no per-person weight; a person inherits its household's).

    Args:
        year: CPS data year. Only 2023 (``cps_2023.h5``) is exercised.

    Returns:
        A :class:`~imputation_paper.data.base.TaskFrame` named
        ``"cps_components"``.
    """
    import h5py

    if year != 2023:  # pragma: no cover - single supported file
        raise ValueError(f"Only the 2023 raw CPS file is supported, got {year}.")

    path = _download_raw_cps()
    with h5py.File(path, "r") as store:
        base = _read_person_columns(store, _CPS_COMPONENTS_PREDICTORS)
        interest = sum(store[p][:].astype(np.float64) for p in _V_INTEREST_PARTS)
        dividend = sum(store[p][:].astype(np.float64) for p in _V_DIVIDEND_PARTS)
        base[_INTEREST_INCOME] = interest
        base[_DIVIDEND_INCOME] = dividend
        base[_PERSON_WEIGHT] = _person_household_weight(
            store, store[_V_PERSON_HOUSEHOLD_ID][:]
        )

    adults = base.loc[base[_V_AGE] >= _ADULT_AGE]
    used = [*_CPS_COMPONENTS_PREDICTORS, *_CPS_COMPONENTS_TARGETS, _PERSON_WEIGHT]
    frame = adults.loc[:, used].astype(np.float64).dropna().reset_index(drop=True)

    notes = (
        f"Source: raw {_HF_FILENAME} from {_HF_REPO_ID} on the Hugging Face Hub "
        "(NOT the enhanced CPS, which embeds prior QRF imputations).",
        "Adults only (age >= 18).",
        "interest_income = taxable_interest_income + tax_exempt_interest_income; "
        "dividend_income = qualified_dividend_income + non_qualified_dividend_income.",
        "person_weight derived from household_weight (raw file has no per-person "
        "weight; a person inherits its household's).",
    )
    return TaskFrame(
        name="cps_components",
        frame=frame,
        predictors=_CPS_COMPONENTS_PREDICTORS,
        targets=_CPS_COMPONENTS_TARGETS,
        weight_column=_PERSON_WEIGHT,
        notes=notes,
    )


def load_cps_households(year: int = 2023) -> pd.DataFrame:
    """Build the household receiver table for the SCF->CPS wealth harness.

    One row per household, with columns matching the SCF predictor semantics as
    closely as the CPS allows (the shared predictor set is
    :data:`_HOUSEHOLD_SHARED_PREDICTORS`, i.e. the SCF predictors minus
    ``edcl``, which has no clean CPS analogue in this file). Household-level
    aggregates are built from the person table:

    * ``age`` -- the household head's age (``is_household_head``), falling back to
      the eldest adult if a household has no flagged head;
    * ``hhsex`` -- the head's sex recoded to SCF's ``1 = male / 2 = female``;
    * ``married`` -- ``1`` if the head sits in a 2-person marital unit, else
      ``2`` (SCF coding), derived from ``person_marital_unit_id``;
    * ``kids`` -- count of household members under 18;
    * ``income`` -- household total income (sum over persons of employment,
      self-employment, interest and dividend income);
    * ``wageinc`` -- household employment income (sum over persons);
    * ``household_weight`` -- the household's weight.

    Args:
        year: CPS data year. Only 2023 (``cps_2023.h5``) is exercised.

    Returns:
        A household-level :class:`~pandas.DataFrame` with the shared predictor
        columns plus ``household_weight``, float64, NaN-free, ``RangeIndex``.
    """
    import h5py

    if year != 2023:  # pragma: no cover - single supported file
        raise ValueError(f"Only the 2023 raw CPS file is supported, got {year}.")

    path = _download_raw_cps()
    with h5py.File(path, "r") as store:
        persons = pd.DataFrame(
            {
                "household_id": store[_V_PERSON_HOUSEHOLD_ID][:],
                "age": store[_V_AGE][:].astype(np.float64),
                "is_female": store[_V_IS_FEMALE][:].astype(bool),
                "is_head": store[_V_IS_HOUSEHOLD_HEAD][:].astype(bool),
                "employment_income": store[_V_EMPLOYMENT_INCOME][:].astype(np.float64),
                "self_employment_income": store[_V_SELF_EMPLOYMENT_INCOME][:].astype(
                    np.float64
                ),
                "interest_income": sum(
                    store[p][:].astype(np.float64) for p in _V_INTEREST_PARTS
                ),
                "dividend_income": sum(
                    store[p][:].astype(np.float64) for p in _V_DIVIDEND_PARTS
                ),
                "married_code": _married_from_marital_unit(
                    store[_V_PERSON_MARITAL_UNIT_ID][:]
                ),
            }
        )
        household_id = store[_V_HOUSEHOLD_ID][:]
        household_weight = store[_V_HOUSEHOLD_WEIGHT][:].astype(np.float64)

    persons["total_income"] = (
        persons["employment_income"]
        + persons["self_employment_income"]
        + persons["interest_income"]
        + persons["dividend_income"]
    )

    heads = _select_household_heads(persons)
    aggregates = _aggregate_household_totals(persons)

    weights = pd.DataFrame(
        {"household_id": household_id, "household_weight": household_weight}
    )
    household = (
        weights.merge(heads, on="household_id", how="inner")
        .merge(aggregates, on="household_id", how="inner")
        .loc[
            :,
            [
                "age",
                "hhsex",
                "married",
                "kids",
                "income",
                "wageinc",
                "household_weight",
            ],
        ]
    )
    return household.astype(np.float64).dropna().reset_index(drop=True)


def _select_household_heads(persons: pd.DataFrame) -> pd.DataFrame:
    """Pick one head row per household and derive head-level SCF predictors.

    Uses the ``is_head`` flag; for any household without a flagged head, falls
    back to its eldest person. Returns ``household_id, age, hhsex, married``
    (the last two in SCF coding).
    """
    flagged = persons.loc[persons["is_head"]]
    # Households with no flagged head: fall back to the eldest member.
    headless = set(persons["household_id"]) - set(flagged["household_id"])
    if headless:
        fallback = (
            persons.loc[persons["household_id"].isin(headless)]
            .sort_values(["household_id", "age"], ascending=[True, False])
            .groupby("household_id", as_index=False)
            .first()
        )
        flagged = pd.concat([flagged, fallback], ignore_index=True)
    # One row per household (defensive against multi-head households).
    flagged = flagged.groupby("household_id", as_index=False).first()

    return pd.DataFrame(
        {
            "household_id": flagged["household_id"].to_numpy(),
            "age": flagged["age"].to_numpy(),
            # SCF hhsex: 1 = male, 2 = female. CPS is_female is a bool.
            "hhsex": np.where(flagged["is_female"].to_numpy(), 2.0, 1.0),
            "married": flagged["married_code"].to_numpy().astype(np.float64),
        }
    )


def _aggregate_household_totals(persons: pd.DataFrame) -> pd.DataFrame:
    """Aggregate person rows to household ``kids``, ``income`` and ``wageinc``."""
    persons = persons.assign(is_kid=(persons["age"] < _ADULT_AGE).astype(np.float64))
    grouped = persons.groupby("household_id", as_index=False).agg(
        kids=("is_kid", "sum"),
        income=("total_income", "sum"),
        wageinc=("employment_income", "sum"),
    )
    return grouped
