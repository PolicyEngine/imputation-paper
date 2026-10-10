"""SCF loader contract, offline: pinned files, a stated dollar basis, re-basing.

The Fed replaces each wave's summary extract when a new wave comes out,
re-basing its dollar columns. These tests pin what the loader does about that
without touching the network: a small synthetic extract is written into a
temporary cache and registered under its own hash, so the full ``load_scf``
path runs in CI. The checks against the real 2022 and 2025 files are in
``tests/test_data.py`` (local-only).

Invariants stated here:

* a file whose hash is not registered is refused, as is a file for another
  wave;
* the returned dollar basis is the one asked for, whichever basis the cached
  file is on; weights, ids, and categorical columns never move;
* re-basing to the file's own basis returns its values bit for bit;
* re-basing is a round trip, and composes;
* the weighted metrics the within-SCF comparison uses are equivariant under a
  change of dollar basis: pinball loss and Wasserstein-1 scale with the
  dollars, zero-share error does not move. That is what licenses reading a
  re-based 2022 level against a 2025 level.
"""

from __future__ import annotations

import hashlib
import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from imputation_paper.data import scf
from imputation_paper.experiments import metrics

FACTOR_2022_TO_2025 = 4789 / 4376

#: Families in the synthetic extract. One id is past the int16 overflow point
#: (``yy1 * 10`` wraps for ``yy1 >= 6554``), so the implicate filter is tested.
_FAMILY_IDS = (1, 2, 7000)


def _synthetic_extract() -> pd.DataFrame:
    """A tiny table with the summary extract's id layout and used columns."""
    rng = np.random.default_rng(0)
    rows = []
    for family in _FAMILY_IDS:
        for implicate in range(1, 6):
            fin = float(rng.integers(0, 50_000))
            nfin = float(rng.integers(0, 400_000))
            debt = float(rng.integers(0, 300_000))
            rows.append(
                {
                    "y1": family * 10 + implicate,
                    "yy1": family,
                    "wgt": float(rng.integers(1_000, 9_000)),
                    "age": float(rng.integers(20, 90)),
                    "hhsex": float(rng.integers(1, 3)),
                    "edcl": float(rng.integers(1, 5)),
                    "married": float(rng.integers(1, 3)),
                    "kids": float(rng.integers(0, 4)),
                    "income": float(rng.integers(0, 200_000)),
                    "wageinc": float(rng.integers(0, 150_000)),
                    "race": float(rng.integers(1, 6)),
                    "housecl": float(rng.integers(1, 3)),
                    "lf": float(rng.integers(0, 2)),
                    "fin": fin,
                    "nfin": nfin,
                    "debt": debt,
                    "networth": fin + nfin - debt,
                    "asset": fin + nfin,
                }
            )
    frame = pd.DataFrame(rows)
    return frame.astype({"y1": "int32", "yy1": "int16"})


def _write_zip(path: Path, frame: pd.DataFrame, member: str = "rscfp2022.dta") -> str:
    """Write ``frame`` as a one-member Stata zip; return the zip's SHA-256."""
    buffer = io.BytesIO()
    frame.to_stata(buffer, write_index=False)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(member, buffer.getvalue())
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def cached_extract(tmp_path, monkeypatch) -> pd.DataFrame:
    """A synthetic 2022 extract in 2025 dollars, cached and registered."""
    monkeypatch.setenv("IMPUTATION_PAPER_CACHE", str(tmp_path))
    raw = _synthetic_extract()
    digest = _write_zip(tmp_path / "scfp2022s.zip", raw)
    monkeypatch.setitem(
        scf.KNOWN_EXTRACTS,
        digest,
        scf.ExtractVintage(2022, 2025, "synthetic 2022 extract in 2025 dollars"),
    )
    return raw


def test_rebase_factor_is_the_feds_cpi_ratio() -> None:
    assert scf.rebase_factor(2022, 2025) == FACTOR_2022_TO_2025
    assert scf.rebase_factor(2025, 2022) == 4376 / 4789
    assert scf.rebase_factor(2022, 2022) == 1.0
    with pytest.raises(ValueError, match="No CPI-U-RS value"):
        scf.rebase_factor(2019, 2025)


def test_registered_files_cover_both_waves_and_both_2022_bases() -> None:
    """The paper's 2022 file, its 2025-dollar re-issue, and the 2025 file."""
    vintages = {(v.survey_year, v.dollar_year) for v in scf.KNOWN_EXTRACTS.values()}
    assert vintages == {(2022, 2022), (2022, 2025), (2025, 2025)}
    assert all(len(digest) == 64 for digest in scf.KNOWN_EXTRACTS)


def test_unregistered_file_is_refused(tmp_path, monkeypatch) -> None:
    """A file with an unknown hash has an unknown dollar basis: refuse it."""
    monkeypatch.setenv("IMPUTATION_PAPER_CACHE", str(tmp_path))
    _write_zip(tmp_path / "scfp2022s.zip", _synthetic_extract())
    with pytest.raises(ValueError, match="not a registered SCF summary extract"):
        scf.load_scf(2022)


def test_file_for_another_wave_is_refused(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("IMPUTATION_PAPER_CACHE", str(tmp_path))
    digest = _write_zip(tmp_path / "scfp2025s.zip", _synthetic_extract())
    monkeypatch.setitem(
        scf.KNOWN_EXTRACTS, digest, scf.ExtractVintage(2022, 2025, "synthetic")
    )
    with pytest.raises(ValueError, match="not the 2025 wave"):
        scf.load_scf(2025)


def test_default_basis_is_the_survey_years_whatever_is_cached(
    cached_extract: pd.DataFrame,
) -> None:
    """A 2025-dollar 2022 file loads in 2022 dollars unless asked otherwise."""
    task = scf.load_scf(2022, profile="populace-scale")
    first = cached_extract[cached_extract["y1"] % 10 == 1].reset_index(drop=True)
    assert task.provenance["file_dollar_year"] == 2025
    assert task.provenance["dollar_year"] == 2022
    assert task.provenance["rebase_factor"] == 4376 / 4789

    for column in ("income", "wageinc", "fin", "nfin", "debt", "networth"):
        np.testing.assert_allclose(
            task.frame[column], first[column] / FACTOR_2022_TO_2025, rtol=1e-15
        )
    for column in ("wgt", "age", "hhsex", "edcl", "married", "kids", "race", "lf"):
        np.testing.assert_array_equal(task.frame[column], first[column])
    # The identity survives re-basing (to floating-point rounding).
    balance = task.frame["fin"] + task.frame["nfin"] - task.frame["debt"]
    np.testing.assert_allclose(task.frame["networth"], balance, rtol=1e-12, atol=1e-6)


def test_own_basis_returns_the_files_values_bit_for_bit(
    cached_extract: pd.DataFrame,
) -> None:
    task = scf.load_scf(2022, dollar_year=2025)
    first = cached_extract[cached_extract["y1"] % 10 == 1].reset_index(drop=True)
    assert task.provenance["rebase_factor"] == 1.0
    for column in task.used_columns:
        np.testing.assert_array_equal(task.frame[column], first[column])


def test_implicate_filter_survives_the_int16_overflow(
    cached_extract: pd.DataFrame,
) -> None:
    """Every family keeps exactly one first implicate, ids past 6553 included."""
    implicates = scf.load_scf_implicates(2022)
    assert sorted(implicates["implicate"].unique()) == [1, 2, 3, 4, 5]
    assert len(implicates) == 5 * len(_FAMILY_IDS)
    assert len(scf.load_scf(2022).frame) == len(_FAMILY_IDS)
    naive = cached_extract["y1"] - cached_extract["yy1"] * 10  # wraps at int16
    assert not naive.isin(range(1, 6)).all(), "fixture no longer overflows int16"


_dollars = st.floats(min_value=-1e10, max_value=1e10, allow_nan=False, width=64)
_years = st.sampled_from(sorted(scf.CPI_U_RS_SEPTEMBER))


@settings(max_examples=100, deadline=None)
@given(values=st.lists(_dollars, min_size=1, max_size=20), start=_years, other=_years)
def test_rebasing_round_trips_and_leaves_non_dollar_columns(
    values: list[float], start: int, other: int
) -> None:
    frame = pd.DataFrame(
        {
            "networth": values,
            "debt": [abs(v) for v in values],
            "wgt": np.arange(1.0, len(values) + 1),
            "edcl": np.arange(len(values)) % 4 + 1.0,
        }
    )
    there = scf.rebase_dollars(frame, start, other)
    back = scf.rebase_dollars(there, other, start)
    for column in ("networth", "debt"):
        np.testing.assert_allclose(back[column], frame[column], rtol=1e-12, atol=0)
        np.testing.assert_allclose(
            there[column],
            frame[column] * scf.rebase_factor(start, other),
            rtol=1e-15,
            atol=0,
        )
    for column in ("wgt", "edcl"):
        np.testing.assert_array_equal(there[column], frame[column])
    if start == other:
        pd.testing.assert_frame_equal(there, frame, check_exact=True)


#: A dollar amount: exactly zero, or at least a dollar. ``zero_share_error``
#: counts magnitudes up to a millionth of a dollar as zeros, so amounts within
#: a re-basing factor of that threshold are excluded; no balance-sheet value
#: sits there.
_amount = st.one_of(
    st.just(0.0), st.floats(min_value=1.0, max_value=1e8, allow_nan=False)
)


@settings(max_examples=100, deadline=None)
@given(data=st.data())
def test_dollar_metrics_are_equivariant_under_rebasing(data: st.DataObject) -> None:
    """Pinball and W1 scale with the dollars; zero-share error does not move."""
    truth = np.array(data.draw(st.lists(_amount, min_size=5, max_size=40)))
    n = len(truth)
    drawn = np.array(data.draw(st.lists(_amount, min_size=n, max_size=n)))
    weights = np.array(
        data.draw(
            st.lists(
                st.floats(min_value=0.1, max_value=1e4, allow_nan=False),
                min_size=n,
                max_size=n,
            )
        )
    )
    c = FACTOR_2022_TO_2025

    pinball = metrics.weighted_pinball_loss(truth, drawn, weights=weights)
    rebased = metrics.weighted_pinball_loss(c * truth, c * drawn, weights=weights)
    np.testing.assert_allclose(rebased, c * pinball, rtol=1e-9, atol=1e-6)

    w1 = metrics.weighted_wasserstein1(
        drawn, truth, imputed_weights=weights, donor_weights=weights
    )
    rebased = metrics.weighted_wasserstein1(
        c * drawn, c * truth, imputed_weights=weights, donor_weights=weights
    )
    np.testing.assert_allclose(rebased, c * w1, rtol=1e-9, atol=1e-6)

    zero_share = metrics.zero_share_error(
        truth, drawn, true_weights=weights, pred_weights=weights
    )
    rebased = metrics.zero_share_error(
        c * truth, c * drawn, true_weights=weights, pred_weights=weights
    )
    assert rebased == zero_share
