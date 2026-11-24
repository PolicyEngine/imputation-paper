"""``imp tables``: the OpenML mean ranks cover exactly the rows the table prints.

``microimpute_qrf`` and ``plain_qrf`` are one estimator under two registry
keys, and the tables print it once. The mean-rank column must be computed over
the printed rows: ranking the hidden duplicate too would give nine ranks to
eight rows and shift every method that sits below the pair.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from imputation_paper.cli import tables
from imputation_paper.cli.tables import make_tables

#: Wasserstein-1 by method, identical on both toy datasets. The two forest keys
#: share a value, as they do in the committed runs.
W1_BY_METHOD = {
    "populace_fit": 1.0,
    "plain_qrf": 2.0,
    "microimpute_qrf": 2.0,
    "statmatch_hotdeck": 3.0,
    "microimpute_ols": 4.0,
}


def _write_openml_run(runs_dir: Path, dataset: str) -> None:
    run = runs_dir / f"openml-{dataset}"
    run.mkdir(parents=True)
    rows = [
        {
            "method": method,
            "seed": seed,
            "target": "y",
            "metric": "wasserstein1",
            "value": value,
        }
        for method, value in W1_BY_METHOD.items()
        for seed in (0, 1)
    ]
    pd.DataFrame(rows).to_csv(run / "metrics_long.csv", index=False)


def _mean_ranks(table: str) -> dict[str, float]:
    """Row label -> mean rank, read from the generated tabular."""
    ranks: dict[str, float] = {}
    for line in table.splitlines():
        if not line.endswith(r"\\") or line.startswith("Method"):
            continue
        cells = [cell.strip() for cell in line.removesuffix(r"\\").split("&")]
        rank = cells[-1].removeprefix(r"\textbf{").removesuffix("}")
        ranks[cells[0]] = float(rank)
    return ranks


def test_openml_mean_ranks_cover_only_printed_rows(tmp_path: Path) -> None:
    """Four printed rows get ranks 1 to 4; the hidden duplicate takes none."""
    runs_dir = tmp_path / "runs"
    for dataset in ("alpha", "beta"):
        _write_openml_run(runs_dir, dataset)
    out_dir = tmp_path / "tables"

    assert make_tables(runs_dir=runs_dir, out_dir=out_dir) == 0

    ranks = _mean_ranks((out_dir / "openml.tex").read_text())
    assert len(ranks) == len(W1_BY_METHOD) - 1  # the duplicate key prints once
    assert sorted(ranks.values()) == [1.0, 2.0, 3.0, 4.0]
    assert max(ranks.values()) == len(ranks)


# ---------------------------------------------------------------------------
# Arms added after a task's first sweep
# ---------------------------------------------------------------------------


def _write_long(run: Path, values: dict[str, list[float]]) -> None:
    """One ``wasserstein1`` row per (method, seed); seeds follow list order."""
    run.mkdir(parents=True)
    rows = [
        {
            "method": method,
            "seed": seed,
            "target": "y",
            "metric": "wasserstein1",
            "value": value,
        }
        for method, per_seed in values.items()
        for seed, value in enumerate(per_seed)
    ]
    pd.DataFrame(rows).to_csv(run / "metrics_long.csv", index=False)


def test_added_arms_are_read_with_their_run(tmp_path: Path) -> None:
    """New methods join the table; the re-run controls are counted once."""
    runs_dir = tmp_path / "runs"
    _write_long(
        runs_dir / "openml-alpha",
        {"populace_fit": [1.0, 3.0], "statmatch_hotdeck": [4.0, 6.0]},
    )
    _write_long(
        runs_dir / "openml-alpha-added-arms",
        {"populace_fit": [1.0, 3.0], "populace_fit_ungated": [2.0, 2.5]},
    )

    merged = tables._read_long(runs_dir / "openml-alpha", "metrics_long.csv")
    assert merged is not None
    assert merged.groupby("method").size().to_dict() == {
        "populace_fit": 2,
        "populace_fit_ungated": 2,
        "statmatch_hotdeck": 2,
    }

    out_dir = tmp_path / "tables"
    assert make_tables(runs_dir=runs_dir, out_dir=out_dir) == 0
    table = (out_dir / "openml.tex").read_text()
    # One dataset column: the added-arms directory is not a dataset of its own.
    assert "Method & alpha & Mean rank" in table
    assert _mean_ranks(table) == {
        "microcosm-fit": 1.0,
        r"\quad -- ungated": 2.0,
        "NND hot deck (py-statmatch)": 3.0,
    }


def test_added_arms_must_reproduce_the_rows_they_share(tmp_path: Path) -> None:
    """A control that comes out differently means different data or splits."""
    runs_dir = tmp_path / "runs"
    _write_long(runs_dir / "openml-alpha", {"populace_fit": [1.0, 3.0]})
    _write_long(
        runs_dir / "openml-alpha-added-arms",
        {"populace_fit": [1.0, 3.0000001], "populace_fit_ungated": [2.0, 2.5]},
    )
    with pytest.raises(ValueError, match="disagrees.*populace_fit"):
        tables._read_long(runs_dir / "openml-alpha", "metrics_long.csv")


def test_added_arms_must_cover_the_runs_seeds(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    _write_long(runs_dir / "openml-alpha", {"populace_fit": [1.0, 3.0, 2.0]})
    _write_long(
        runs_dir / "openml-alpha-added-arms", {"populace_fit_ungated": [2.0, 2.5]}
    )
    with pytest.raises(ValueError, match="populace_fit_ungated.*seeds"):
        tables._read_long(runs_dir / "openml-alpha", "metrics_long.csv")


def test_added_arms_without_their_run_write_nothing(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    _write_long(
        runs_dir / "openml-alpha-added-arms", {"populace_fit_ungated": [2.0, 2.5]}
    )
    assert tables._read_long(runs_dir / "openml-alpha", "metrics_long.csv") is None
    assert make_tables(runs_dir=runs_dir, out_dir=tmp_path / "tables") == 1


# ---------------------------------------------------------------------------
# Rows that coincide by construction, and the ablations table's columns
# ---------------------------------------------------------------------------


def test_openml_table_prints_an_unchained_arm_once_with_its_chained_twin(
    tmp_path: Path,
) -> None:
    """With one target there is nothing to chain on: the twins share one row."""
    runs_dir = tmp_path / "runs"
    _write_long(
        runs_dir / "openml-alpha",
        {
            "populace_fit": [1.0, 3.0],
            "populace_fit_unchained": [1.0, 3.0],
            "populace_fit_ungated": [4.0, 4.0],
            "populace_fit_ungated_unchained": [4.0, 4.0],
            "statmatch_hotdeck": [5.0, 7.0],
        },
    )
    out_dir = tmp_path / "tables"
    assert make_tables(runs_dir=runs_dir, out_dir=out_dir) == 0
    assert _mean_ranks((out_dir / "openml.tex").read_text()) == {
        "microcosm-fit": 1.0,
        r"\quad -- ungated": 2.0,
        "NND hot deck (py-statmatch)": 3.0,
    }


def test_openml_table_prints_the_random_hot_decks_once_under_equal_weights(
    tmp_path: Path,
) -> None:
    """With uniform weights the weighted draw is the unweighted one: one row."""
    runs_dir = tmp_path / "runs"
    _write_long(
        runs_dir / "openml-alpha",
        {
            "populace_fit": [1.0, 3.0],
            "statmatch_rand_hotdeck_unweighted": [4.0, 6.0],
            "statmatch_rand_hotdeck": [4.0, 6.0],
        },
    )
    out_dir = tmp_path / "tables"
    assert make_tables(runs_dir=runs_dir, out_dir=out_dir) == 0
    assert _mean_ranks((out_dir / "openml.tex").read_text()) == {
        "microcosm-fit": 1.0,
        "Weighted random hot deck": 2.0,
    }


def test_openml_table_keeps_an_unchained_arm_that_differs(tmp_path: Path) -> None:
    """The twin is dropped only when the two rows really are the same."""
    runs_dir = tmp_path / "runs"
    _write_long(
        runs_dir / "openml-alpha",
        {"populace_fit": [1.0, 3.0], "populace_fit_unchained": [1.0, 5.0]},
    )
    out_dir = tmp_path / "tables"
    assert make_tables(runs_dir=runs_dir, out_dir=out_dir) == 0
    assert _mean_ranks((out_dir / "openml.tex").read_text()) == {
        "microcosm-fit": 1.0,
        r"\quad -- unchained": 2.0,
    }


def test_ablations_table_has_one_column_per_knockout_present(tmp_path: Path) -> None:
    """Paired deltas from the candidate, for the knockouts only."""
    runs_dir = tmp_path / "runs"
    _write_long(
        runs_dir / "scf-wealth",
        {
            "populace_fit": [10.0, 20.0, 30.0],
            "populace_fit_unweighted": [12.0, 25.0, 30.0],
            "plain_qrf": [50.0, 50.0, 50.0],
        },
    )
    _write_long(
        runs_dir / "scf-wealth-added-arms",
        {
            "populace_fit": [10.0, 20.0, 30.0],
            "populace_fit_ungated": [11.0, 21.0, 31.0],
        },
    )
    out_dir = tmp_path / "tables"
    assert make_tables(runs_dir=runs_dir, out_dir=out_dir) == 0
    table = (out_dir / "ablations.tex").read_text()
    assert (
        r"Task & Target & Metric & $\Delta$ unweighted & $\Delta$ ungated \\" in table
    )
    assert "microimpute" not in table  # a baseline, not a knockout
    # unweighted: deltas 2, 5, 0 -> mean 2.33, sd 2.52; ungated: 1, 1, 1.
    assert r"scf wealth & y & wasserstein1 & 2.33 (2.52) & 1.00 (0.000) \\" in table
