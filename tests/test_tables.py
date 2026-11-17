"""``imp tables``: the OpenML mean ranks cover exactly the rows the table prints.

``microimpute_qrf`` and ``plain_qrf`` are one estimator under two registry
keys, and the tables print it once. The mean-rank column must be computed over
the printed rows: ranking the hidden duplicate too would give nine ranks to
eight rows and shift every method that sits below the pair.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

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
