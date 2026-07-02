"""``imp sweep``: run registered methods over a task's repeated splits.

The sweep is the paper's evidence generator: every number in the manuscript is
an aggregation of a ``metrics_long.csv`` some sweep wrote under ``runs/``. The
inner cell is :func:`~imputation_paper.experiments.conditions.run_condition`;
this module adds the (task, method, seed) iteration, the skip accounting, and
the artifact writing.

Only the built-in ``toy`` task is wired so far. The paper tasks (SCF->CPS
wealth, CPS/SIPP/PSID cross-survey, PUF zero-inflated components, the OpenML
cross-dataset suite) are specified in PLAN.md and land with their data loaders
behind a ``data`` extra; the sweep loop itself will not change.

Skips are never silent: a method whose package is missing (``methods`` extra
not installed) or whose adapter is not yet implemented is recorded in
``skipped.csv`` next to the metrics, and summarized on stdout.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

from imputation_paper import methods as method_registry
from imputation_paper import smoke
from imputation_paper.experiments.conditions import (
    ConditionResult,
    rows_from_result,
    run_condition,
)
from imputation_paper.experiments.holdout import paired_splits


def _toy_task() -> tuple[pd.DataFrame, tuple[str, ...], tuple[str, ...], str]:
    """The built-in toy task: pooled table, predictors, targets, weight column."""
    dataset = smoke.make_toy_dataset(seed=0, n=800)
    pooled = pd.concat([dataset.train, dataset.test], ignore_index=True)
    return pooled, dataset.predictors, dataset.targets, dataset.weight_column


def run_sweep(
    *,
    task: str = "toy",
    out: Path = Path("runs/toy-sweep"),
    methods: list[str] | None = None,
    n_seeds: int = 10,
) -> int:
    """Run ``methods`` over ``n_seeds`` paired splits of ``task``; write artifacts.

    Writes ``metrics_long.csv`` (one row per method x seed x target x metric)
    and ``skipped.csv`` (one row per skipped method with the reason) into
    ``out``.

    Args:
        task: Task key; only ``"toy"`` is currently wired (see module
            docstring).
        out: Run directory to create/write.
        methods: Registry keys to run; ``None`` runs every registered method.
        n_seeds: Number of repeated paired splits.

    Returns:
        Process exit code (``0`` if at least one method produced metrics).
    """
    if task != "toy":
        raise SystemExit(
            f"Unknown task {task!r}: only the built-in 'toy' task is wired so "
            "far. The paper tasks (SCF->CPS wealth, CPS/SIPP/PSID "
            "cross-survey, PUF zero-inflated components, OpenML cross-dataset) "
            "are specified in PLAN.md and land with their data loaders."
        )
    frame, predictors, targets, weight_column = _toy_task()

    requested = methods if methods is not None else list(method_registry.REGISTRY)
    unknown = [key for key in requested if key not in method_registry.REGISTRY]
    if unknown:
        raise SystemExit(
            f"Unknown method key(s) {unknown}; registered: "
            f"{sorted(method_registry.REGISTRY)}."
        )

    seeds = tuple(range(n_seeds))
    rows: list[dict[str, object]] = []
    skipped: list[dict[str, str]] = []
    for key in requested:
        try:
            for split in paired_splits(frame, seeds=seeds):
                result: ConditionResult = run_condition(
                    key,
                    split.train,
                    split.test,
                    predictors,
                    targets,
                    weight_column=weight_column,
                    seed=split.seed,
                )
                rows.extend(rows_from_result(result))
        except (NotImplementedError, ModuleNotFoundError) as reason:
            # Skip accounting, never a silent drop: adapters and method
            # packages arrive incrementally, and the artifact must say which
            # cells are absent and why.
            skipped.append({"method": key, "reason": str(reason)})
            continue

    out.mkdir(parents=True, exist_ok=True)
    metrics_path = out / "metrics_long.csv"
    with metrics_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["method", "seed", "target", "metric", "value"]
        )
        writer.writeheader()
        writer.writerows(rows)
    skipped_path = out / "skipped.csv"
    with skipped_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["method", "reason"])
        writer.writeheader()
        writer.writerows(skipped)

    ran = sorted({str(row["method"]) for row in rows})
    print(f"imp sweep -- task={task!r}, seeds={n_seeds}")
    print(f"  wrote {metrics_path} ({len(rows)} metric rows; methods ran: {ran})")
    print(f"  wrote {skipped_path} ({len(skipped)} skipped)")
    for entry in skipped:
        print(f"  skipped {entry['method']}: {entry['reason'][:100]}")
    return 0 if rows else 1
