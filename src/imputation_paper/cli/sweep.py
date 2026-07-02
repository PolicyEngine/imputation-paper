"""``imp sweep``: run registered methods over a task's repeated splits.

The sweep is the paper's evidence generator: every number in the manuscript is
an aggregation of a ``metrics_long.csv`` some sweep wrote under ``runs/``. The
inner cell is :func:`~imputation_paper.experiments.conditions.run_condition`;
this module adds the task registry, the (task, method, seed) iteration, the
skip accounting, and the artifact writing.

Tasks resolve through lazy loaders: the built-in ``toy`` task needs nothing,
the real tasks (SCF wealth, CPS zero-inflated components, the OpenML suite)
import their loaders from :mod:`imputation_paper.data` only when selected, so
the sweep module itself imports on the base install.

Skips are never silent: a method whose package is missing (``methods`` extra
not installed) is recorded in ``skipped.csv`` next to the metrics, and
summarized on stdout. Oversized tasks are capped by a deterministic uniform
row subsample (seed 0, shared by every method and seed) and the cap is
recorded in ``manifest.json`` -- no silent truncation.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from imputation_paper import methods as method_registry
from imputation_paper import smoke
from imputation_paper.experiments.conditions import (
    ConditionResult,
    rows_from_result,
    run_condition,
)
from imputation_paper.experiments.holdout import paired_splits

if TYPE_CHECKING:  # pragma: no cover - typing only
    from imputation_paper.data.base import TaskFrame

#: The OpenML AutoML Benchmark regression datasets from the microimpute
#: manuscript's cross-dataset appendix.
OPENML_NAMES: tuple[str, ...] = (
    "space_ga",
    "elevators",
    "brazilian_houses",
    "onlinenewspopularity",
    "abalone",
    "house_sales",
)


def _load_toy() -> TaskFrame:
    """The built-in dependency-free toy task."""
    from imputation_paper.data.base import TaskFrame

    dataset = smoke.make_toy_dataset(seed=0, n=800)
    pooled = pd.concat([dataset.train, dataset.test], ignore_index=True)
    return TaskFrame(
        name="toy",
        frame=pooled,
        predictors=dataset.predictors,
        targets=dataset.targets,
        weight_column=dataset.weight_column,
    )


def _load_scf() -> TaskFrame:
    from imputation_paper.data.scf import load_scf

    return load_scf()


def _load_cps() -> TaskFrame:
    from imputation_paper.data.cps import load_cps

    return load_cps()


def _load_openml(name: str) -> TaskFrame:
    from imputation_paper.data.openml import load_openml

    return load_openml(name)


#: Task key -> lazy loader. The paper's task set; adding a task is a PLAN.md
#: change, not drift.
TASK_LOADERS: dict[str, Callable[[], TaskFrame]] = {
    "toy": _load_toy,
    "scf_wealth": _load_scf,
    "cps_components": _load_cps,
    **{f"openml_{name}": partial(_load_openml, name) for name in OPENML_NAMES},
}


def _capped(frame: pd.DataFrame, max_rows: int) -> tuple[pd.DataFrame, bool]:
    """Deterministically subsample ``frame`` to ``max_rows`` (uniform, seed 0).

    The cap is shared by every method and seed (it happens before the paired
    splits), so it changes the task's effective sample, never the pairing.
    """
    if len(frame) <= max_rows:
        return frame, False
    capped = frame.sample(n=max_rows, random_state=0).reset_index(drop=True)
    return capped, True


def run_sweep(
    *,
    task: str = "toy",
    out: Path = Path("runs/toy-sweep"),
    methods: list[str] | None = None,
    n_seeds: int = 10,
    max_rows: int = 20_000,
) -> int:
    """Run ``methods`` over ``n_seeds`` paired splits of ``task``; write artifacts.

    Writes into ``out``: ``metrics_long.csv`` (one row per method x seed x
    target x metric), ``skipped.csv`` (one row per skipped method with the
    reason), and ``manifest.json`` (task, rows used, cap, seeds, methods).

    Args:
        task: Task key in :data:`TASK_LOADERS`.
        out: Run directory to create/write.
        methods: Registry keys to run; ``None`` runs every registered method.
        n_seeds: Number of repeated paired splits.
        max_rows: Deterministic row cap applied before splitting.

    Returns:
        Process exit code (``0`` if at least one method produced metrics).
    """
    if task not in TASK_LOADERS:
        raise SystemExit(
            f"Unknown task {task!r}; available: {sorted(TASK_LOADERS)}."
        )
    task_frame = TASK_LOADERS[task]()
    frame, was_capped = _capped(task_frame.frame, max_rows)

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
                    task_frame.predictors,
                    task_frame.targets,
                    weight_column=task_frame.weight_column,
                    seed=split.seed,
                )
                rows.extend(rows_from_result(result))
                print(f"  done {task}/{key}/seed={split.seed}", flush=True)
        except (NotImplementedError, ModuleNotFoundError) as reason:
            # Skip accounting, never a silent drop: method packages arrive
            # with the `methods` extra, and the artifact must say which cells
            # are absent and why.
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
    manifest = {
        "task": task,
        "rows_used": int(len(frame)),
        "rows_available": int(len(task_frame.frame)),
        "row_cap_applied": was_capped,
        "max_rows": max_rows,
        "seeds": list(seeds),
        "methods_requested": requested,
        "predictors": list(task_frame.predictors),
        "targets": list(task_frame.targets),
        "weight_column": task_frame.weight_column,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))

    ran = sorted({str(row["method"]) for row in rows})
    print(f"imp sweep -- task={task!r}, seeds={n_seeds}, rows={len(frame)}")
    print(f"  wrote {metrics_path} ({len(rows)} metric rows; methods ran: {ran})")
    print(f"  wrote {skipped_path} ({len(skipped)} skipped)")
    for entry in skipped:
        print(f"  skipped {entry['method']}: {entry['reason'][:100]}")
    return 0 if rows else 1
