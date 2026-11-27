"""``imp conditional-fit``: score conditional fit within a survey.

The sweep's within-survey metrics look at one imputed column at a time. This
command reruns the sweep's cells (same task loaders, same row cap, same paired
splits, same method seeds, so the same draws) and scores each imputed file
with the paired classifier test of
:mod:`imputation_paper.experiments.conditional_fit`: can a classifier tell the
receivers' true target values from the imputed ones, given the predictors?

It writes its own run directory and leaves the sweep's committed runs alone.
``metrics_long.csv`` has the sweep's schema. The target column is ``joint``
for the test over all targets together, and a target's name for the test over
that target alone (written when the task has more than one target). Each
cell's Wasserstein-1 distances are carried over from the sweep's own scoring,
so a reader can confirm that the draws scored here are the ones behind the
paper's tables.

The classifier counts households, so it barely moves when a method misplaces
a few very large values. On the survey tasks, whose targets are dollar
amounts, each target therefore also gets three measures in dollars: the
Wasserstein-1 distance between imputed and true values within groups of
similar receivers (:data:`GROUPINGS`), and the imputed total and upper
quantiles as multiples of the full sample's.
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np

from imputation_paper import methods as method_registry
from imputation_paper.cli.sweep import TASK_LOADERS, _capped, _load_scf
from imputation_paper.experiments.conditional_fit import (
    ReceiverGroups,
    paired_two_sample_auc,
    ratios_to_sample,
    swap_mask,
    within_group_wasserstein1,
)
from imputation_paper.experiments.conditions import run_condition
from imputation_paper.experiments.holdout import paired_splits

#: Folds over receiver rows in the paired classifier test.
N_FOLDS = 5

#: Target label of the test over all of a task's targets together.
JOINT = "joint"

#: How each survey task's receivers are grouped for the within-group distance:
#: the income predictor, the weighted quantile levels of its cut points, and
#: whether rows with no income form a band of their own. Every grouping also
#: splits by three age bands (under 40, 40 to 64, 65 and over). The SCF task
#: has 15 groups (income fifths by age band) and the CPS task 12 (no earnings
#: and thirds of positive earnings, by age band). The OpenML tasks have no
#: income or age column and are scored without the dollar measures.
GROUPINGS: dict[str, tuple[str, tuple[float, ...], bool]] = {
    "scf_wealth": ("income", (0.2, 0.4, 0.6, 0.8), False),
    "cps_components": ("employment_income", (1 / 3, 2 / 3), True),
}

#: Upper quantiles reported as multiples of the full sample's.
QUANTILE_LEVELS = (0.90, 0.99)


def run_conditional_fit(
    *,
    task: str = "toy",
    out: Path = Path("runs/conditional-fit-toy"),
    methods: list[str] | None = None,
    n_seeds: int = 10,
    max_rows: int = 20_000,
    scf_year: int = 2022,
) -> int:
    """Run ``methods`` over ``n_seeds`` paired splits and score conditional fit.

    Args:
        task: Task key in :data:`imputation_paper.cli.sweep.TASK_LOADERS`.
        out: Run directory to create/write.
        methods: Registry keys to run; ``None`` runs every registered method.
        n_seeds: Number of repeated paired splits.
        max_rows: Deterministic row cap applied before splitting.
        scf_year: SCF wave for the ``scf_wealth`` task.

    Returns:
        Process exit code (``0`` if at least one method produced metrics).
    """
    if task not in TASK_LOADERS:
        raise SystemExit(f"Unknown task {task!r}; available: {sorted(TASK_LOADERS)}.")
    if task == "scf_wealth":
        task_frame = _load_scf(scf_year, None)
    elif scf_year != 2022:
        raise SystemExit(f"--scf-year applies only to scf_wealth, not {task!r}.")
    else:
        task_frame = TASK_LOADERS[task]()
    frame, was_capped = _capped(task_frame.frame, max_rows)

    requested = methods if methods is not None else list(method_registry.REGISTRY)
    unknown = [key for key in requested if key not in method_registry.REGISTRY]
    if unknown:
        raise SystemExit(
            f"Unknown method key(s) {unknown}; registered: "
            f"{sorted(method_registry.REGISTRY)}."
        )

    predictors = list(task_frame.predictors)
    targets = list(task_frame.targets)
    seeds = tuple(range(n_seeds))
    groups: ReceiverGroups | None = None
    if task in GROUPINGS:
        income_column, levels, zero_band = GROUPINGS[task]
        groups = ReceiverGroups.from_sample(
            frame,
            income_column=income_column,
            levels=levels,
            weight_column=task_frame.weight_column,
            zero_band=zero_band,
        )
    sample_weights = frame[task_frame.weight_column].to_numpy(dtype=np.float64)
    rows: list[dict[str, object]] = []
    skipped: list[dict[str, str]] = []
    for key in requested:
        try:
            for split in paired_splits(frame, seeds=seeds):
                started = time.monotonic()
                result = run_condition(
                    key,
                    split.train,
                    split.test,
                    predictors,
                    targets,
                    weight_column=task_frame.weight_column,
                    seed=split.seed,
                )
                fitted = time.monotonic()
                x = split.test[predictors].to_numpy(dtype=np.float64)
                truth = split.test[targets].to_numpy(dtype=np.float64)
                imputed = result.imputed[targets].to_numpy(dtype=np.float64)
                weights = split.test[task_frame.weight_column].to_numpy(
                    dtype=np.float64
                )

                def record(target: str, metric: str, value: float) -> None:
                    rows.append(
                        {
                            "method": key,  # noqa: B023 - consumed in this iteration
                            "seed": split.seed,  # noqa: B023
                            "target": target,
                            "metric": metric,
                            "value": value,
                        }
                    )

                record(
                    JOINT,
                    "conditional_auc",
                    paired_two_sample_auc(
                        x,
                        truth,
                        imputed,
                        weights=weights,
                        n_splits=N_FOLDS,
                        seed=split.seed,
                    ),
                )
                record(
                    JOINT,
                    "conditional_auc_null",
                    paired_two_sample_auc(
                        x,
                        truth,
                        imputed,
                        weights=weights,
                        swap=swap_mask(len(x), seed=split.seed),
                        n_splits=N_FOLDS,
                        seed=split.seed,
                    ),
                )
                for position, target in enumerate(targets):
                    if len(targets) > 1:
                        record(
                            target,
                            "conditional_auc",
                            paired_two_sample_auc(
                                x,
                                truth[:, position],
                                imputed[:, position],
                                weights=weights,
                                n_splits=N_FOLDS,
                                seed=split.seed,
                            ),
                        )
                    record(
                        target,
                        "wasserstein1",
                        result.metrics[f"{target}.wasserstein1"],
                    )
                    if groups is None:
                        continue
                    record(
                        target,
                        "within_group_wasserstein1",
                        within_group_wasserstein1(
                            imputed[:, position],
                            truth[:, position],
                            weights,
                            groups.labels(split.test),
                        ),
                    )
                    ratios = ratios_to_sample(
                        imputed[:, position],
                        weights,
                        frame[target].to_numpy(dtype=np.float64),
                        sample_weights,
                        levels=QUANTILE_LEVELS,
                    )
                    for name, value in ratios.items():
                        record(target, name, value)
                print(
                    f"  done {task}/{key}/seed={split.seed} "
                    f"(fit {fitted - started:.1f}s, "
                    f"test {time.monotonic() - fitted:.1f}s)",
                    flush=True,
                )
        except (NotImplementedError, ModuleNotFoundError) as reason:
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
    manifest: dict[str, object] = {
        "task": task,
        "rows_used": int(len(frame)),
        "rows_available": int(len(task_frame.frame)),
        "row_cap_applied": was_capped,
        "max_rows": max_rows,
        "seeds": list(seeds),
        "methods_requested": requested,
        "predictors": predictors,
        "targets": targets,
        "weight_column": task_frame.weight_column,
        "test": {
            "statistic": "weighted out-of-fold AUC of a classifier telling the "
            "receivers' true target values from the imputed ones, on "
            "predictors and targets together",
            "classifier": "sklearn HistGradientBoostingClassifier at its defaults",
            "folds": N_FOLDS,
            "folds_over": "receiver rows (both copies of a row share a fold)",
            "null": "the same test with the labels of a random half of the "
            "receiver rows swapped",
        },
    }
    if groups is not None:
        manifest["dollar_measures"] = {
            "within_group_wasserstein1": "weighted Wasserstein-1 distance "
            "between imputed and true values inside each group of receivers, "
            "averaged by group weight",
            "groups": {
                "income_column": groups.income_column,
                "income_cuts": list(groups.income_cuts),
                "zero_band": groups.zero_band,
                "age_column": groups.age_column,
                "age_cuts": list(groups.age_cuts),
                "cut_on": "the task's full sample, weighted",
            },
            "ratios": "imputed weighted mean and upper quantiles over the "
            "receivers, as multiples of the full sample's",
            "quantile_levels": list(QUANTILE_LEVELS),
        }
    if task == "scf_wealth":
        manifest["scf_year"] = scf_year
    if task_frame.provenance:
        manifest["input_provenance"] = dict(task_frame.provenance)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))

    ran = sorted({str(row["method"]) for row in rows})
    print(f"imp conditional-fit -- task={task!r}, seeds={n_seeds}, rows={len(frame)}")
    print(f"  wrote {metrics_path} ({len(rows)} metric rows; methods ran: {ran})")
    print(f"  wrote {skipped_path} ({len(skipped)} skipped)")
    return 0 if rows else 1
