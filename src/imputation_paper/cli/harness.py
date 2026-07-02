"""``imp harness``: the population-view experiment (SCF donor, CPS receiver).

This is the paper's instantiation of the population-view harness: each method
fits wealth conditionals on an SCF donor split, imputes them onto real CPS
households, and the resulting candidate population -- CPS demographics plus
imputed wealth, under CPS household weights -- is scored against the *held-out
SCF* through the SCF view (shared predictors + wealth variables, jointly).
A method that recovers the population's demographics-wealth joint scores well;
one that donates marginals while breaking the joint, or collapses toward modal
households, is exposed by the energy/coverage/C2ST axes.

Two references anchor the scale:

* ``scf_sample_reference`` -- the SCF *donor split itself* scored as a
  candidate against the holdout: the sampling-noise floor no method can beat.
* Method rows are read relative to that floor; the gap between the floor and a
  method is imputation error, the floor itself is survey sampling noise.

The receiver sample is fixed once (deterministic, seed 0) across methods and
seeds, so seed-to-seed variation isolates donor-split noise, and every method
sees identical receivers.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pandas as pd

from imputation_paper import methods as method_registry
from imputation_paper.experiments.holdout import split_frame
from imputation_paper.experiments.views import SurveyView, harness_scorecard

#: Registry key used for the sampling-noise floor rows.
REFERENCE_KEY = "scf_sample_reference"


def run_harness(
    *,
    out: Path = Path("runs/scf-to-cps-harness"),
    methods: list[str] | None = None,
    n_seeds: int = 10,
    max_receiver_rows: int = 20_000,
) -> int:
    """Run the SCF->CPS population-view experiment; write ``harness_long.csv``.

    Args:
        out: Run directory to create/write.
        methods: Registry keys to run; ``None`` runs every registered method.
        n_seeds: Number of repeated SCF donor/holdout splits.
        max_receiver_rows: Deterministic cap on the CPS receiver sample.

    Returns:
        Process exit code (``0`` if at least one method produced rows).
    """
    from imputation_paper.data.cps import load_cps_households
    from imputation_paper.data.scf import load_scf

    scf = load_scf()
    receiver_full = load_cps_households()
    shared = [p for p in scf.predictors if p in receiver_full.columns]
    dropped = [p for p in scf.predictors if p not in receiver_full.columns]
    targets = list(scf.targets)

    if len(receiver_full) > max_receiver_rows:
        receiver = receiver_full.sample(
            n=max_receiver_rows, random_state=0
        ).reset_index(drop=True)
    else:
        receiver = receiver_full.reset_index(drop=True)

    requested = methods if methods is not None else list(method_registry.REGISTRY)
    unknown = [key for key in requested if key not in method_registry.REGISTRY]
    if unknown:
        raise SystemExit(
            f"Unknown method key(s) {unknown}; registered: "
            f"{sorted(method_registry.REGISTRY)}."
        )

    # The SCF view: shared demographics plus the wealth block, jointly, under
    # the survey's weights. This is the only informative view -- the
    # candidate's demographic block IS the real CPS, so a CPS view would score
    # the receiver against itself.
    view = SurveyView(
        name="scf",
        columns=(*shared, *targets),
        weight_column=scf.weight_column,
    )

    rows: list[dict[str, object]] = []
    skipped: list[dict[str, str]] = []
    for seed in range(n_seeds):
        split = split_frame(scf.frame, seed=seed)
        holdouts = {"scf": split.test}

        # Sampling-noise floor: the donor split itself as a candidate.
        for row in harness_scorecard(
            split.train, scf.weight_column, [view], holdouts, seed=seed
        ):
            rows.append({"method": REFERENCE_KEY, "seed": seed, **row})

        for key in requested:
            try:
                fit = method_registry.get_method(key).constructor()
                predict = fit(
                    split.train, shared, targets, scf.weight_column, seed=seed
                )
                imputed = predict(receiver.loc[:, shared])
                candidate = pd.concat(
                    [
                        receiver.loc[:, [*shared, "household_weight"]],
                        imputed.set_axis(receiver.index),
                    ],
                    axis=1,
                )
                for row in harness_scorecard(
                    candidate, "household_weight", [view], holdouts, seed=seed
                ):
                    rows.append({"method": key, "seed": seed, **row})
                print(f"  done harness/{key}/seed={seed}", flush=True)
            except (NotImplementedError, ModuleNotFoundError) as reason:
                if not any(entry["method"] == key for entry in skipped):
                    skipped.append({"method": key, "reason": str(reason)})
                continue

    out.mkdir(parents=True, exist_ok=True)
    long_path = out / "harness_long.csv"
    with long_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["method", "seed", "view", "metric", "value"]
        )
        writer.writeheader()
        writer.writerows(rows)
    skipped_path = out / "skipped.csv"
    with skipped_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["method", "reason"])
        writer.writeheader()
        writer.writerows(skipped)
    manifest = {
        "experiment": "scf_to_cps_population_view",
        "shared_predictors": shared,
        "donor_only_predictors_dropped": dropped,
        "targets": targets,
        "receiver_rows_used": int(len(receiver)),
        "receiver_rows_available": int(len(receiver_full)),
        "donor_rows": int(len(scf.frame)),
        "seeds": list(range(n_seeds)),
        "methods_requested": requested,
        "reference_key": REFERENCE_KEY,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))

    ran = sorted({str(row["method"]) for row in rows})
    print(f"imp harness -- seeds={n_seeds}, receiver rows={len(receiver)}")
    print(f"  shared predictors: {shared} (dropped from donor: {dropped})")
    print(f"  wrote {long_path} ({len(rows)} rows; ran: {ran})")
    print(f"  wrote {skipped_path} ({len(skipped)} skipped)")
    return 0 if rows else 1
