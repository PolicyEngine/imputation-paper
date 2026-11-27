"""``imp seed-stability``: how far the estimator's own seed moves an imputed file.

In every other experiment of the paper a seed changes three things together:
which rows are donors, which rows are receivers, and the estimator's random
draws. A production build is a different case. Its donors are fixed, its
receivers are many, and only the estimator's seed differs between two runs.

This command reproduces that case on the population view's minimal profile.
It holds an SCF donor split fixed, imputes onto the same 20,000 CPS
households each time, and changes only the estimator's seed. For each donor
split and each arm it records two kinds of repeat:

* ``refit``: one fit and one draw per estimator seed;
* ``redraw``: further draws from the first seed's fit, which shows how much
  of the spread a draw alone accounts for.

Each repeat is scored by the imputed total and the 90th and 99th percentiles
of every target over the receivers, as multiples of the same statistics of
the donor split.

The arms are the estimator as shipped (``populace_fit``) and the diagnostic
variant that samples by weight inside each tree
(``populace_fit_per_tree``, see
:mod:`imputation_paper.experiments.tree_sampling`).
"""

from __future__ import annotations

import contextlib
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

from imputation_paper.cli.harness import PROFILE_RECEIVER_YEARS
from imputation_paper.experiments.conditional_fit import ratios_to_sample
from imputation_paper.experiments.holdout import split_frame
from imputation_paper.experiments.metrics import _weighted_quantile
from imputation_paper.experiments.tree_sampling import (
    chance_in_one_off_resample,
    per_tree_weighted_sampling,
)

#: The arms, by registry-style key: the context each one fits in.
ARMS = {
    "populace_fit": contextlib.nullcontext,
    "populace_fit_per_tree": per_tree_weighted_sampling,
}

#: Profile of the population view the experiment runs on.
PROFILE = "minimal"

#: Share of the split's median beyond which a repeat's value counts as far
#: from typical in :func:`summarize`.
FAR_FROM_MEDIAN = 0.25

FIELDS = ["scf_year", "split", "arm", "kind", "repeat", "target", "metric", "value"]


def default_run_dir(scf_year: int = 2022) -> Path:
    """The run directory a donor wave defaults to."""
    suffix = "" if scf_year == 2022 else f"-{scf_year}"
    return Path(f"runs/seed-stability{suffix}")


def summarize(long: pd.DataFrame) -> pd.DataFrame:
    """Per arm, kind, target and metric: spread across repeats within a split.

    Returns one row per cell with the standard deviation pooled over the
    donor splits (root mean of the within-split variances), the number of
    repeats, and the number of repeats whose value lies more than
    :data:`FAR_FROM_MEDIAN` of the split's median away from that median.
    """
    rows = []
    keys = ["arm", "kind", "target", "metric"]
    for cell, group in long.groupby(keys, sort=True):
        variances, far, total = [], 0, 0
        for _, split in group.groupby("split"):
            values = split["value"].to_numpy(dtype=np.float64)
            median = float(np.median(values))
            variances.append(float(np.var(values, ddof=1)))
            far += int((np.abs(values - median) > FAR_FROM_MEDIAN * abs(median)).sum())
            total += len(values)
        rows.append(
            {
                **dict(zip(keys, cell, strict=True)),
                "pooled_sd": float(np.sqrt(np.mean(variances))),
                "repeats": total,
                "far_from_median": far,
            }
        )
    return pd.DataFrame(rows)


def run_seed_stability(
    *,
    scf_year: int = 2022,
    out: Path | None = None,
    splits: tuple[int, ...] = (0, 1, 2),
    n_seeds: int = 20,
    max_receiver_rows: int = 20_000,
    arms: tuple[str, ...] = tuple(ARMS),
) -> int:
    """Hold each donor split fixed and vary only the estimator's seed.

    Args:
        scf_year: SCF donor wave, 2022 or 2025.
        out: Run directory; ``None`` uses :func:`default_run_dir`.
        splits: Seeds of the donor splits to hold fixed, one at a time.
        n_seeds: Estimator seeds per split and arm, and redraws from the
            first seed's fit.
        max_receiver_rows: Size of the fixed receiver sample.
        arms: Keys of :data:`ARMS` to run.

    Returns:
        Process exit code.
    """
    from populace.fit import RegimeGatedQRF

    from imputation_paper.data.cps import load_cps_households
    from imputation_paper.data.scf import load_scf

    unknown = [arm for arm in arms if arm not in ARMS]
    if unknown:
        raise SystemExit(f"Unknown arm(s) {unknown}; available: {sorted(ARMS)}.")
    out = default_run_dir(scf_year) if out is None else out

    scf = load_scf(scf_year, profile=PROFILE)
    receiver_full = load_cps_households(
        PROFILE_RECEIVER_YEARS[PROFILE], profile=PROFILE
    )
    shared = [p for p in scf.predictors if p in receiver_full.columns]
    targets = list(scf.targets)
    # The population view's own receiver sample: fixed once, seed 0.
    receiver = receiver_full.sample(n=max_receiver_rows, random_state=0).reset_index(
        drop=True
    )
    features = receiver.loc[:, shared]
    receiver_weights = receiver["household_weight"].to_numpy(dtype=np.float64)

    rows: list[dict[str, object]] = []
    resample: dict[str, dict[str, float]] = {}
    for split_seed in splits:
        split = split_frame(scf.frame, seed=split_seed)
        table = split.train.loc[:, [*shared, *targets]]
        donor_weights = split.train[scf.weight_column].to_numpy(dtype=np.float64)
        # What the one-off resample is expected to keep of this donor split:
        # of all rows, and of the rows above the weighted 99th percentile of
        # the last target (net worth), which the SCF oversamples.
        kept = chance_in_one_off_resample(donor_weights)
        last = split.train[targets[-1]].to_numpy(dtype=np.float64)
        top = last > _weighted_quantile(last, donor_weights, np.array([0.99]))[0]
        resample[str(split_seed)] = {
            "donor_rows": int(len(donor_weights)),
            "expected_share_of_rows_kept": float(kept.mean()),
            "rows_above_weighted_q99": int(top.sum()),
            "expected_share_kept_above_weighted_q99": float(kept[top].mean()),
        }

        def score(
            drawn: pd.DataFrame,
            kind: str,
            arm: str,
            repeat: int,
            split_seed: int = split_seed,
            split: object = split,
            donor_weights: np.ndarray = donor_weights,
        ) -> None:
            for target in targets:
                ratios = ratios_to_sample(
                    drawn[target].to_numpy(dtype=np.float64),
                    receiver_weights,
                    split.train[target].to_numpy(dtype=np.float64),
                    donor_weights,
                )
                for metric, value in ratios.items():
                    rows.append(
                        {
                            "scf_year": scf_year,
                            "split": split_seed,
                            "arm": arm,
                            "kind": kind,
                            "repeat": repeat,
                            "target": target,
                            "metric": metric,
                            "value": value,
                        }
                    )

        for arm in arms:
            for seed in range(n_seeds):
                with ARMS[arm]():
                    fitted = RegimeGatedQRF(seed=seed).fit(
                        table, shared, targets, weights=donor_weights
                    )
                score(fitted.predict(features), "refit", arm, seed)
                if seed == 0:
                    for repeat in range(1, n_seeds + 1):
                        score(fitted.predict(features), "redraw", arm, repeat)
            print(
                f"  done seed-stability/{scf_year}/split={split_seed}/{arm}",
                flush=True,
            )

    out.mkdir(parents=True, exist_ok=True)
    long_path = out / "stability_long.csv"
    with long_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    summary = summarize(pd.DataFrame(rows))
    summary_path = out / "stability_summary.csv"
    summary.to_csv(summary_path, index=False)
    manifest = {
        "scf_year": scf_year,
        "profile": PROFILE,
        "receiver_years": list(PROFILE_RECEIVER_YEARS[PROFILE]),
        "receiver_rows": int(len(receiver)),
        "receiver_sample_seed": 0,
        "shared_predictors": shared,
        "targets": targets,
        "donor_split_seeds": list(splits),
        "one_off_resample": {
            "target_of_the_upper_tail": targets[-1],
            "equal_weight_share_of_rows_kept": 1.0 - float(np.exp(-1.0)),
            "by_donor_split": resample,
        },
        "estimator_seeds": list(range(n_seeds)),
        "arms": list(arms),
        "kinds": {
            "refit": "one fit and one draw per estimator seed",
            "redraw": "further draws from the first seed's fit",
        },
        "ratios": "imputed weighted mean and upper quantiles over the receivers, "
        "as multiples of the donor split's",
        "far_from_median": FAR_FROM_MEDIAN,
    }
    if scf.provenance:
        manifest["input_provenance"] = dict(scf.provenance)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"imp seed-stability -- scf_year={scf_year}, splits={list(splits)}")
    print(f"  wrote {long_path} ({len(rows)} rows)")
    print(f"  wrote {summary_path} ({len(summary)} rows)")
    return 0 if rows else 1
