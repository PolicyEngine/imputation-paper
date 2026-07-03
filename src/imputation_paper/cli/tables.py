"""``imp tables``: curate run summaries into the manuscript's LaTeX tables.

Reads the committed run directories under ``runs/`` and writes the tables the
paper ``\\input``s from ``paper/tables/``. Everything here is a pure
aggregation of ``metrics_long.csv`` / ``harness_long.csv`` -- no number enters
the manuscript any other way. Missing runs are reported and their tables
skipped, never silently faked.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

#: Display order and names. The reference floor row leads the harness table.
METHOD_ORDER: tuple[tuple[str, str], ...] = (
    ("scf_sample_reference", "SCF sample (floor)"),
    ("populace_fit", "populace-fit"),
    ("populace_fit_unweighted", r"\quad -- unweighted"),
    ("populace_fit_unchained", r"\quad -- unchained"),
    ("plain_qrf", r"\quad -- ungated/unchained forest"),
    ("microimpute_qrf", "QRF (microimpute)"),
    ("microimpute_ols", "OLS"),
    ("microimpute_quantreg", "Quantile regression"),
    ("statmatch_hotdeck", "NND hot deck (py-statmatch)"),
    ("weighted_marginal", "Weighted marginal draw"),
)
_DISPLAY = dict(METHOD_ORDER)

#: Metrics where lower is better (used for bolding the best value).
#: fragility_kappa5 is deliberately absent: its target is the *truth's* own
#: exposure, not zero -- too low means the method never generates realistic
#: tails (OLS), too high means landmines. Best = closest to truth.
_LOWER_BETTER = {
    "pinball_loss",
    "wasserstein1",
    "zero_share_error",
    "energy_distance",
}


def _fmt(value: float) -> str:
    """Format a metric value for a table cell at sensible precision."""
    if not np.isfinite(value):
        return "--"
    magnitude = abs(value)
    if magnitude >= 1000:
        return f"{value:,.0f}"
    if magnitude >= 1:
        return f"{value:,.2f}"
    return f"{value:.3f}"


def _cell(mean: float, sd: float, *, bold: bool = False) -> str:
    """A ``mean (sd)`` cell, optionally bolding the mean."""
    body = _fmt(mean)
    if bold:
        body = rf"\textbf{{{body}}}"
    return f"{body} ({_fmt(sd)})"


#: Excluded from display: `microimpute_qrf` is the identical estimator and
#: configuration as the `plain_qrf` ablation row (one implementation, two
#: registry roles); showing both would duplicate a row. Results prose notes
#: the collapse.
_DISPLAY_EXCLUDE = {"microimpute_qrf"}


def _ordered_methods(present: set[str]) -> list[str]:
    """Registry-ordered subset of methods present in a summary."""
    present = present - _DISPLAY_EXCLUDE
    ordered = [key for key, _ in METHOD_ORDER if key in present]
    ordered += sorted(present - set(ordered))  # future-proof: never drop rows
    return ordered


def _pivot(summary: pd.DataFrame) -> pd.DataFrame:
    """Index (method) x columns (target, metric) frame of (mean, sd) tuples."""
    frame = summary.copy()
    frame["cell"] = list(zip(frame["mean"], frame["sd"], strict=True))
    return frame.pivot_table(
        index="method",
        columns=["target", "metric"],
        values="cell",
        aggfunc="first",
    )


def _tabular(
    summary: pd.DataFrame,
    columns: list[tuple[str, str, str]],
    caption_note: str,
) -> str:
    """Render a methods-by-metrics tabular from a run summary.

    Args:
        summary: A ``summary.csv``/``harness_summary.csv`` frame
            (method/target/metric/mean/sd).
        columns: ``(target, metric, header)`` column spec, in order.
        caption_note: One-line generated-by comment embedded in the file.
    """
    pivot = _pivot(summary)
    methods = _ordered_methods(set(pivot.index))

    best: dict[tuple[str, str], float] = {}
    for target, metric, _ in columns:
        if (target, metric) not in pivot.columns:
            continue
        series = pivot[(target, metric)].dropna()
        # The floor row is a reference, not a contender; exclude it from "best".
        contenders = series[series.index != "scf_sample_reference"]
        if contenders.empty:
            continue
        means = contenders.map(lambda cell: cell[0])
        if metric == "c2st_auc":
            # Indistinguishability is the target: best is closest to 0.5,
            # not the extremum.
            target_metric_best = means.loc[(means - 0.5).abs().idxmin()]
        elif metric == "fragility_kappa5":
            # The reference exposure is the truth's own fragility: too low
            # means missing tails, too high means landmines.
            truth_key = (target, "fragility_kappa5_truth")
            if truth_key not in pivot.columns:
                continue
            truth_mean = float(
                pivot[truth_key].dropna().map(lambda cell: cell[0]).mean()
            )
            target_metric_best = means.loc[(means - truth_mean).abs().idxmin()]
        elif metric.endswith("_truth"):
            # Constant reference columns are never bolded.
            continue
        elif metric in _LOWER_BETTER:
            target_metric_best = means.min()
        else:
            target_metric_best = means.max()
        best[(target, metric)] = float(target_metric_best)

    header = " & ".join(["Method", *(h for _, _, h in columns)])
    lines = [
        f"% Generated by `imp tables` -- do not edit by hand. {caption_note}",
        rf"\begin{{tabular}}{{l{'r' * len(columns)}}}",
        r"\hline",
        header + r" \\",
        r"\hline",
    ]
    for method in methods:
        cells = []
        for target, metric, _ in columns:
            key = (target, metric)
            if key not in pivot.columns or pd.isna(pivot.loc[method].get(key)):
                cells.append("--")
                continue
            mean, sd = pivot.loc[method, key]
            is_best = (
                key in best
                and method != "scf_sample_reference"
                and np.isclose(mean, best[key])
            )
            cells.append(_cell(mean, sd, bold=is_best))
        lines.append(" & ".join([_DISPLAY.get(method, method), *cells]) + r" \\")
        if method == "scf_sample_reference":
            lines.append(r"\hline")
    lines += [r"\hline", r"\end{tabular}", ""]
    return "\n".join(lines)


def _load_summary(run_dir: Path, filename: str) -> pd.DataFrame | None:
    path = run_dir / filename
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    return None if frame.empty else frame


def _long_to_summary(run_dir: Path, filename: str, unit: str) -> pd.DataFrame | None:
    """Aggregate a long file directly (so tables never require `imp figures`)."""
    path = run_dir / filename
    if not path.exists():
        return None
    long = pd.read_csv(path)
    if long.empty:
        return None
    return (
        long.groupby(["method", unit, "metric"], as_index=False)["value"]
        .agg(mean="mean", sd="std")
        .fillna({"sd": 0.0})
        .rename(columns={unit: "target"})
    )


def _openml_table(runs_dir: Path, out_dir: Path, written: list[str]) -> None:
    """Wasserstein-1 by method across the six OpenML datasets, plus mean rank."""
    frames = []
    for run in sorted(runs_dir.glob("openml-*")):
        summary = _long_to_summary(run, "metrics_long.csv", "target")
        if summary is None:
            continue
        w1 = summary[summary["metric"] == "wasserstein1"].copy()
        w1["dataset"] = run.name.removeprefix("openml-")
        frames.append(w1)
    if not frames:
        print("  openml: no runs found; table skipped")
        return
    table = pd.concat(frames, ignore_index=True)
    pivot = table.pivot_table(index="method", columns="dataset", values="mean")
    ranks = pivot.rank(axis=0)  # lower W1 = better rank
    pivot["mean_rank"] = ranks.mean(axis=1)
    methods = _ordered_methods(set(pivot.index))

    datasets = [c for c in pivot.columns if c != "mean_rank"]
    lines = [
        "% Generated by `imp tables` -- do not edit by hand. "
        "Wasserstein-1 to donor, mean over 10 paired seeds; lower is better.",
        rf"\begin{{tabular}}{{l{'r' * (len(datasets) + 1)}}}",
        r"\hline",
        " & ".join(
            [
                "Method",
                *(d.replace("_", r"\_") for d in datasets),
                "Mean rank",
            ]
        )
        + r" \\",
        r"\hline",
    ]
    best_rank = pivot["mean_rank"].min()
    for method in methods:
        row = pivot.loc[method]
        cells = [_fmt(row[d]) for d in datasets]
        rank = f"{row['mean_rank']:.2f}"
        if np.isclose(row["mean_rank"], best_rank):
            rank = rf"\textbf{{{rank}}}"
        lines.append(" & ".join([_DISPLAY.get(method, method), *cells, rank]) + r" \\")
    lines += [r"\hline", r"\end{tabular}", ""]
    (out_dir / "openml.tex").write_text("\n".join(lines))
    written.append("openml.tex")


def _ablation_table(runs_dir: Path, out_dir: Path, written: list[str]) -> None:
    """Paired-seed deltas: ablation minus candidate, per task and metric.

    Positive delta on a lower-is-better metric means the ablation is worse --
    i.e. the knocked-out design choice was contributing.
    """
    ablations = ("populace_fit_unweighted", "populace_fit_unchained", "plain_qrf")
    metrics = ("pinball_loss", "wasserstein1", "zero_share_error")
    rows = []
    for run_name in ("scf-wealth", "cps-components"):
        long_path = runs_dir / run_name / "metrics_long.csv"
        if not long_path.exists():
            continue
        long = pd.read_csv(long_path)
        base = long[long["method"] == "populace_fit"].set_index(
            ["seed", "target", "metric"]
        )["value"]
        for ablation in ablations:
            abl = long[long["method"] == ablation].set_index(
                ["seed", "target", "metric"]
            )["value"]
            delta = (abl - base).dropna()
            for (target, metric), values in delta.groupby(
                [delta.index.get_level_values(1), delta.index.get_level_values(2)]
            ):
                if metric not in metrics:
                    continue
                rows.append(
                    {
                        "task": run_name,
                        "target": target,
                        "metric": metric,
                        "ablation": ablation,
                        "delta_mean": float(values.mean()),
                        "delta_sd": float(values.std(ddof=1))
                        if len(values) > 1
                        else 0.0,
                    }
                )
    if not rows:
        print("  ablations: no runs found; table skipped")
        return
    table = pd.DataFrame(rows)
    pivot = table.pivot_table(
        index=["task", "target", "metric"],
        columns="ablation",
        values="delta_mean",
    )
    lines = [
        "% Generated by `imp tables` -- do not edit by hand. Ablation minus "
        "candidate, mean paired-seed delta; positive = ablation worse "
        "(lower-is-better metrics).",
        r"\begin{tabular}{lllrrr}",
        r"\hline",
        r"Task & Target & Metric & $\Delta$ unweighted & $\Delta$ unchained "
        r"& $\Delta$ plain forest \\",
        r"\hline",
    ]
    for (task, target, metric), row in pivot.iterrows():
        cells = [
            _fmt(row.get(a, float("nan"))) if pd.notna(row.get(a)) else "--"
            for a in ablations
        ]
        lines.append(
            " & ".join(
                [
                    task.replace("-", " "),
                    target.replace("_", r"\_"),
                    metric.replace("_", r"\_"),
                    *cells,
                ]
            )
            + r" \\"
        )
    lines += [r"\hline", r"\end{tabular}", ""]
    (out_dir / "ablations.tex").write_text("\n".join(lines))
    written.append("ablations.tex")


def make_tables(
    runs_dir: Path = Path("runs"), out_dir: Path = Path("paper/tables")
) -> int:
    """Write the manuscript's tables from the run directories.

    Returns:
        Process exit code (``0`` if at least one table was written).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    scf = _long_to_summary(runs_dir / "scf-wealth", "metrics_long.csv", "target")
    if scf is not None:
        (out_dir / "scf_wealth.tex").write_text(
            _tabular(
                scf,
                [
                    ("debt", "pinball_loss", "Debt pinball"),
                    ("debt", "wasserstein1", "Debt $W_1$"),
                    ("debt", "zero_share_error", "Debt zero-share err."),
                    ("networth", "pinball_loss", "Net worth pinball"),
                    ("networth", "wasserstein1", "Net worth $W_1$"),
                ],
                "SCF wealth task, mean (sd) over 10 paired seeds.",
            )
        )
        written.append("scf_wealth.tex")
    else:
        print("  scf-wealth: no run found; table skipped")

    cps = _long_to_summary(runs_dir / "cps-components", "metrics_long.csv", "target")
    if cps is not None:
        (out_dir / "cps_components.tex").write_text(
            _tabular(
                cps,
                [
                    ("interest_income", "pinball_loss", "Interest pinball"),
                    ("interest_income", "zero_share_error", "Interest zero-share"),
                    ("dividend_income", "pinball_loss", "Dividend pinball"),
                    ("dividend_income", "zero_share_error", "Dividend zero-share"),
                ],
                "CPS zero-inflated components, mean (sd) over 10 paired seeds.",
            )
        )
        written.append("cps_components.tex")
        (out_dir / "fragility.tex").write_text(
            _tabular(
                cps,
                [
                    ("interest_income", "fragility_kappa5", "Interest fragility"),
                    (
                        "interest_income",
                        "fragility_kappa5_truth",
                        "(truth)",
                    ),
                    ("dividend_income", "fragility_kappa5", "Dividend fragility"),
                    (
                        "dividend_income",
                        "fragility_kappa5_truth",
                        "(truth)",
                    ),
                ],
                "Worst-case single-record aggregate share at kappa=5.",
            )
        )
        written.append("fragility.tex")
    else:
        print("  cps-components: no run found; tables skipped")

    for run_name, out_name, note in (
        (
            "scf-to-cps-harness",
            "harness.tex",
            "Minimal profile: 6 shared predictors, 2 targets, ASEC 2025 receiver.",
        ),
        (
            "scf-to-cps-harness-scale",
            "harness_scale.tex",
            "Populace-scale profile: 10 shared predictors, 4 chained targets, "
            "pooled ASEC 2023-2025 receiver.",
        ),
    ):
        harness = _long_to_summary(runs_dir / run_name, "harness_long.csv", "view")
        if harness is None:
            print(f"  {run_name}: no run found; table skipped")
            continue
        (out_dir / out_name).write_text(
            _tabular(
                harness,
                [
                    ("scf", "energy_distance", "Energy distance"),
                    ("scf", "prdc_coverage", "Coverage"),
                    ("scf", "prdc_recall", "Recall"),
                    ("scf", "c2st_auc", "C2ST AUC"),
                ],
                f"SCF->CPS population view vs held-out SCF, mean (sd) over 10 "
                f"donor splits. AUC of 0.5 is indistinguishable. {note}",
            )
        )
        written.append(out_name)

    _openml_table(runs_dir, out_dir, written)
    _ablation_table(runs_dir, out_dir, written)

    for name in written:
        print(f"  wrote {out_dir / name}")
    return 0 if written else 1
