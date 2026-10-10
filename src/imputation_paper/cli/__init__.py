"""``imp``: the paper's command line.

Three subcommands mirror the reproduction workflow:

* ``imp demo``    -- run the dependency-free toy pipeline end to end (CI's path).
* ``imp sweep``   -- run registered methods over a task's repeated splits and
  write ``metrics_long.csv`` under ``runs/``.
* ``imp figures`` -- aggregate a run's ``metrics_long.csv`` into the summary
  table artifacts the paper includes.

Subcommand modules are imported lazily inside :func:`main` so ``imp --help``
stays fast and the base install never imports more than it needs.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def _add_scf_wave_arguments(parser: argparse.ArgumentParser, subject: str) -> None:
    """Add the SCF wave and dollar-basis options shared by sweep and harness."""
    parser.add_argument(
        "--scf-year",
        type=int,
        default=2022,
        help=f"SCF wave for {subject}: 2022 (the paper's) or 2025 (the "
        "replication wave).",
    )
    parser.add_argument(
        "--scf-dollar-year",
        type=int,
        default=None,
        help=f"Dollar basis for {subject} (default: the survey year's own "
        "dollars, whichever file the Fed currently serves).",
    )


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``imp`` console script."""
    parser = argparse.ArgumentParser(
        prog="imp",
        description="Reproduction commands for the populace imputation paper.",
    )
    subcommands = parser.add_subparsers(dest="command", required=True)

    demo = subcommands.add_parser(
        "demo",
        help="Run the dependency-free toy pipeline end to end.",
    )
    demo.add_argument("--seed", type=int, default=0, help="Toy-data seed.")
    demo.add_argument("--n", type=int, default=800, help="Toy-data row count.")

    sweep = subcommands.add_parser(
        "sweep",
        help="Run registered methods over a task's repeated splits.",
    )
    sweep.add_argument(
        "--task",
        default="toy",
        help="Task to run (only the built-in 'toy' task is wired so far; "
        "the paper tasks are specified in PLAN.md).",
    )
    sweep.add_argument(
        "--out",
        type=Path,
        default=Path("runs/toy-sweep"),
        help="Run directory to write metrics_long.csv into.",
    )
    sweep.add_argument(
        "--methods",
        nargs="*",
        default=None,
        help="Registry keys to run (default: every registered method; "
        "methods whose adapters or packages are missing are reported as "
        "skipped, never silently dropped).",
    )
    sweep.add_argument(
        "--seeds",
        type=int,
        default=10,
        help="Number of repeated donor/receiver splits.",
    )
    sweep.add_argument(
        "--max-rows",
        type=int,
        default=20_000,
        help="Deterministic row cap applied to the task before splitting.",
    )
    _add_scf_wave_arguments(sweep, "the scf_wealth task")

    harness = subcommands.add_parser(
        "harness",
        help="Run the SCF->CPS population-view experiment.",
    )
    harness.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Run directory (default depends on --profile).",
    )
    harness.add_argument(
        "--profile",
        default="minimal",
        choices=("minimal", "populace-scale"),
        help="minimal: 7 shared predictors, 2 targets, one receiver vintage; "
        "populace-scale: 10 predictors, 4 chained targets, pooled 2023-2025 "
        "receiver.",
    )
    harness.add_argument(
        "--methods",
        nargs="*",
        default=None,
        help="Registry keys to run (default: every registered method).",
    )
    harness.add_argument(
        "--seeds", type=int, default=10, help="Number of SCF donor/holdout splits."
    )
    harness.add_argument(
        "--max-receiver-rows",
        type=int,
        default=20_000,
        help="Deterministic cap on the CPS receiver sample.",
    )
    _add_scf_wave_arguments(harness, "the SCF donor")

    tables = subcommands.add_parser(
        "tables",
        help="Write the manuscript's LaTeX tables from the run directories.",
    )
    tables.add_argument(
        "--runs-dir", type=Path, default=Path("runs"), help="Runs root."
    )
    tables.add_argument(
        "--out",
        type=Path,
        default=Path("paper/tables"),
        help="Directory the manuscript \\input's tables from.",
    )

    replication = subcommands.add_parser(
        "replication",
        help="Score the SCF 2025 replication against the paper's 2022 runs.",
    )
    replication.add_argument(
        "--runs-dir", type=Path, default=Path("runs"), help="Runs root."
    )
    replication.add_argument(
        "--docs-out",
        type=Path,
        default=Path("docs/scf-2025-replication"),
        help="Directory for the comparison document and its CSVs.",
    )
    replication.add_argument(
        "--tables-out",
        type=Path,
        default=Path("paper/tables"),
        help="Directory for the appendix's generated LaTeX inputs.",
    )

    scf_checks = subcommands.add_parser(
        "scf-checks",
        help="Check SCF weights, the balance-sheet identity, and the Fed's "
        "published tables on both waves.",
    )
    scf_checks.add_argument(
        "--out",
        type=Path,
        default=Path("runs/scf-population-checks"),
        help="Run directory to write manifest.json into.",
    )

    hotdeck_reuse = subcommands.add_parser(
        "hotdeck-reuse",
        help="Measure hot-deck donor reuse in the population view, both waves "
        "(needs the methods extra).",
    )
    hotdeck_reuse.add_argument(
        "--out",
        type=Path,
        default=Path("runs/scf-hotdeck-reuse"),
        help="Run directory to write manifest.json into.",
    )
    hotdeck_reuse.add_argument(
        "--runs-dir", type=Path, default=Path("runs"), help="Runs root."
    )

    figures = subcommands.add_parser(
        "figures",
        help="Aggregate a run's long-format artifacts into summary tables.",
    )
    figures.add_argument(
        "run_dir",
        type=Path,
        help="A run directory containing metrics_long.csv and/or "
        "harness_long.csv (from `imp sweep` / `imp harness`).",
    )

    args = parser.parse_args(argv)

    if args.command == "demo":
        from imputation_paper.cli.demo import run_demo

        return run_demo(seed=args.seed, n=args.n)
    if args.command == "sweep":
        from imputation_paper.cli.sweep import run_sweep

        return run_sweep(
            task=args.task,
            out=args.out,
            methods=args.methods,
            n_seeds=args.seeds,
            max_rows=args.max_rows,
            scf_year=args.scf_year,
            scf_dollar_year=args.scf_dollar_year,
        )
    if args.command == "harness":
        from imputation_paper.cli.harness import run_harness

        return run_harness(
            out=args.out,
            methods=args.methods,
            n_seeds=args.seeds,
            max_receiver_rows=args.max_receiver_rows,
            profile=args.profile,
            scf_year=args.scf_year,
            scf_dollar_year=args.scf_dollar_year,
        )
    if args.command == "tables":
        from imputation_paper.cli.tables import make_tables

        return make_tables(runs_dir=args.runs_dir, out_dir=args.out)
    if args.command == "replication":
        from imputation_paper.cli.replication import make_replication

        return make_replication(
            runs_dir=args.runs_dir,
            docs_out=args.docs_out,
            tables_out=args.tables_out,
        )
    if args.command == "scf-checks":
        from imputation_paper.cli.replication import make_population_checks

        return make_population_checks(out=args.out)
    if args.command == "hotdeck-reuse":
        from imputation_paper.cli.replication import make_hotdeck_reuse

        return make_hotdeck_reuse(out=args.out, runs_dir=args.runs_dir)
    if args.command == "figures":
        from imputation_paper.cli.figures import make_figures

        return make_figures(args.run_dir)
    raise AssertionError(f"Unhandled command {args.command!r}.")  # pragma: no cover
