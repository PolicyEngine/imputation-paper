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

    harness = subcommands.add_parser(
        "harness",
        help="Run the SCF->CPS population-view experiment.",
    )
    harness.add_argument(
        "--out",
        type=Path,
        default=Path("runs/scf-to-cps-harness"),
        help="Run directory to write harness_long.csv into.",
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
        )
    if args.command == "harness":
        from imputation_paper.cli.harness import run_harness

        return run_harness(
            out=args.out,
            methods=args.methods,
            n_seeds=args.seeds,
            max_receiver_rows=args.max_receiver_rows,
        )
    if args.command == "figures":
        from imputation_paper.cli.figures import make_figures

        return make_figures(args.run_dir)
    raise AssertionError(f"Unhandled command {args.command!r}.")  # pragma: no cover
