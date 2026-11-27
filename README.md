# imputation-paper

Paper and experiment workspace for Microcosm's weight-aware imputation
approach: the regime-gated, sequentially-chained, weighted-bootstrap
quantile-regression forest in
[`microcosm-fit`](https://github.com/PolicyEngine/microcosm/tree/main/packages/microcosm-fit),
benchmarked against standard survey-imputation methods (plain QRF, OLS,
quantile regression, hot-deck statistical matching) with ablations attributing
the gains to weighting and chaining. (Microcosm was first released under the name
Populace; the reported sweeps pin a pre-rename commit, so the experiment code
and its pinned dependencies keep the legacy `populace` spellings.)

Sibling of [l0-paper](https://github.com/PolicyEngine/l0-paper) (which covers
record selection and weighting; this paper covers filling records in) and the
successor to the unpublished microimpute manuscript, whose SCF→CPS wealth
task, SSI policy exhibit, and cross-dataset appendix it absorbs.

## Reproduction

```bash
uv sync                      # base install: harness + metrics only
uv run imp demo              # dependency-free end-to-end toy run (CI's path)
uv run imp sweep --task toy  # writes runs/<name>/metrics_long.csv + skipped.csv
uv run imp figures runs/toy-sweep   # aggregates to summary.csv / summary.tex
quarto render paper/index.qmd       # builds the manuscript
```

The manuscript follows the International Journal of Microsimulation's
submission format: APA references (`apacite`), float labels above and notes
below, float text in a 10 pt sans serif, and wide floats on landscape pages
(`pdflscape`, `afterpage`, `longtable`, `caption`). `paper/ijm.sty` and
`paper/ijmarticle.cls` hold the layout. A TeX distribution without those
packages needs them installed first; Quarto's TinyTeX fetches missing
packages on the first build.

### SCF 2025 replication

The paper's SCF experiments run on the 2022 wave. The 2025 wave (released
2026-10-09) is a replication on data no method was tuned on; the rules were
committed first, in `docs/scf-2025-replication-criteria.md`.

```bash
uv sync --extra methods
export OMP_NUM_THREADS=1     # one OpenMP thread per run; see note below
uv run imp sweep --task scf_wealth --scf-year 2025 --out runs/scf-wealth-2025
uv run imp harness --profile minimal --scf-year 2025         # runs/scf-to-cps-harness-2025
uv run imp harness --profile populace-scale --scf-year 2025  # runs/scf-to-cps-harness-scale-2025
uv run imp scf-checks        # weights, balance-sheet identity, Fed Table 2, both waves
uv run imp hotdeck-reuse     # aggregate donor reuse behind hot deck's 2025 tail
uv run imp pinball-weighting # pinball loss as coded vs with weighted imputed quantiles
uv run imp replication       # docs/scf-2025-replication/comparison.md + appendix inputs
uv run imp tables            # also writes the *_2025.tex tables
```

Diagnostic arms: `--scf-year 2022` into `*-2022-rerun` (does today's
environment reproduce the committed runs?) and `--scf-year 2022
--scf-dollar-year 2025` into `*-2022-in-2025-dollars` (the change of dollar
basis alone). The Fed replaced `scfp2022s.zip` on 2026-10-09 with a copy in
2025 dollars. The loader identifies each extract by SHA-256 and returns the
survey year's dollars by default. From the re-issued copy that recovers the
paper's 2022 inputs only to rounding error, and the forest-based methods
are sensitive to it, so reproducing the committed 2022 runs exactly needs
the original file (SHA-256 `3bb4d890…c9d80`), which the Fed no longer
serves. `runs/scf-wealth-2022-from-reissue` is the within-SCF sweep run from
the re-issued copy (`IMPUTATION_PAPER_CACHE` pointed at a directory holding
only that file), and `imp replication` reports how far it sits from the
committed runs. With the default OpenMP thread count (one per
core), several runs side by side oversubscribe the CPU; on a loaded machine
cells ran many times slower. With one thread per run the 2022
rerun reproduces the committed runs (see the comparison document's
diagnostics).

The real method surface installs with the `methods` extra
(`uv sync --extra methods`), which pulls populace-fit/populace-frame (git,
pinned to a pre-rename commit of the microcosm repo), microimpute (PyPI), and
py-statmatch (git). CI deliberately runs without it:
the registry imports methods lazily, and sweeps record unavailable methods in
`skipped.csv` rather than failing or silently dropping them.

### Conditional fit within each survey

The paired classifier test and the dollar measures of the paper's section on
conditional fit have their own runs. `imp conditional-fit` refits every method
on the sweep's splits with the sweep's seeds, so it scores the same draws, and
it writes its own run directory:

```bash
export OMP_NUM_THREADS=1
uv run imp conditional-fit --task scf_wealth --seeds 30 --out runs/conditional-fit-scf-wealth
uv run imp conditional-fit --task scf_wealth --scf-year 2025 --seeds 30 --out runs/conditional-fit-scf-wealth-2025
uv run imp conditional-fit --task cps_components --seeds 30 --out runs/conditional-fit-cps-components
for name in space_ga elevators brazilian_houses onlinenewspopularity abalone house_sales; do
  uv run imp conditional-fit --task "openml_$name" --out "runs/conditional-fit-openml-$name"
done
uv run imp tables
```

The survey tasks use 30 seeds; the first ten are the sweep's. A test checks
that each run's Wasserstein-1 distances equal the sweep's, cell for cell, on
the seeds they share.

One diagnostic variant sits beside those runs: the constrained hot deck with
two uses allowed per donor, which is kept out of the method registry. Its
runs rerun three methods as controls and record how many receivers share a
value with another receiver:

```bash
ARMS=(populace_fit statmatch_hotdeck statmatch_constrained_hotdeck statmatch_constrained_hotdeck_two_uses)
uv run imp conditional-fit --task scf_wealth --seeds 30 --value-reuse --methods "${ARMS[@]}" --out runs/conditional-fit-scf-wealth-added-arms
uv run imp conditional-fit --task scf_wealth --scf-year 2025 --seeds 30 --value-reuse --methods "${ARMS[@]}" --out runs/conditional-fit-scf-wealth-2025-added-arms
uv run imp conditional-fit --task cps_components --seeds 30 --value-reuse --constrained-max-cells 200000000 --methods "${ARMS[@]}" --out runs/conditional-fit-cps-components-added-arms
uv run imp tables
```

### What the one-off weighted resample costs

One diagnostic variant of the estimator samples by weight inside each tree,
in place of one weighted resample per forest. It is kept out of the method
registry, so no sweep or table includes it unless it is named. Its runs sit
beside the conditional-fit runs, with the estimator rerun as a control, and a
second command holds the donors fixed and changes only the estimator's seed:

```bash
export OMP_NUM_THREADS=1
ARMS=(populace_fit populace_fit_per_tree)
uv run imp conditional-fit --task scf_wealth --seeds 30 --methods "${ARMS[@]}" --out runs/conditional-fit-scf-wealth-per-tree
uv run imp conditional-fit --task scf_wealth --scf-year 2025 --seeds 30 --methods "${ARMS[@]}" --out runs/conditional-fit-scf-wealth-2025-per-tree
uv run imp conditional-fit --task cps_components --seeds 30 --methods "${ARMS[@]}" --out runs/conditional-fit-cps-components-per-tree
uv run imp seed-stability --scf-year 2022   # runs/seed-stability
uv run imp seed-stability --scf-year 2025   # runs/seed-stability-2025
uv run imp tables
```

`imp tables` writes the variant's table only if the control's rows are
identical in the two run directories.

### Arms added after the first sweeps

Seven methods joined the surface after the first sweeps were committed: the
gates-off ablations, the net-worth-derived variants, and three hot decks. Their
runs go into `runs/<name>-added-arms/`, beside the committed run and never
into it:

```bash
export OMP_NUM_THREADS=1
ARMS=(populace_fit weighted_marginal statmatch_hotdeck populace_fit_ungated populace_fit_ungated_unchained populace_fit_derived populace_fit_unchained_derived statmatch_rand_hotdeck statmatch_constrained_hotdeck statmatch_rand_hotdeck_unweighted)
uv run imp sweep --task scf_wealth --methods "${ARMS[@]}" --out runs/scf-wealth-added-arms
uv run imp harness --profile populace-scale --methods "${ARMS[@]}" --out runs/scf-to-cps-harness-scale-added-arms
uv run imp tables
```

The first three keys are controls: methods the first sweep also ran. `imp
tables` reads the two directories together and refuses to write a table
unless the controls' rows are identical in both, so the added arms appear
only when they saw the same data, splits, and environment. The same holds for
the other tasks and for the 2025 wave (`--scf-year 2025`, into
`runs/<name>-2025-added-arms`). An arm that does not apply to a task is
recorded in that run's `skipped.csv`: the net-worth-derived variants outside
the production-scale profile, and the constrained hot deck on the population
view, where its assignment problem is too large.

## Installing the estimator on its own

The estimator is one package in the
[Microcosm repository](https://github.com/PolicyEngine/microcosm) and installs
without the rest of the stack. It is not on PyPI, and the PyPI project named
`microcosm` is unrelated software, so install it as a Git dependency. With uv
one requirement is enough: uv resolves the sibling packages it needs from the
same commit.

The version the paper's runs used, when the package was named `populace-fit`:

```bash
uv pip install "populace-fit @ git+https://github.com/PolicyEngine/microcosm.git@b538d8a4ecac9f0bad33c4f8d27acc453e280624#subdirectory=packages/populace-fit"
```

The current package, `microcosm-fit`, at a commit of your choice:

```bash
uv pip install "microcosm-fit @ git+https://github.com/PolicyEngine/microcosm.git@<commit>#subdirectory=packages/microcosm-fit"
```

pip does not read uv's workspace sources, so with pip pass one such URL for
each package: `populace-fit` and `populace-frame` at the pinned commit, or
`microcosm-fit`, `microcosm-frame` and `microcosm-graph` at a current one.

## Layout

- `PLAN.md` — the experiment plan: tasks × methods × ablations × metrics, the
  protocol, the honesty rules, and sequencing. Read this first.
- `src/imputation_paper/methods.py` — the method-surface registry (candidate,
  ablations, baselines).
- `src/imputation_paper/experiments/` — weighted metrics, paired
  donor/receiver splits, the condition runner.
- `src/imputation_paper/cli/` — `imp demo` / `imp sweep` / `imp figures`.
- `paper/` — Quarto + LaTeX manuscript (IJM style), sections under
  `paper/sections/`, generated tables under `paper/tables/`.
- `runs/` — sweep artifacts (`metrics_long.csv`, `skipped.csv`, summaries);
  large data artifacts stay out of git.

## Boundary

- microcosm packages must not import from this repository.
- Sweeps take explicit, pinned inputs; they do not discover artifacts from
  working directories.
- Every number in the manuscript regenerates from a committed run config;
  results tables are generated (`imp figures`), never hand-edited.

## License

Code in this repository is released under the [MIT License](LICENSE). Original text and figures are released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) with attribution to PolicyEngine. Third-party data and materials keep their own terms.
