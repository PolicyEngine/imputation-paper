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
2025 dollars; the loader identifies each extract by SHA-256 and returns the
survey year's dollars by default, so the paper's 2022 inputs are unchanged
whichever copy is cached. With the default OpenMP thread count (one per
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
  `paper/sections/`.
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
