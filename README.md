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
