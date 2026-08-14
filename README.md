# imputation-paper

Paper and experiment workspace for Microcosm's weight-aware imputation
approach: the regime-gated, sequentially-chained, weighted-bootstrap
quantile-regression forest in
[`microcosm-fit`](https://github.com/PolicyEngine/microcosm/tree/main/packages/microcosm-fit),
benchmarked against standard survey-imputation methods (plain QRF, OLS,
quantile regression, hot-deck statistical matching) with ablations attributing
the gains to each design choice. (Microcosm was first released under the name
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
