# Contributing

## Review tiers

Work in this portfolio routes by issue label: `tier:fable` (frontier-model:
design, claim-bearing prose, statistical semantics) versus `tier:standard`
(spec-complete build/assembly; acceptance tests judge the output). Full
doctrine: PolicyEngine/populace#305.

## Guardian files

Changes to the following require frontier-model (or maintainer) review
regardless of green CI — their failure mode is plausible-and-silently-wrong,
and this repo has already caught three such bugs that passed lint and tests
(an int16 overflow silently dropping 29% of SCF households, joint metrics
certifying a 2x tail error, and best-value bolding with inverted semantics):

- `src/imputation_paper/experiments/metrics.py` and
  `src/imputation_paper/experiments/views.py` (until extracted to popdgp),
  and `src/imputation_paper/experiments/holdout.py`
- every loader's weight handling in `src/imputation_paper/data/` (implied
  decimals, implicate filters, weight scaling, reference-person selection)
- `src/imputation_paper/cli/tables.py` best-value semantics (closest-to-truth
  for fragility, closest-to-1 for tail ratios, closest-to-0.5 for AUC)
