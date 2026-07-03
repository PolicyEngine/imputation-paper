# Extending forest imputation tails: option space (PAUSED — no decision)

Status: deliberately paused (2026-07-03). This memo preserves the option
space and literature pointers for the measured limitation that all
conditional forest methods understate the extreme tail (imputed net-worth
q99 ratios 0.70–0.81 vs the holdout across the paper's harness runs;
donation-based draws sit near 1; the sampling floor wobbles to ~1.10).
Support-bounded draws cannot exceed observed donor values, and the weighted
bootstrap further thins rare high-weight-relevant donors.

## Options, ordered roughly by principledness

1. **Rank-preserving quantile mapping (marginal repair).** Post-draw,
   monotonically map the imputed weighted CDF onto the donor's weighted CDF;
   records keep conditional ranks, the marginal (tail included) becomes exact
   by construction. No threshold, deterministic, composes with any method.
   Homes: quantile/CDF mapping in statistical downscaling (Cannon et al.,
   quantile delta mapping); rank-preserving transformations in the
   wage-decomposition literature. Risk: repairs the marginal at the
   conditional's expense where they disagree — measurable via C2ST/energy.
2. **Conditional generalized-Pareto tail (EVT).** Forest body below a
   threshold; GPD draws above it, parameters from the donor's weighted
   exceedances, optionally covariate-conditional. Domain-standard for
   wealth/income tops (Vermeulen's fat-tail adjustment; WID generalized
   Pareto interpolation, Blanchet–Fournier–Piketty). Forest-native versions
   exist: extremal random forests (Gnecco, Terefe & Engelke, JASA ~2024) and
   gradient-boosted extreme quantile regression (Velthoen et al., Extremes).
   Threshold selection has principled EVT diagnostics (mean-excess /
   parameter-stability plots) — estimated, not hand-picked.
3. **Hot-deck tail splice.** Donate weighted donor values above a conditional
   quantile. Simplest; splice point is genuinely arbitrary; two draw
   mechanisms welded together. Dominated by 1/2/6 unless measurement says
   otherwise.
4. **Leaf-level kernel smoothing.** Extends support marginally (smoothed
   bootstrap); cannot close a ~25% q99 gap alone.
5. **Does NOT work: transform-space fitting.** Quantiles are equivariant
   under monotone transforms, so log/asinh fitting leaves draw support
   bounded by the observed max. (Helps mean-regression retransformation,
   not quantile draws.)
6. **Per-variable donation.** Hot deck outright for designated extreme-tail
   variables; measured near-1 tails and near-peer joints in the paper. Costs:
   donor reuse, matching's gluing on multi-target blocks (mitigate by
   donating whole tail blocks per record).

## Evaluation plan (when unpaused)

Pilot 1 and 2 with 6 as the pragmatic control, adjudicated by the harness's
tail block + C2ST + energy against the sampling floor, at both profiles.
Option 2 is publishable as its own methods contribution ("extremal tails for
weighted survey imputation").

Separate issue regardless of method: even a perfect fit only matches the
donor survey's tail; the true top (rich-list) undercoverage is a donor
problem (Vermeulen-style augmentation), not a method problem.
