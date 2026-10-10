# SCF 2025 replication: criteria fixed before running

Written and committed on 2026-10-09, before any method was fit on the 2025
Survey of Consumer Finances. The Federal Reserve released that wave the same
day. Every SCF experiment in the paper runs on the 2022 wave, and every design
decision in the estimator and the harness was made with 2022 results in view.
The 2025 wave is data no method here was tuned on. This file states which of
the paper's findings count as replicated on it, and by what rule, so the rules
cannot bend toward the results.

## What was looked at before writing this

- The committed 2022 runs (`runs/scf-wealth`, `runs/scf-to-cps-harness`,
  `runs/scf-to-cps-harness-scale`) and the paper text. Every 2022 number below
  is computed from those files.
- The Fed's 2025 codebook (`codebk2025.txt`) and its summary-variable program
  (`bulletin.macro.txt`, last modified 2026-10-09).
- The 2025 summary extract's schema only: column names, row and family counts,
  the code sets of the categorical predictors, and NaN counts.

No dollar value, weight, or method output from the 2025 wave was examined. The
task brief supplied two published figures (all-family median and mean net
worth from the Fed's Table 2); they play no part in the criteria below.

## Schema findings that shape the design

1. **All 17 columns the two profiles use exist in the 2025 extract** (`y1`,
   `yy1`, `wgt`, `age`, `hhsex`, `edcl`, `married`, `kids`, `income`,
   `wageinc`, `race`, `housecl`, `lf`, `fin`, `nfin`, `debt`, `networth`), with
   the same code sets as 2022 and no missing values. The 2025 extract has
   21,800 rows (4,360 families, five implicates) and 362 columns; it drops no
   column the 2022 extract had and adds five.
2. **One used column changes definition: `race`.** The macro builds `RACE`
   from the general race question (`X6809`) in both waves. In 2025 it folds
   the Middle Eastern code (7) into class 1 (white). The separate Hispanic-origin
   questions the 2025 questionnaire dropped (`X7004`, `X7404`) never entered
   `RACE`; they fed `H_RACE` and `H_RACECL`, which the paper does not use. Of
   the 177 variables the codebook lists as removed since 2022, the macro
   references 16, all in lender-type arrays, a 2016–2022 education-loan branch,
   or those other race variables. None enters the 17 columns above.
3. **The Fed re-issued the 2022 extract on 2026-10-09, re-based to 2025
   dollars, at the same URL.** The file the paper's runs used
   (SHA-256 `3bb4d890…c9d80`, 2022 dollars) is no longer what
   `scfp2022s.zip` serves (now SHA-256 `d4e139c6…2d657`). Every dollar column
   in the new file equals the old one times 4789/4376 = 1.094378 (the macro's
   CPI-U-RS factors for September 2025 and September 2022); weights and
   categorical columns are identical. Two consequences:
   - A cold-cache `load_scf(2022)` no longer reproduces the paper. The loader
     will pin file hashes and return a stated dollar basis.
   - The 2022 runs are in 2022 dollars and the 2025 runs will be in 2025
     dollars, so dollar-denominated metrics are not compared in levels across
     waves (see "Comparability").

## Design

Same code, same locked environment (`uv.lock` at `origin/main` c040b09:
populace-fit at 71b5a83, py-statmatch at 094a242, microimpute 1.15.1), same
seeds (0–9), same 80/20 protocol, same receiver cap (20,000 rows, seed 0).
Only the donor wave changes.

| Arm | Donor | Dollars | Role | Run folders |
| --- | --- | --- | --- | --- |
| A | SCF 2022, file the paper used | 2022 | The paper's committed runs, untouched | `scf-wealth`, `scf-to-cps-harness`, `scf-to-cps-harness-scale` |
| B | SCF 2025 | 2025 | **The replication** | same names with `-2025` |
| C | SCF 2022, re-based | 2025 | Diagnostic: separates the dollar basis from the wave | same names with `-2022-in-2025-dollars` |
| D | SCF 2022, file the paper used | 2022 | Diagnostic: does today's environment reproduce arm A? | same names with `-2022-rerun` |

**Receiver.** The same ASEC vintages the paper uses: ASEC 2025 for the
minimal profile and pooled ASEC 2023–2025 for the populace-scale profile.
ASEC 2026 is on census.gov (posted 2026-09-15) but is not wired into the
harness, and holding the receiver fixed is what makes arm B a clean change of
donor wave.

**Verdicts come from arm B against arm A.** Arms C and D carry no verdicts.
If arm D shows that the current environment does not reproduce arm A (any
displayed table cell's ten-seed mean differing from the committed value by
more than half the committed seed sd), arm D replaces arm A as the baseline
and the report says so.

**Not rerun.** The CPS-components and OpenML tasks do not involve the SCF.
The paper's headline gating result (0.6 against 4.4 percentage points of
dividend zero-share error) and its fragility results come from the
CPS-components task, so a new SCF wave says nothing about them.

## Comparability

- **Scale-free metrics compare directly**: zero-share error, energy distance,
  PRDC coverage, C2ST AUC, tail W1/sd, and the q90 and q99 ratios.
- **Dollar-denominated metrics** (pinball loss and Wasserstein-1 in the
  within-SCF task) are compared only as within-wave ratios between methods.
  The comparison table also shows 2022 levels re-based to 2025 dollars with
  the Fed's factor 4789/4376, for reading, not for verdicts.
- **Levels shared by every method may move and do not count against
  replication.** SCF 2025 incomes refer to 2024, the same year as the ASEC
  2025 receiver's; SCF 2022 incomes refer to 2021. The donor–receiver offset
  that every method shares can therefore shrink. The paper's claims are
  contrasts between methods, and so are the criteria.

## Rules

Notation: `x[m, s]` is a metric for method `m` on seed `s`; `mean[m]` is its
ten-seed mean; for a contrast of method B against method A,
`d[s] = x[B, s] − x[A, s]`, `dbar` its mean, and `k` the number of seeds on
which `d[s]` has the claimed sign. In the harness runs the **floor sd** is the
ten-seed sample sd (ddof = 1) of the `scf_sample_reference` row for the same
metric, profile, and wave. The paper uses that row to calibrate sampling
noise, and so do these rules.

**Difference claims** ("B is worse than A on this metric").

- Within-SCF task: *met* if `dbar` has the claimed sign and `k ≥ 8`; *failed*
  if `dbar` has the opposite sign or is zero; otherwise *inconclusive*.
- Harness: *met* if `dbar` has the claimed sign, `|dbar| > 2 × floor sd`, and
  `k ≥ 8`; *failed* if `dbar` has the opposite sign or `|dbar| ≤ floor sd`;
  otherwise *inconclusive*.

**Tie claims** ("B is indistinguishable from A on this metric").

- Within-SCF task: *met* if neither method wins on 8 or more seeds and the
  ratio of means lies in [0.8, 1.25]; *failed* if one method wins on 8 or
  more seeds and the ratio lies outside that band; otherwise *inconclusive*.
- Harness: *met* if `|dbar| ≤ floor sd`; *failed* if `|dbar| > 2 × floor sd`;
  otherwise *inconclusive*.

**Level claims** carry their own thresholds in the table below.

**Magnitude.** Where the paper states an effect size, the effect is
`mean[B] / mean[A] − 1` for a ratio claim and `dbar` for a difference claim.
The magnitude *holds* if the 2025 effect lies within a factor of two of the
2022 effect, that is in [e/2, 2e].

**Verdicts.**

- **Replicated**: the rule is met and, where a magnitude is attached, it holds.
- **Replicated in direction, magnitude differs**: the rule is met and the
  magnitude falls outside the band (the report says larger or smaller).
- **Inconclusive**: the rule is neither met nor failed.
- **Not replicated**: the rule is failed.

The same rules are applied to the 2022 runs, and the report shows that
result beside each 2025 verdict. Some of the paper's 2022 claims do not meet
these rules on the 2022 data itself; those are marked below. A claim that was
within seed noise in 2022 can at best come out inconclusive.

## Findings

Primary findings are the SCF-based claims the abstract or the conclusion
makes. The rest are secondary. "Candidate" is `populace_fit`.

### Within-SCF wealth task (`scf_wealth` table, SCF rows of the ablations table)

| ID | Finding in the paper (2022 value) | Rule on the 2025 run | 2022 under the same rule |
| --- | --- | --- | --- |
| W1 (primary) | Fit unweighted, the same estimator is six times worse on net-worth W1 (829,158 against 135,871; ratio 6.10). | Difference, unweighted against candidate, net-worth W1. Magnitude: ratio in [3.55, 11.21]. | Met, 10/10 |
| W2 | Unweighted is 2.5 times worse on debt W1 (32,247 against 13,001; ratio 2.48). | Difference, unweighted against candidate, debt W1. Magnitude: ratio in [1.74, 3.96]. | Met, 10/10 |
| W3a | Weighting is the largest of the three ablation effects on net-worth W1 (+693,287 against +98,496 for the ungated, unchained forest and +22,735 for unchained). | Level: the unweighted ablation's mean paired delta on net-worth W1 is the largest of the three, and positive. | Met |
| W3b | The ungated, unchained forest is worse than the candidate on net-worth W1 (ratio 1.72). | Difference, `plain_qrf` against candidate, net-worth W1. Magnitude: ratio in [1.36, 2.45]. | **Inconclusive, 6/10** |
| W3c | Removing chaining raises net-worth W1 by 17 percent ("consistent but smaller"). | Difference, unchained against candidate, net-worth W1. Magnitude: ratio in [1.08, 1.33]. | **Inconclusive, 7/10** |
| W4a | Hot deck nearly ties the candidate on debt W1 (13,770 against 13,001). | Tie, hot deck against candidate, debt W1. | Met, 5/10 each way, ratio 1.06 |
| W4b | Hot deck is 3.7 times worse on net-worth W1 (509,013; ratio 3.75). | Difference, hot deck against candidate, net-worth W1. Magnitude: ratio in [2.37, 6.49]. | Met, 9/10 |
| W5 | OLS is catastrophic on both targets (debt W1 ratio 12.0, net-worth W1 ratio 26.2). | Level: OLS has the highest mean W1 of all methods on both targets, at least five times the candidate's on each. | Met |
| W6 | Linear quantile regression leaves 15 percent of the debt zero mass unaccounted for, against at most 2 percent for the forest-based methods. | Level: quantile regression's mean debt zero-share error is at least 0.10, and the candidate, its two ablations, and `plain_qrf` are each at most 0.03. | Met (0.151; 0.012 to 0.020) |
| W7 | The unconditional weighted marginal draw has the best pinball loss on both targets. | Level: `weighted_marginal` has the lowest mean pinball loss of all methods on debt and on net worth. | Met |
| G1 | Gating shows no advantage on SCF debt zero share: the candidate (0.020) and the ungated forest (0.018) both hold it. | Tie, `plain_qrf` against candidate, debt zero-share error, judged on seeds alone (neither wins on 8 or more); the ratio band is not applied because both values are near zero. | Met, 4/10 and 6/10 |

### Population view, both profiles (`harness` and `harness_scale` tables)

"Min" is the minimal profile and "scale" the populace-scale profile. A row
with two 2022 values lists min then scale, and the rule must hold in both
profiles unless it names one.

| ID | Finding in the paper (2022 value) | Rule on the 2025 runs | 2022 under the same rule |
| --- | --- | --- | --- |
| F0 | The floor behaves as designed: AUC 0.496 and 0.499, energy 0.009 and 0.012, coverage 0.987 and 0.986, net-worth q99 ratio 1.10. | Level, both profiles: floor AUC within 2 floor sd of 0.5; floor energy below one fifth of every method's; floor coverage at least every method's; floor net-worth q99 ratio within 2 floor sd of 1. | Met |
| F1a (primary) | The unweighted ablation ties the candidate on energy distance, coverage, and C2ST AUC in both profiles. | Tie, unweighted against candidate, on each of the three metrics in each profile (six ties; all must be met). | Met (largest gap 0.72 floor sd) |
| F1b (primary) | Its imputed net-worth q99 is 1.93 (min) and 2.25 (scale) times the holdout's: "wrong by a factor of two". | Difference, unweighted against candidate, net-worth q99 ratio, plus level: the unweighted mean exceeds 1.5 in both profiles. Magnitude: mean in [1.47, 2.86] (min) and [1.62, 3.50] (scale). | Met, 10/10 both |
| F1c | Its tail W1/sd is five times the candidate's (0.159 against 0.031; 0.133 against 0.027). | Difference, unweighted against candidate, net-worth W1/sd. Magnitude: ratio in [3.04, 9.16] (min) and [2.97, 8.89] (scale). | Met, 10/10 both |
| F2a (primary) | With two targets, the unchained ablation is indistinguishable from the candidate (AUC 0.742 against 0.732). | Tie, unchained against candidate, C2ST AUC, minimal profile. | Met (0.40 floor sd) |
| F2b (primary) | On the four-component balance sheet, fitting targets independently costs 5.8 AUC points (0.819 against 0.761). | Difference, unchained against candidate, C2ST AUC, scale profile. Magnitude: `dbar` in [0.029, 0.115]. | Met, 9/10, 2.70 floor sd |
| F2c | Unchaining leaves the marginal and tail statistics essentially unchanged at scale. | Tie, unchained against candidate, scale profile, on energy distance, coverage, net-worth W1/sd, and net-worth q99 ratio (all four must be met). | Met (largest gap 0.76 floor sd) |
| F2d | The ungated forest loses 5.6 AUC points at scale (0.817). | Difference, `plain_qrf` against candidate, C2ST AUC, scale profile. Magnitude: `dbar` in [0.028, 0.112]. | Met, 10/10, 2.63 floor sd |
| F3a (primary) | The classifier separates the unconditional marginal draw almost perfectly at scale: AUC 0.968 against the candidate's 0.761. | Difference, `weighted_marginal` against candidate, C2ST AUC, scale profile, plus level: the marginal draw's mean AUC is at least 0.95. Magnitude: `dbar` in [0.103, 0.413]. | Met, 10/10 |
| F3b | Conditioning richness makes the test decisive: the marginal draw's AUC rises from 0.848 (min) to 0.968 (scale). | Level: the marginal draw's mean AUC at scale exceeds its minimal-profile mean by more than 0.060 (half the 2022 rise of 0.120). | Met |
| F3c | The candidate stays closest to the floor (0.732 and 0.761). | Level, both profiles: the candidate has the lowest mean AUC among the candidate and the four baselines (OLS, quantile regression, hot deck, marginal draw). Its own ablations are excluded: in 2022 the unweighted ablation is 0.002 and 0.007 lower. | Met |
| F4a (primary) | Hot deck is a near-peer: within noise of the candidate on energy distance and coverage in both profiles. | Tie, hot deck against candidate, on each of the two metrics in each profile (four ties; all must be met). | Met (largest gap 0.71 floor sd) |
| F4b (primary) | Hot deck is within 2.1 (min) and 0.7 (scale) AUC points of the candidate. | Tie, hot deck against candidate, C2ST AUC, each profile. | Met (0.86 and 0.30 floor sd) |
| F4c | At scale hot deck slightly beats the candidate on tail W1/sd (0.024 against 0.027). | Level, scale profile: hot deck's mean net-worth W1/sd is no more than one floor sd above the candidate's. | Met (0.90 floor sd below) |
| F5a | The candidate understates the extreme tail: net-worth q99 ratio 0.70 (min) and 0.77 (scale). | Level, both profiles: the candidate's ratio is below 1 on at least 8 of 10 seeds and its mean lies in [0.60, 0.90]. | Met (10/10 and 9/10) |
| F5b | Read from the 2022 tables rather than the text (see the note below): hot deck does not fix the tail either (0.76 and 0.75), while the marginal draw sits near the floor (1.09 and 1.12 against 1.10). | Tie, hot deck against candidate, and tie, marginal draw against floor, on the net-worth q99 ratio, each profile. | Met |

**Two places where the 2022 text and the 2022 tables already disagree.**
They are recorded here because they bear on F5, and are not replication
failures.

- The results and discussion sections say forest draws' q99 ratios "sit at
  0.70–0.81". That holds in the minimal profile (0.696, 0.765, 0.805). At
  scale the ungated forest is at 1.15 (sd 0.45).
- They also say "donation-based draws sit near 1" and call that "a genuine
  advantage of hot-deck matching". The hot-deck rows are 0.762 and 0.747,
  indistinguishable from the candidate's. The draw that sits near 1 is the
  unconditional marginal draw (1.09 and 1.12).

### Method rankings (every column of the three SCF tables)

For each of the fifteen displayed columns (five per table), rank the eight
displayed non-floor methods by the table's own notion of best (lowest for
pinball, W1, zero-share error, energy distance, and W1/sd; highest for
coverage; closest to 0.5 for AUC; closest to 1 for the q99 ratio). The
ranking **replicates** if the Spearman correlation between the 2022 and 2025
orderings is at least 0.8, **partly replicates** between 0.5 and 0.8, and
**does not replicate** below 0.5. The report also says whether the bolded
best method is the same. Columns in which several methods sat within one
floor sd of each other in 2022 (energy distance and coverage in both
profiles) are expected to reorder; they are reported by the same rule, with
that note.

### Reported without a verdict

The paper makes no claim about the net-worth q90 ratio, or about the tail
metrics of the other three balance-sheet targets at scale. The comparison
table reports them for both waves.

## Differential check against the Fed's published table

Separately from the replication, a test compares the public extract's
all-family median and mean net worth with the Fed's Table 2. The Fed computes
its tables from internal data; the public file is disclosure-limited, so a gap
is expected. Its size is calibrated on 2022, against the Fed's published 2022 figures: the
tolerance for each statistic is the larger of 1 percent and twice the 2022
relative gap, fixed before the 2025 values are computed.

## What the report will contain

- One comparison table with every headline number: 2022 (arm A), 2025 (arm
  B), the rule, the 2022 and 2025 status, and the verdict. Arms C and D appear
  as diagnostic columns.
- The full method-by-metric tables for both waves.
- A robustness appendix draft for the paper stating what replicated and what
  did not. Failures and inconclusive results are reported as plainly as
  successes, and nothing in this file is revised after the runs. If a rule
  turns out to be ill-posed, the report says so and shows the result under the
  rule as written.
