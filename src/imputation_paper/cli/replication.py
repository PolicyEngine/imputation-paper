"""``imp replication``: score the SCF 2025 replication and write its report.

Reads the committed run directories and writes everything the replication
reports, all of it a pure function of those runs and of the rules in
:mod:`imputation_paper.experiments.replication`:

* ``docs/scf-2025-replication/comparison.md`` -- the 2022-versus-2025
  comparison: verdicts, every headline number, method rankings, the full
  tables for both waves, and the diagnostic arms;
* ``findings.csv``, ``headline_numbers.csv``, ``rankings.csv``, and
  ``reproduction_check.csv`` next to it;
* ``paper/tables/scf2025_replication.tex`` -- the verdict table the appendix
  inputs -- and ``paper/tables/scf2025_numbers.tex`` -- one macro per number
  the appendix prose cites, so no number in it is typed by hand.

The arms (see ``docs/scf-2025-replication-criteria.md``): **A** the paper's
committed 2022 runs, **B** the 2025 runs, **C** 2022 re-based to 2025 dollars,
**D** 2022 rerun in today's environment. Verdicts compare B with A. If D shows
that today's environment does not reproduce A, D replaces A as the baseline
and the report says so.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from imputation_paper.cli.tables import HARNESS_COLUMNS, SCF_WEALTH_COLUMNS
from imputation_paper.experiments import replication as rules
from imputation_paper.experiments.replication import (
    CANDIDATE,
    DISPLAYED_METHODS,
    FLOOR,
    HOT_DECK,
    MARGINAL,
    OLS,
    PLAIN_FOREST,
    QUANTREG,
    UNCHAINED,
    UNWEIGHTED,
    Arm,
    Judgement,
)

#: Run-directory suffix of each arm.
ARM_SUFFIXES: dict[str, str] = {
    "A": "",
    "B": "-2025",
    "C": "-2022-in-2025-dollars",
    "D": "-2022-rerun",
}

#: The Fed's factor taking 2022 dollars to 2025 dollars (CPI-U-RS, September).
REBASE_2022_TO_2025 = 4789 / 4376

#: A displayed cell of arm D may sit this many committed seed sds from arm A
#: before the environment counts as not reproducing the paper's runs.
REPRODUCTION_TOLERANCE_SD = 0.5

#: Plain-text method names for the report.
METHOD_NAMES: dict[str, str] = {
    FLOOR: "SCF sample (floor)",
    CANDIDATE: "populace-fit",
    UNWEIGHTED: "populace-fit, unweighted",
    UNCHAINED: "populace-fit, unchained",
    PLAIN_FOREST: "ungated/unchained forest",
    OLS: "OLS",
    QUANTREG: "quantile regression",
    HOT_DECK: "NND hot deck",
    MARGINAL: "weighted marginal draw",
}

#: The three displayed tables: run key -> (title, columns).
TABLES: dict[str, tuple[str, list[tuple[str, str, str]]]] = {
    "wealth": ("Wealth within the SCF", SCF_WEALTH_COLUMNS),
    "min": ("Population view, minimal profile", HARNESS_COLUMNS),
    "scale": ("Population view, populace-scale profile", HARNESS_COLUMNS),
}

_PLAIN_HEADERS = {
    "Debt $W_1$": "Debt W1",
    "Net worth $W_1$": "Net worth W1",
    r"NW $W_1$/sd": "NW W1/sd",
    "Debt zero-share err.": "Debt zero-share error",
}


def _plain(header: str) -> str:
    """A table header without LaTeX markup."""
    return _PLAIN_HEADERS.get(header, header)


# ---------------------------------------------------------------------------
# Headline numbers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Headline:
    """One number the paper's SCF text cites.

    Attributes:
        key: Stable identifier; also the macro key the appendix uses.
        label: What the number is.
        where: Where the paper cites it.
        value: Computes it from an arm.
        kind: ``"dollars"`` (re-based for display), ``"ratio"``, ``"points"``
            (AUC points, times 100), or ``"plain"``.
    """

    key: str
    label: str
    where: str
    value: Callable[[Arm], float]
    kind: str = "plain"


def _mean(
    run: str, method: str, metric: str, target: str | None = None
) -> Callable[[Arm], float]:
    return lambda arm: arm.mean(run, method, metric, target)


def _ratio(
    run: str, metric: str, other: str, target: str | None = None
) -> Callable[[Arm], float]:
    return lambda arm: (
        arm.mean(run, other, metric, target) / arm.mean(run, CANDIDATE, metric, target)
    )


def _gap(
    run: str, metric: str, other: str, scale: float = 1.0
) -> Callable[[Arm], float]:
    return lambda arm: (
        scale * (arm.mean(run, other, metric) - arm.mean(run, CANDIDATE, metric))
    )


def _delta(other: str) -> Callable[[Arm], float]:
    return lambda arm: (
        rules.contrast(arm, "wealth", "wasserstein1", CANDIDATE, other, "networth").dbar
    )


def _max_forest_zero_share(arm: Arm) -> float:
    return max(
        arm.mean("wealth", method, "zero_share_error", "debt")
        for method in (CANDIDATE, *rules.ABLATIONS)
    )


def _headlines() -> list[Headline]:
    """Every SCF number the abstract, results, discussion, or conclusion cites."""
    w1, pin, zero = "wasserstein1", "pinball_loss", "zero_share_error"
    wealth = "SCF wealth table"
    items: list[Headline] = [
        Headline(
            "nw-w1-candidate",
            "Net-worth W1, candidate",
            wealth,
            _mean("wealth", CANDIDATE, w1, "networth"),
            "dollars",
        ),
        Headline(
            "nw-w1-unweighted",
            "Net-worth W1, unweighted",
            wealth,
            _mean("wealth", UNWEIGHTED, w1, "networth"),
            "dollars",
        ),
        Headline(
            "nw-w1-unweighted-ratio",
            "Net-worth W1, unweighted over candidate ('six times worse')",
            "Abstract, results, discussion, conclusion",
            _ratio("wealth", w1, UNWEIGHTED, "networth"),
            "ratio",
        ),
        Headline(
            "debt-w1-candidate",
            "Debt W1, candidate",
            wealth,
            _mean("wealth", CANDIDATE, w1, "debt"),
            "dollars",
        ),
        Headline(
            "debt-w1-unweighted",
            "Debt W1, unweighted",
            wealth,
            _mean("wealth", UNWEIGHTED, w1, "debt"),
            "dollars",
        ),
        Headline(
            "debt-w1-unweighted-ratio",
            "Debt W1, unweighted over candidate ('factor of 2.5')",
            "Results",
            _ratio("wealth", w1, UNWEIGHTED, "debt"),
            "ratio",
        ),
        Headline(
            "nw-w1-forest",
            "Net-worth W1, ungated/unchained forest",
            wealth,
            _mean("wealth", PLAIN_FOREST, w1, "networth"),
            "dollars",
        ),
        Headline(
            "nw-w1-forest-ratio",
            "Net-worth W1, ungated forest over candidate ('1.7 times')",
            "Results",
            _ratio("wealth", w1, PLAIN_FOREST, "networth"),
            "ratio",
        ),
        Headline(
            "nw-w1-unchained",
            "Net-worth W1, unchained",
            wealth,
            _mean("wealth", UNCHAINED, w1, "networth"),
            "dollars",
        ),
        Headline(
            "nw-w1-unchained-ratio",
            "Net-worth W1, unchained over candidate ('17 percent')",
            "Results",
            _ratio("wealth", w1, UNCHAINED, "networth"),
            "ratio",
        ),
        Headline(
            "debt-w1-hotdeck",
            "Debt W1, hot deck",
            wealth,
            _mean("wealth", HOT_DECK, w1, "debt"),
            "dollars",
        ),
        Headline(
            "debt-w1-hotdeck-ratio",
            "Debt W1, hot deck over candidate",
            "Results ('nearly ties')",
            _ratio("wealth", w1, HOT_DECK, "debt"),
            "ratio",
        ),
        Headline(
            "nw-w1-hotdeck",
            "Net-worth W1, hot deck",
            wealth,
            _mean("wealth", HOT_DECK, w1, "networth"),
            "dollars",
        ),
        Headline(
            "nw-w1-hotdeck-ratio",
            "Net-worth W1, hot deck over candidate ('3.7 times worse')",
            "Results",
            _ratio("wealth", w1, HOT_DECK, "networth"),
            "ratio",
        ),
        Headline(
            "debt-w1-ols",
            "Debt W1, OLS",
            wealth,
            _mean("wealth", OLS, w1, "debt"),
            "dollars",
        ),
        Headline(
            "nw-w1-ols",
            "Net-worth W1, OLS",
            wealth,
            _mean("wealth", OLS, w1, "networth"),
            "dollars",
        ),
        Headline(
            "debt-zero-quantreg",
            "Debt zero-share error, quantile regression ('15 percent')",
            "Results",
            _mean("wealth", QUANTREG, zero, "debt"),
        ),
        Headline(
            "debt-zero-forest-max",
            "Debt zero-share error, largest forest-based ('at most 2 percent')",
            "Results",
            _max_forest_zero_share,
        ),
        Headline(
            "debt-zero-candidate",
            "Debt zero-share error, candidate",
            wealth,
            _mean("wealth", CANDIDATE, zero, "debt"),
        ),
        Headline(
            "debt-zero-forest",
            "Debt zero-share error, ungated forest",
            wealth,
            _mean("wealth", PLAIN_FOREST, zero, "debt"),
        ),
        Headline(
            "debt-pinball-marginal",
            "Debt pinball, marginal draw",
            wealth,
            _mean("wealth", MARGINAL, pin, "debt"),
            "dollars",
        ),
        Headline(
            "debt-pinball-candidate",
            "Debt pinball, candidate",
            wealth,
            _mean("wealth", CANDIDATE, pin, "debt"),
            "dollars",
        ),
        Headline(
            "nw-pinball-marginal",
            "Net-worth pinball, marginal draw",
            wealth,
            _mean("wealth", MARGINAL, pin, "networth"),
            "dollars",
        ),
        Headline(
            "nw-pinball-candidate",
            "Net-worth pinball, candidate",
            wealth,
            _mean("wealth", CANDIDATE, pin, "networth"),
            "dollars",
        ),
        Headline(
            "delta-unweighted",
            "Ablation delta on net-worth W1, unweighted",
            "Ablations table",
            _delta(UNWEIGHTED),
            "dollars",
        ),
        Headline(
            "delta-forest",
            "Ablation delta on net-worth W1, ungated forest",
            "Ablations table",
            _delta(PLAIN_FOREST),
            "dollars",
        ),
        Headline(
            "delta-unchained",
            "Ablation delta on net-worth W1, unchained",
            "Ablations table",
            _delta(UNCHAINED),
            "dollars",
        ),
    ]
    energy, coverage, auc = "energy_distance", "prdc_coverage", "c2st_auc"
    tail, q99, q90 = "w1_over_sd.networth", "q99_ratio.networth", "q90_ratio.networth"
    for run, name, table in (
        ("min", "minimal", "Minimal-profile table"),
        ("scale", "scale", "Populace-scale table"),
    ):
        items += [
            Headline(
                f"{run}-floor-energy",
                f"Floor energy distance, {name}",
                table,
                _mean(run, FLOOR, energy),
            ),
            Headline(
                f"{run}-floor-coverage",
                f"Floor coverage, {name}",
                table,
                _mean(run, FLOOR, coverage),
            ),
            Headline(
                f"{run}-floor-auc",
                f"Floor C2ST AUC, {name}",
                table,
                _mean(run, FLOOR, auc),
            ),
            Headline(
                f"{run}-floor-q99",
                f"Floor net-worth q99 ratio, {name}",
                table,
                _mean(run, FLOOR, q99),
            ),
            Headline(
                f"{run}-energy-candidate",
                f"Energy distance, candidate, {name}",
                table,
                _mean(run, CANDIDATE, energy),
            ),
            Headline(
                f"{run}-energy-unweighted",
                f"Energy distance, unweighted, {name}",
                table,
                _mean(run, UNWEIGHTED, energy),
            ),
            Headline(
                f"{run}-energy-hotdeck",
                f"Energy distance, hot deck, {name}",
                table,
                _mean(run, HOT_DECK, energy),
            ),
            Headline(
                f"{run}-coverage-candidate",
                f"Coverage, candidate, {name}",
                table,
                _mean(run, CANDIDATE, coverage),
            ),
            Headline(
                f"{run}-coverage-unweighted",
                f"Coverage, unweighted, {name}",
                table,
                _mean(run, UNWEIGHTED, coverage),
            ),
            Headline(
                f"{run}-coverage-hotdeck",
                f"Coverage, hot deck, {name}",
                table,
                _mean(run, HOT_DECK, coverage),
            ),
            Headline(
                f"{run}-auc-candidate",
                f"C2ST AUC, candidate, {name}",
                table,
                _mean(run, CANDIDATE, auc),
            ),
            Headline(
                f"{run}-auc-unweighted",
                f"C2ST AUC, unweighted, {name}",
                table,
                _mean(run, UNWEIGHTED, auc),
            ),
            Headline(
                f"{run}-auc-unchained",
                f"C2ST AUC, unchained, {name}",
                table,
                _mean(run, UNCHAINED, auc),
            ),
            Headline(
                f"{run}-auc-unchained-gap",
                f"AUC points lost by unchaining, {name}",
                "Abstract, results ('5.8 points' at scale)",
                _gap(run, auc, UNCHAINED, 100.0),
                "points",
            ),
            Headline(
                f"{run}-auc-forest",
                f"C2ST AUC, ungated forest, {name}",
                table,
                _mean(run, PLAIN_FOREST, auc),
            ),
            Headline(
                f"{run}-auc-forest-gap",
                f"AUC points lost by the ungated forest, {name}",
                "Results, discussion ('5.6 points' at scale)",
                _gap(run, auc, PLAIN_FOREST, 100.0),
                "points",
            ),
            Headline(
                f"{run}-auc-hotdeck",
                f"C2ST AUC, hot deck, {name}",
                table,
                _mean(run, HOT_DECK, auc),
            ),
            Headline(
                f"{run}-auc-hotdeck-gap",
                f"AUC points, hot deck above candidate, {name}",
                "Results ('within 2.1 and 0.7 AUC points')",
                _gap(run, auc, HOT_DECK, 100.0),
                "points",
            ),
            Headline(
                f"{run}-auc-marginal",
                f"C2ST AUC, marginal draw, {name}",
                "Abstract, results ('0.848' and '0.968')",
                _mean(run, MARGINAL, auc),
            ),
            Headline(
                f"{run}-auc-quantreg",
                f"C2ST AUC, quantile regression, {name}",
                table,
                _mean(run, QUANTREG, auc),
            ),
            Headline(
                f"{run}-auc-ols", f"C2ST AUC, OLS, {name}", table, _mean(run, OLS, auc)
            ),
            Headline(
                f"{run}-tail-candidate",
                f"Net-worth W1/sd, candidate, {name}",
                table,
                _mean(run, CANDIDATE, tail),
            ),
            Headline(
                f"{run}-tail-unweighted",
                f"Net-worth W1/sd, unweighted, {name}",
                table,
                _mean(run, UNWEIGHTED, tail),
            ),
            Headline(
                f"{run}-tail-unweighted-ratio",
                f"Net-worth W1/sd, unweighted over candidate, {name}",
                "Results ('five times')",
                _ratio(run, tail, UNWEIGHTED),
                "ratio",
            ),
            Headline(
                f"{run}-tail-hotdeck",
                f"Net-worth W1/sd, hot deck, {name}",
                table,
                _mean(run, HOT_DECK, tail),
            ),
            Headline(
                f"{run}-q99-candidate",
                f"Net-worth q99 ratio, candidate, {name}",
                table,
                _mean(run, CANDIDATE, q99),
            ),
            Headline(
                f"{run}-q99-unweighted",
                f"Net-worth q99 ratio, unweighted, {name}",
                "Abstract, results, discussion, conclusion ('1.9 to 2.3')",
                _mean(run, UNWEIGHTED, q99),
            ),
            Headline(
                f"{run}-q99-unchained",
                f"Net-worth q99 ratio, unchained, {name}",
                table,
                _mean(run, UNCHAINED, q99),
            ),
            Headline(
                f"{run}-q99-forest",
                f"Net-worth q99 ratio, ungated forest, {name}",
                table,
                _mean(run, PLAIN_FOREST, q99),
            ),
            Headline(
                f"{run}-q99-hotdeck",
                f"Net-worth q99 ratio, hot deck, {name}",
                table,
                _mean(run, HOT_DECK, q99),
            ),
            Headline(
                f"{run}-q99-marginal",
                f"Net-worth q99 ratio, marginal draw, {name}",
                table,
                _mean(run, MARGINAL, q99),
            ),
            Headline(
                f"{run}-q90-floor",
                f"Net-worth q90 ratio, floor, {name}",
                "Not cited (reported without a verdict)",
                _mean(run, FLOOR, q90),
            ),
            Headline(
                f"{run}-q90-candidate",
                f"Net-worth q90 ratio, candidate, {name}",
                "Not cited (reported without a verdict)",
                _mean(run, CANDIDATE, q90),
            ),
            Headline(
                f"{run}-q90-unweighted",
                f"Net-worth q90 ratio, unweighted, {name}",
                "Not cited (reported without a verdict)",
                _mean(run, UNWEIGHTED, q90),
            ),
            Headline(
                f"{run}-q90-hotdeck",
                f"Net-worth q90 ratio, hot deck, {name}",
                "Not cited (reported without a verdict)",
                _mean(run, HOT_DECK, q90),
            ),
            Headline(
                f"{run}-q90-marginal",
                f"Net-worth q90 ratio, marginal draw, {name}",
                "Not cited (reported without a verdict)",
                _mean(run, MARGINAL, q90),
            ),
        ]
    for target in ("fin", "nfin", "debt"):
        for method, name in ((CANDIDATE, "candidate"), (UNWEIGHTED, "unweighted")):
            items.append(
                Headline(
                    f"scale-q99-{target}-{name}",
                    f"{target} q99 ratio, {name}, scale",
                    "Not cited (reported without a verdict)",
                    _mean("scale", method, f"q99_ratio.{target}"),
                )
            )
    return items


HEADLINES: tuple[Headline, ...] = tuple(_headlines())


def _show(value: float, kind: str) -> str:
    """Format one headline value."""
    if not np.isfinite(value):
        return "--"
    if kind == "dollars":
        return f"{value:,.0f}"
    if kind == "ratio":
        return f"{value:.2f}"
    if kind == "points":
        return f"{value:.1f}"
    return f"{value:.3f}" if abs(value) < 1 else f"{value:.2f}"


def headline_table(arms: dict[str, Arm]) -> pd.DataFrame:
    """Every headline number for every arm present.

    One column per arm letter. Arms A and D are in 2022 dollars and arms B and
    C in 2025 dollars; the report re-bases the 2022-dollar levels with
    :data:`REBASE_2022_TO_2025` where it sets them beside 2025.
    """
    rows = []
    for item in HEADLINES:
        row: dict[str, object] = {
            "key": item.key,
            "number": item.label,
            "where": item.where,
            "kind": item.kind,
        }
        for letter, arm in arms.items():
            row[letter] = item.value(arm)
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Reproduction check (arm D against arm A)
# ---------------------------------------------------------------------------


def reproduction_check(arm_a: Arm, arm_d: Arm) -> pd.DataFrame:
    """Every displayed cell of the paper's SCF tables: committed versus rerun.

    Returns:
        One row per table cell (floor rows included) with both means, the
        committed seed sd, the gap in committed sds, the largest per-seed
        relative difference, and whether all ten seeds match exactly.
    """
    rows = []
    for run, (_, columns) in TABLES.items():
        methods = list(DISPLAYED_METHODS)
        if run != "wealth":
            methods = [FLOOR, *methods]
        for target, metric, header in columns:
            wealth_target = target if run == "wealth" else None
            for method in methods:
                a = arm_a.values(run, method, metric, wealth_target)
                d = arm_d.values(run, method, metric, wealth_target)
                sd = float(a.std(ddof=1))
                gap = abs(float(d.mean() - a.mean()))
                scale = np.maximum(a.abs().to_numpy(), 1e-300)
                rows.append(
                    {
                        "run": run,
                        "column": _plain(header),
                        "method": method,
                        "mean_committed": float(a.mean()),
                        "sd_committed": sd,
                        "mean_rerun": float(d.mean()),
                        "gap_in_sd": gap / sd
                        if sd > 0
                        else (0.0 if gap == 0 else np.inf),
                        "max_seed_relative_difference": float(
                            np.max(np.abs(d.to_numpy() - a.to_numpy()) / scale)
                        ),
                        "identical": bool(np.array_equal(a.to_numpy(), d.to_numpy())),
                    }
                )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Report:
    """Everything the replication reports, as frames.

    Attributes:
        arms: Arm letter -> loaded arm, for the arms that have runs.
        baseline_letter: ``"A"`` unless arm D failed the reproduction check.
        judgements: One per finding, replication (B) against the baseline.
        findings: The verdict table.
        headlines: Every headline number, per arm.
        rankings: Spearman agreement per displayed column.
        reproduction: Arm D against arm A per displayed cell (empty without D).
        diagnostics: Each finding's status on arms C and D (empty without them).
    """

    arms: dict[str, Arm]
    baseline_letter: str
    judgements: list[Judgement]
    findings: pd.DataFrame
    headlines: pd.DataFrame
    rankings: pd.DataFrame
    reproduction: pd.DataFrame
    diagnostics: pd.DataFrame


def _band_text(judgement: Judgement) -> str:
    """The magnitude band and the replication's effect, or an empty string."""
    if not judgement.bands:
        return ""
    name = judgement.finding.effect_name
    parts = [
        f"{effect:.2f} in [{low:.2f}, {high:.2f}]"
        if judgement.finding.effect_offset
        else f"{effect:.3f} in [{low:.3f}, {high:.3f}]"
        for effect, (low, high) in zip(judgement.effects, judgement.bands, strict=True)
    ]
    held = {True: "holds", False: "outside", None: "not assessed"}[
        judgement.magnitude_holds
    ]
    return f"{name}: {'; '.join(parts)} ({held})"


def build_report(runs_dir: Path = Path("runs")) -> Report:
    """Load the arms, apply the rules, and assemble the report frames.

    Raises:
        FileNotFoundError: If arm A or arm B is missing a run.
    """
    arms: dict[str, Arm] = {
        "A": rules.load_arm(runs_dir, ARM_SUFFIXES["A"], "2022 (paper)"),
        "B": rules.load_arm(runs_dir, ARM_SUFFIXES["B"], "2025"),
    }
    for letter, label in (("C", "2022 in 2025 dollars"), ("D", "2022 rerun")):
        try:
            arms[letter] = rules.load_arm(runs_dir, ARM_SUFFIXES[letter], label)
        except FileNotFoundError:
            continue

    reproduction = pd.DataFrame()
    baseline_letter = "A"
    if "D" in arms:
        reproduction = reproduction_check(arms["A"], arms["D"])
        if (reproduction["gap_in_sd"] > REPRODUCTION_TOLERANCE_SD).any():
            baseline_letter = "D"
    baseline = arms[baseline_letter]

    judgements = [rules.judge(f, baseline, arms["B"]) for f in rules.FINDINGS]
    findings = pd.DataFrame(
        {
            "id": j.finding.key,
            "primary": j.finding.primary,
            "finding": j.finding.claim,
            "rule": j.finding.rule,
            "value_2022": j.baseline.text,
            "value_2025": j.replication.text,
            "status_2022": j.baseline.status,
            "status_2025": j.replication.status,
            "magnitude": _band_text(j),
            "verdict": j.verdict,
        }
        for j in judgements
    )

    diagnostic_rows = []
    for finding in rules.FINDINGS:
        row: dict[str, object] = {"id": finding.key}
        for letter in ("C", "D"):
            if letter in arms:
                evaluation = finding.evaluate(arms[letter])
                row[f"status_{letter}"] = evaluation.status
                row[f"value_{letter}"] = evaluation.text
        diagnostic_rows.append(row)
    diagnostics = pd.DataFrame(diagnostic_rows)

    columns = {run: spec for run, (_, spec) in TABLES.items()}
    rankings = rules.ranking_agreement(baseline, arms["B"], columns)
    rankings["column"] = rankings["column"].map(_plain)
    return Report(
        arms=arms,
        baseline_letter=baseline_letter,
        judgements=judgements,
        findings=findings,
        headlines=headline_table(arms),
        rankings=rankings,
        reproduction=reproduction,
        diagnostics=diagnostics,
    )


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    """A GitHub-flavored markdown table (pipes in cells are escaped)."""

    def clean(cell: object) -> str:
        return str(cell).replace("|", r"\|")

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines += ["| " + " | ".join(clean(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def _wave_table(arm: Arm, run: str) -> str:
    """One displayed table for one arm, as markdown ``mean (sd)`` cells."""
    _, columns = TABLES[run]
    methods = list(DISPLAYED_METHODS)
    if run != "wealth":
        methods = [FLOOR, *methods]
    rows = []
    for method in methods:
        cells = [METHOD_NAMES[method]]
        for target, metric, _ in columns:
            values = arm.values(
                run, method, metric, target if run == "wealth" else None
            )
            cells.append(
                f"{rules.format_number(float(values.mean()))} "
                f"({rules.format_number(float(values.std(ddof=1)))})"
            )
        rows.append(cells)
    return _md_table(["Method", *(_plain(h) for _, _, h in columns)], rows)


def _count(findings: pd.DataFrame, primary: bool) -> str:
    """Verdict counts for the primary or secondary findings."""
    subset = findings[findings["primary"].eq(primary)]
    counts = subset["verdict"].value_counts()
    return f"{len(subset)} findings: " + "; ".join(
        f"{count} {verdict.lower()}" for verdict, count in counts.items()
    )


def _checks_section(checks: dict | None) -> list[str]:
    """The published-table check, from ``scf-population-checks/manifest.json``."""
    lines = ["## Check against the Fed's published tables", ""]
    if checks is None:
        return [
            *lines,
            "Not run: `runs/scf-population-checks/manifest.json` is absent "
            "(`imp scf-checks` writes it).",
            "",
        ]
    lines += [
        "All five implicates, pooled, weighted by `wgt`, in 2025 dollars. The "
        "Fed computes its tables from the internal data; the public extract is "
        "altered to protect respondents and holds seven fewer families than "
        "were interviewed in each wave. The tolerance on 2025 is the larger of "
        "a fixed floor and twice the 2022 gap.",
        "",
    ]
    rows = []
    labels = {
        "weight_total": "Families represented",
        "net_worth_median": "Median net worth, all families",
        "net_worth_mean": "Mean net worth, all families",
    }
    for key, label in labels.items():
        gap = checks["published_gaps"][key]
        rows.append(
            [
                label,
                f"{gap['published_calibration']:,.0f}",
                f"{gap['public_calibration']:,.0f}",
                f"{100 * gap['gap_calibration']:+.2f}%",
                f"{gap['published_replication']:,.0f}",
                f"{gap['public_replication']:,.0f}",
                f"{100 * gap['gap_replication']:+.2f}%",
                f"{100 * gap['tolerance']:.2f}%",
                "yes" if gap["within_tolerance"] else "**no**",
            ]
        )
    lines += [
        _md_table(
            [
                "Statistic",
                "2022 published",
                "2022 public",
                "2022 gap",
                "2025 published",
                "2025 public",
                "2025 gap",
                "Tolerance",
                "Within",
            ],
            rows,
        ),
        "",
    ]
    rows = []
    for year in ("2022", "2025"):
        wave = checks["waves"][year]
        rows.append(
            [
                year,
                f"{wave['rows']:,}",
                f"{wave['families']:,}",
                str(wave["implicates"]),
                f"{wave['net_worth_identity_max_abs_error']:.2g}",
                f"{wave['asset_identity_max_abs_error']:.2g}",
            ]
        )
    lines += [
        "The balance-sheet identity, on every row of every implicate (largest "
        "absolute violation, in dollars):",
        "",
        _md_table(
            [
                "Wave",
                "Rows",
                "Families",
                "Implicates",
                "networth − (fin + nfin − debt)",
                "asset − (fin + nfin)",
            ],
            rows,
        ),
        "",
        f"Published figures: {checks['published_source']}",
        "",
    ]
    return lines


def render_markdown(report: Report, checks: dict | None = None) -> str:
    """The comparison document."""
    findings = report.findings
    baseline = report.arms[report.baseline_letter]
    lines = [
        "# SCF 2025 replication: 2022 versus 2025",
        "",
        "Generated by `imp replication` from the committed run directories; do "
        "not edit by hand. The rules are in "
        "`docs/scf-2025-replication-criteria.md`, committed before any method "
        "was fit on the 2025 wave.",
        "",
        "- **2022** is the paper's committed runs (SCF 2022 donor, 2022 dollars)."
        if report.baseline_letter == "A"
        else "- **2022** is the 2022 rerun in today's environment, because the "
        "paper's committed runs did not reproduce (see Diagnostics).",
        "- **2025** is the same code, environment, seeds, and receiver files "
        "with the SCF 2025 donor (2025 dollars). The receiver is unchanged: "
        "ASEC 2025 (minimal profile) and pooled ASEC 2023–2025 (scale).",
        "",
        "## Summary",
        "",
        f"- Primary ({_count(findings, True)}).",
        f"- Secondary ({_count(findings, False)}).",
        f"- Method rankings: {int(report.rankings['verdict'].eq('Replicated').sum())} "
        f"of {len(report.rankings)} table columns replicate, "
        f"{int(report.rankings['verdict'].eq('Partly replicated').sum())} partly, "
        f"{int(report.rankings['verdict'].eq('Not replicated').sum())} do not; the "
        f"best method is unchanged in {int(report.rankings['same_best'].sum())}.",
        "",
    ]
    not_clean = findings[findings["verdict"].ne(rules.REPLICATED)]
    if not_clean.empty:
        lines += ["Every finding replicated.", ""]
    else:
        lines += ["Findings that did not come out as replicated:", ""]
        for row in not_clean.itertuples():
            tag = " (primary)" if row.primary else ""
            lines.append(
                f"- **{row.id}**{tag}: {row.verdict}. {row.finding} "
                f"2022: {row.value_2022}. 2025: {row.value_2025}."
                + (f" {row.magnitude}." if row.magnitude else "")
            )
        lines.append("")

    lines += [
        "## Verdicts",
        "",
        '"Rule on 2022" applies the same rule to the 2022 runs. A claim that '
        "was within seed noise in 2022 can at best be inconclusive.",
        "",
        _md_table(
            [
                "ID",
                "Finding",
                "2022",
                "2025",
                "Rule on 2022",
                "Rule on 2025",
                "Magnitude (2025 effect in band)",
                "Verdict",
            ],
            [
                [
                    f"{row.id}{' (primary)' if row.primary else ''}",
                    row.finding,
                    row.value_2022,
                    row.value_2025,
                    row.status_2022,
                    row.status_2025,
                    row.magnitude or "–",
                    f"**{row.verdict}**",
                ]
                for row in findings.itertuples()
            ],
        ),
        "",
        "## Headline numbers",
        "",
        "Every SCF number the abstract, results, discussion, or conclusion "
        "cites, plus the tail statistics reported without a verdict. Dollar "
        "amounts in the 2022 column are in 2022 dollars; the next column "
        "re-bases them to 2025 dollars with the Fed's CPI-U-RS factor "
        f"(4789/4376 = {REBASE_2022_TO_2025:.4f}), the only basis on which a "
        "level can be read against 2025.",
        "",
    ]
    head_rows = []
    for row in report.headlines.itertuples():
        a_value = getattr(row, report.baseline_letter)
        head_rows.append(
            [
                row.number,
                row.where,
                _show(a_value, row.kind),
                _show(a_value * REBASE_2022_TO_2025, row.kind)
                if row.kind == "dollars"
                else "–",
                _show(row.B, row.kind),
            ]
        )
    lines += [
        _md_table(
            [
                "Number",
                "Where the paper cites it",
                "2022",
                "2022 in 2025 dollars",
                "2025",
            ],
            head_rows,
        ),
        "",
        "## Method rankings",
        "",
        "Spearman correlation between the 2022 and 2025 orderings of the eight "
        "displayed methods, per table column, ranked by the table's own notion "
        "of best. At least 0.8 replicates; 0.5 to 0.8 partly; below 0.5 does "
        "not.",
        "",
        _md_table(
            ["Table", "Column", "Spearman", "Best in 2022", "Best in 2025", "Verdict"],
            [
                [
                    TABLES[row.run][0],
                    row.column,
                    f"{row.spearman:.2f}",
                    METHOD_NAMES[row.best_baseline],
                    METHOD_NAMES[row.best_replication],
                    row.verdict,
                ]
                for row in report.rankings.itertuples()
            ],
        ),
        "",
        "## Full tables",
        "",
        "Mean (sd) over ten seeds, as in the paper's tables.",
        "",
    ]
    for run, (title, _) in TABLES.items():
        lines += [
            f"### {title}",
            "",
            f"2022 ({'2022 dollars' if run == 'wealth' else 'SCF 2022 donor'}):",
            "",
            _wave_table(baseline, run),
            "",
            f"2025 ({'2025 dollars' if run == 'wealth' else 'SCF 2025 donor'}):",
            "",
            _wave_table(report.arms["B"], run),
            "",
        ]

    lines += ["## Diagnostics", ""]
    lines += ["### Does today's environment reproduce the paper's 2022 runs?", ""]
    if report.reproduction.empty:
        lines += ["Arm D (the 2022 rerun) has not been run.", ""]
    else:
        check = report.reproduction
        identical = int(check["identical"].sum())
        worst = check.loc[check["gap_in_sd"].idxmax()]
        over = check[check["gap_in_sd"] > REPRODUCTION_TOLERANCE_SD]
        lines += [
            f"Of {len(check)} displayed table cells, {identical} match the "
            "committed runs exactly on all ten seeds. The largest gap between "
            f"a rerun mean and a committed mean is {worst['gap_in_sd']:.3f} "
            f"committed seed sds ({METHOD_NAMES[worst['method']]}, "
            f"{worst['column']}, {TABLES[worst['run']][0].lower()}). "
            + (
                "No cell exceeds the half-sd tolerance, so the committed 2022 "
                "runs stand as the baseline."
                if over.empty
                else f"{len(over)} cell(s) exceed the half-sd tolerance, so the "
                "rerun replaces the committed runs as the baseline."
            ),
            "",
        ]
        by_method = (
            check.groupby("method")
            .agg(
                cells=("identical", "size"),
                identical=("identical", "sum"),
                largest_gap_in_sd=("gap_in_sd", "max"),
                largest_seed_relative_difference=(
                    "max_seed_relative_difference",
                    "max",
                ),
            )
            .reindex([FLOOR, *DISPLAYED_METHODS])
        )
        lines += [
            _md_table(
                [
                    "Method",
                    "Cells",
                    "Identical",
                    "Largest gap (sd)",
                    "Largest per-seed relative difference",
                ],
                [
                    [
                        METHOD_NAMES[str(method)],
                        str(int(row.cells)),
                        str(int(row.identical)),
                        f"{row.largest_gap_in_sd:.3f}",
                        f"{row.largest_seed_relative_difference:.2g}",
                    ]
                    for method, row in by_method.iterrows()
                ],
            ),
            "",
        ]

    lines += ["### The same rules on the diagnostic arms", ""]
    if report.diagnostics.shape[1] <= 1:
        lines += ["Neither diagnostic arm has been run.", ""]
    else:
        lines += [
            "Arm C is the 2022 wave in 2025 dollars (the file the Fed now "
            "serves): it separates the change of dollar basis from the change "
            "of wave. Arm D is the 2022 rerun. Neither carries a verdict.",
            "",
        ]
        headers = ["ID", "2022 (paper)", "2025"]
        if "status_C" in report.diagnostics:
            headers += ["2022 in 2025 dollars (C)", "C numbers"]
        if "status_D" in report.diagnostics:
            headers += ["2022 rerun (D)"]
        rows = []
        merged = report.diagnostics.merge(findings, on="id")
        for row in merged.itertuples():
            cells = [row.id, row.status_2022, row.status_2025]
            if "status_C" in report.diagnostics:
                cells += [row.status_C, row.value_C]
            if "status_D" in report.diagnostics:
                cells += [row.status_D]
            rows.append(cells)
        lines += [_md_table(headers, rows), ""]

    lines += _checks_section(checks)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# LaTeX
# ---------------------------------------------------------------------------

_TEX_ESCAPES = {
    "&": r"\&",
    "%": r"\%",
    "_": r"\_",
    "#": r"\#",
    "−": "$-$",
    "–": "--",
}


def _tex(text: str) -> str:
    """Escape report text for LaTeX."""
    for raw, escaped in _TEX_ESCAPES.items():
        text = text.replace(raw, escaped)
    return text


def render_verdict_table(report: Report) -> str:
    """The verdict table the appendix inputs (two tabulars, one per task)."""
    blocks = [
        "% Generated by `imp replication` -- do not edit by hand. Pre-registered "
        "rules of docs/scf-2025-replication-criteria.md applied to the committed "
        "2022 and 2025 runs.",
    ]
    for prefix, caption in (("WG", "within-SCF"), ("F", "population view")):
        subset = [j for j in report.judgements if j.finding.key[0] in prefix]
        blocks += [
            f"% --- {caption} ---",
            r"\begin{tabular}{@{}lp{0.21\textwidth}p{0.27\textwidth}"
            r"p{0.27\textwidth}p{0.13\textwidth}@{}}",
            r"\hline",
            r"ID & Finding & 2022 & 2025 & Verdict \\",
            r"\hline",
        ]
        for j in subset:
            key = j.finding.key + (r"$^{\ast}$" if j.finding.primary else "")
            blocks.append(
                " & ".join(
                    [
                        key,
                        _tex(j.finding.claim),
                        _tex(j.baseline.text),
                        _tex(j.replication.text),
                        _tex(j.verdict),
                    ]
                )
                + r" \\"
            )
        blocks += [r"\hline", r"\end{tabular}", ""]
        if prefix == "WG":
            blocks += [r"\par\medskip", ""]
    return "\n".join(blocks)


def number_macros(report: Report, checks: dict | None = None) -> dict[str, str]:
    """Macro key -> text for every number the appendix prose may cite.

    Keys: ``<headline key>:2022`` and ``:2025`` (and ``:2022r`` for a dollar
    amount re-based to 2025 dollars); ``<finding id>:verdict``,
    ``:status2022``, ``:status2025``; ``rank:<n replicated>`` style counts; and
    the published-table check when its manifest exists.
    """
    macros: dict[str, str] = {}
    for row in report.headlines.itertuples():
        a_value = getattr(row, report.baseline_letter)
        macros[f"{row.key}:2022"] = _show(a_value, row.kind)
        macros[f"{row.key}:2025"] = _show(row.B, row.kind)
        if row.kind == "dollars":
            macros[f"{row.key}:2022r"] = _show(a_value * REBASE_2022_TO_2025, row.kind)
    for row in report.findings.itertuples():
        macros[f"{row.id}:verdict"] = row.verdict.lower()
        macros[f"{row.id}:status2022"] = row.status_2022
        macros[f"{row.id}:status2025"] = row.status_2025
    for primary, tag in ((True, "primary"), (False, "secondary")):
        subset = report.findings[report.findings["primary"].eq(primary)]
        macros[f"count:{tag}"] = str(len(subset))
        macros[f"count:{tag}:replicated"] = str(
            int(subset["verdict"].eq(rules.REPLICATED).sum())
        )
        macros[f"count:{tag}:direction"] = str(
            int(subset["verdict"].str.startswith(rules.DIRECTION_ONLY).sum())
        )
        macros[f"count:{tag}:inconclusive"] = str(
            int(subset["verdict"].eq(rules.VERDICT_INCONCLUSIVE).sum())
        )
        macros[f"count:{tag}:failed"] = str(
            int(subset["verdict"].eq(rules.NOT_REPLICATED).sum())
        )
    rankings = report.rankings
    macros["rank:columns"] = str(len(rankings))
    macros["rank:replicated"] = str(int(rankings["verdict"].eq("Replicated").sum()))
    macros["rank:partly"] = str(int(rankings["verdict"].eq("Partly replicated").sum()))
    macros["rank:failed"] = str(int(rankings["verdict"].eq("Not replicated").sum()))
    macros["rank:samebest"] = str(int(rankings["same_best"].sum()))
    macros["rank:min"] = f"{rankings['spearman'].min():.2f}"
    for row in rankings.itertuples():
        macros[f"rank:{row.run}:{row.metric}"] = f"{row.spearman:.2f}"
    if not report.reproduction.empty:
        check = report.reproduction
        macros["repro:cells"] = str(len(check))
        macros["repro:identical"] = str(int(check["identical"].sum()))
        macros["repro:maxgap"] = f"{check['gap_in_sd'].max():.2f}"
    if checks is not None:
        for key, gap in checks["published_gaps"].items():
            scale = 1e6 if key == "weight_total" else 1e3
            for wave, tag in (("calibration", "2022"), ("replication", "2025")):
                macros[f"check:{key}:public:{tag}"] = (
                    f"{gap[f'public_{wave}'] / scale:,.1f}"
                )
                macros[f"check:{key}:published:{tag}"] = (
                    f"{gap[f'published_{wave}'] / scale:,.1f}"
                )
                macros[f"check:{key}:gap:{tag}"] = (
                    f"{100 * abs(gap[f'gap_{wave}']):.2f}"
                )
        for year in ("2022", "2025"):
            macros[f"check:families:{year}"] = f"{checks['waves'][year]['families']:,}"
            macros[f"check:rows:{year}"] = f"{checks['waves'][year]['rows']:,}"
    return macros


def render_number_macros(macros: dict[str, str]) -> str:
    r"""The macro file: ``\rep{key}`` expands to the generated number."""
    lines = [
        "% Generated by `imp replication` -- do not edit by hand. Every number "
        "the SCF 2025 appendix cites, as \\rep{key}.",
        r"\providecommand{\rep}[1]{\ifcsname rep:#1\endcsname"
        r"\csname rep:#1\endcsname\else\textbf{[missing: #1]}\fi}",
    ]
    for key in sorted(macros):
        lines.append(
            rf"\expandafter\def\csname rep:{key}\endcsname{{{_tex(macros[key])}}}"
        )
    return "\n".join(lines) + "\n"


def make_replication(
    runs_dir: Path = Path("runs"),
    docs_out: Path = Path("docs/scf-2025-replication"),
    tables_out: Path = Path("paper/tables"),
) -> int:
    """Write the replication report and the appendix's generated inputs.

    Returns:
        Process exit code (``0`` on success, ``1`` if a required arm is
        missing).
    """
    try:
        report = build_report(runs_dir)
    except FileNotFoundError as reason:
        print(f"imp replication: {reason}")
        return 1
    checks_path = runs_dir / "scf-population-checks" / "manifest.json"
    checks = json.loads(checks_path.read_text()) if checks_path.exists() else None

    docs_out.mkdir(parents=True, exist_ok=True)
    tables_out.mkdir(parents=True, exist_ok=True)
    (docs_out / "comparison.md").write_text(render_markdown(report, checks))
    report.findings.to_csv(docs_out / "findings.csv", index=False)
    report.headlines.to_csv(docs_out / "headline_numbers.csv", index=False)
    report.rankings.to_csv(docs_out / "rankings.csv", index=False)
    if not report.reproduction.empty:
        report.reproduction.to_csv(docs_out / "reproduction_check.csv", index=False)
    (tables_out / "scf2025_replication.tex").write_text(render_verdict_table(report))
    (tables_out / "scf2025_numbers.tex").write_text(
        render_number_macros(number_macros(report, checks))
    )

    print(f"imp replication -- baseline arm {report.baseline_letter}")
    for verdict, count in report.findings["verdict"].value_counts().items():
        print(f"  {count:2d} {verdict}")
    print(f"  wrote {docs_out / 'comparison.md'}")
    print(f"  wrote {tables_out / 'scf2025_replication.tex'}")
    print(f"  wrote {tables_out / 'scf2025_numbers.tex'}")
    return 0


def make_population_checks(out: Path = Path("runs/scf-population-checks")) -> int:
    """Run the SCF population checks on both waves; write ``manifest.json``.

    Returns:
        Process exit code (``0`` if every check holds).
    """
    from imputation_paper.data import scf_checks

    waves = {str(year): scf_checks.population_summary(year) for year in (2022, 2025)}
    gaps = scf_checks.published_gaps(waves["2022"], waves["2025"])
    manifest = {
        "checks": "scf_population",
        "published_source": scf_checks.PUBLISHED_SOURCE,
        "published_dollar_year": scf_checks.PUBLISHED_DOLLAR_YEAR,
        "families_interviewed": {
            str(year): count
            for year, count in scf_checks.PUBLISHED_FAMILIES_INTERVIEWED.items()
        },
        "waves": waves,
        "published_gaps": gaps,
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    ok = all(bool(gap["within_tolerance"]) for gap in gaps.values())
    for year, wave in waves.items():
        print(
            f"  SCF {year}: {wave['rows']:,} rows, {wave['families']:,} families, "
            f"{wave['weight_total'] / 1e6:.2f}M represented; identity error "
            f"{wave['net_worth_identity_max_abs_error']:.2g}"
        )
    for key, gap in gaps.items():
        print(
            f"  {key}: 2022 gap {100 * gap['gap_calibration']:+.3f}%, 2025 gap "
            f"{100 * gap['gap_replication']:+.3f}%, tolerance "
            f"{100 * gap['tolerance']:.2f}% -> "
            f"{'ok' if gap['within_tolerance'] else 'OUTSIDE'}"
        )
    print(f"  wrote {out / 'manifest.json'}")
    return 0 if ok else 1
