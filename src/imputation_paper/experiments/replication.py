"""The pre-registered SCF 2025 replication rules, as code.

``docs/scf-2025-replication-criteria.md`` fixed, before any method was fit on
the 2025 wave, which of the paper's SCF findings count as replicated and by
what rule. This module is those rules and nothing else: each
:class:`Finding` evaluates one claim on one *arm* (a wave's three run
directories), and :func:`judge` turns the baseline arm's and the replication
arm's evaluations into a verdict. Nothing here knows which wave it is looking
at, so the same code scores the 2022 runs, the 2025 runs, and the two
diagnostic arms, and a verdict cannot be adjusted without changing a rule that
the criteria file states and ``tests/test_replication.py`` pins.

Vocabulary (from the criteria file):

* A **difference** claim says method B is worse (higher) than method A on a
  metric. A **tie** claim says they are indistinguishable. A **level** claim
  carries its own threshold.
* Each rule returns ``"met"``, ``"inconclusive"``, or ``"failed"``.
* In the population-view runs the noise unit is the **floor sd**: the
  ten-seed sample sd of the ``scf_sample_reference`` row for the same metric.
* Where the paper states an effect size, its **magnitude** holds if the
  replication's effect is within a factor of two of the baseline's.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = [
    "ABLATIONS",
    "CANDIDATE",
    "DISPLAYED_METHODS",
    "FAILED",
    "FINDINGS",
    "FLOOR",
    "INCONCLUSIVE",
    "MET",
    "Arm",
    "Contrast",
    "Evaluation",
    "Finding",
    "Judgement",
    "contrast",
    "difference_status",
    "format_number",
    "judge",
    "load_arm",
    "ranking_agreement",
    "tie_status",
]

MET = "met"
INCONCLUSIVE = "inconclusive"
FAILED = "failed"

CANDIDATE = "populace_fit"
UNWEIGHTED = "populace_fit_unweighted"
UNCHAINED = "populace_fit_unchained"
PLAIN_FOREST = "plain_qrf"
OLS = "microimpute_ols"
QUANTREG = "microimpute_quantreg"
HOT_DECK = "statmatch_hotdeck"
MARGINAL = "weighted_marginal"
FLOOR = "scf_sample_reference"

#: The candidate's three ablations, in the paper's order.
ABLATIONS: tuple[str, ...] = (UNWEIGHTED, UNCHAINED, PLAIN_FOREST)

#: The eight non-floor rows the paper's tables display (``microimpute_qrf`` is
#: the same estimator as ``plain_qrf`` and is shown once).
DISPLAYED_METHODS: tuple[str, ...] = (
    CANDIDATE,
    UNWEIGHTED,
    UNCHAINED,
    PLAIN_FOREST,
    OLS,
    QUANTREG,
    HOT_DECK,
    MARGINAL,
)

#: Seeds on which a contrast must agree in sign to count as seed-consistent.
MIN_CONSISTENT_SEEDS = 8

#: Ratio-of-means band inside which a within-SCF tie holds.
TIE_RATIO_BAND = (0.8, 1.25)

#: Run keys -> (run-directory stem, long-file name).
_RUNS: dict[str, tuple[str, str]] = {
    "wealth": ("scf-wealth", "metrics_long.csv"),
    "min": ("scf-to-cps-harness", "harness_long.csv"),
    "scale": ("scf-to-cps-harness-scale", "harness_long.csv"),
}

#: Profile key -> the word the report uses for it.
PROFILE_NAMES: dict[str, str] = {"min": "minimal", "scale": "scale"}


@dataclass(frozen=True)
class Arm:
    """One wave's three runs, in the long format the sweep and harness write.

    Attributes:
        label: What the report calls the arm (e.g. ``"2025"``).
        wealth: ``scf-wealth*/metrics_long.csv``
            (method, seed, target, metric, value).
        minimal: ``scf-to-cps-harness*/harness_long.csv``
            (method, seed, view, metric, value).
        scale: ``scf-to-cps-harness-scale*/harness_long.csv``.
    """

    label: str
    wealth: pd.DataFrame
    minimal: pd.DataFrame
    scale: pd.DataFrame

    def frame(self, run: str) -> pd.DataFrame:
        """The long frame for run key ``"wealth"``, ``"min"``, or ``"scale"``."""
        return {"wealth": self.wealth, "min": self.minimal, "scale": self.scale}[run]

    def values(
        self, run: str, method: str, metric: str, target: str | None = None
    ) -> pd.Series:
        """Per-seed values of one metric for one method, indexed by seed."""
        frame = self.frame(run)
        mask = frame["method"].eq(method) & frame["metric"].eq(metric)
        if run == "wealth":
            if target is None:
                raise ValueError("The within-SCF run needs a target.")
            mask &= frame["target"].eq(target)
        series = frame.loc[mask].set_index("seed")["value"].sort_index()
        if series.empty:
            raise KeyError(
                f"Arm {self.label!r} has no {metric!r} rows for {method!r} in "
                f"run {run!r} (target {target!r})."
            )
        return series

    def mean(
        self, run: str, method: str, metric: str, target: str | None = None
    ) -> float:
        """Seed mean of one metric for one method."""
        return float(self.values(run, method, metric, target).mean())

    def methods(self, run: str) -> list[str]:
        """Every method present in a run, floor row included."""
        return sorted(self.frame(run)["method"].unique())

    def floor_sd(self, run: str, metric: str) -> float:
        """The floor row's seed sd (ddof = 1) for a population-view metric."""
        if run == "wealth":
            raise ValueError("The within-SCF run has no floor row.")
        return float(self.values(run, FLOOR, metric).std(ddof=1))


def load_arm(runs_dir: Path, suffix: str, label: str) -> Arm:
    """Read one arm's three long files from ``runs_dir``.

    Args:
        runs_dir: The runs root.
        suffix: Appended to each run-directory stem (``""`` for the paper's
            2022 runs, ``"-2025"`` for the replication).
        label: The arm's label in the report.

    Raises:
        FileNotFoundError: Naming every missing long file.
    """
    paths = {
        key: runs_dir / f"{stem}{suffix}" / filename
        for key, (stem, filename) in _RUNS.items()
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Arm {label!r} is missing run file(s): {missing}.")
    frames = {key: pd.read_csv(path) for key, path in paths.items()}
    return Arm(label, frames["wealth"], frames["min"], frames["scale"])


@dataclass(frozen=True)
class Contrast:
    """Method B against method A on one metric, paired by seed.

    Attributes:
        mean_a: Seed mean for method A.
        mean_b: Seed mean for method B.
        dbar: Mean of the paired differences ``B - A``.
        k_pos: Seeds on which B is higher than A.
        k_neg: Seeds on which B is lower than A.
        n: Seeds compared.
        floor_sd: The floor sd for the metric (population-view runs), else
            ``None``.
    """

    mean_a: float
    mean_b: float
    dbar: float
    k_pos: int
    k_neg: int
    n: int
    floor_sd: float | None = None

    @property
    def ratio(self) -> float:
        """Ratio of means, B over A."""
        return self.mean_b / self.mean_a

    @property
    def gap(self) -> float:
        """``|dbar|`` in floor sds (population-view runs only)."""
        if self.floor_sd is None:
            raise ValueError("No floor sd on a within-SCF contrast.")
        return abs(self.dbar) / self.floor_sd


def contrast(
    arm: Arm, run: str, metric: str, a: str, b: str, target: str | None = None
) -> Contrast:
    """Pair method ``b`` against method ``a`` by seed on one metric."""
    series_a = arm.values(run, a, metric, target)
    series_b = arm.values(run, b, metric, target)
    if not series_a.index.equals(series_b.index):
        raise ValueError(
            f"Arm {arm.label!r}: {a!r} and {b!r} do not share seeds on {metric!r}."
        )
    difference = series_b - series_a
    return Contrast(
        mean_a=float(series_a.mean()),
        mean_b=float(series_b.mean()),
        dbar=float(difference.mean()),
        k_pos=int((difference > 0).sum()),
        k_neg=int((difference < 0).sum()),
        n=int(len(difference)),
        floor_sd=None if run == "wealth" else arm.floor_sd(run, metric),
    )


def difference_status(c: Contrast) -> str:
    """The difference rule: is B worse (higher) than A?

    Within-SCF: met if ``dbar > 0`` on at least eight seeds' agreement; failed
    if ``dbar <= 0``. Population view: met if ``dbar`` exceeds two floor sds
    with at least eight seeds agreeing; failed if ``dbar`` is at most one
    floor sd (which covers the wrong sign). Otherwise inconclusive.
    """
    consistent = c.k_pos >= MIN_CONSISTENT_SEEDS
    if c.floor_sd is None:
        if c.dbar > 0 and consistent:
            return MET
        return FAILED if c.dbar <= 0 else INCONCLUSIVE
    if c.dbar > 2 * c.floor_sd and consistent:
        return MET
    return FAILED if c.dbar <= c.floor_sd else INCONCLUSIVE


def tie_status(c: Contrast, *, ratio_band: bool = True) -> str:
    """The tie rule: is B indistinguishable from A?

    Population view: met within one floor sd, failed beyond two. Within-SCF:
    met if neither method wins on eight or more seeds and the ratio of means
    is inside :data:`TIE_RATIO_BAND`; failed if one wins on eight or more and
    the ratio is outside it. With ``ratio_band=False`` (values near zero,
    where a ratio is meaningless) the seeds alone decide.
    """
    if c.floor_sd is not None:
        gap = abs(c.dbar)
        if gap <= c.floor_sd:
            return MET
        return FAILED if gap > 2 * c.floor_sd else INCONCLUSIVE
    lopsided = max(c.k_pos, c.k_neg) >= MIN_CONSISTENT_SEEDS
    if not ratio_band:
        return FAILED if lopsided else MET
    low, high = TIE_RATIO_BAND
    in_band = low <= c.ratio <= high
    if not lopsided and in_band:
        return MET
    return FAILED if lopsided and not in_band else INCONCLUSIVE


def _all_of(statuses: Sequence[str]) -> str:
    """Combine parts that must all hold: any failure fails, else any doubt doubts."""
    if FAILED in statuses:
        return FAILED
    return MET if all(status == MET for status in statuses) else INCONCLUSIVE


def _level(condition: bool) -> str:
    """A level claim is binary: its threshold holds or it does not."""
    return MET if condition else FAILED


def format_number(value: float) -> str:
    """Format a number the way the paper's tables do."""
    if not np.isfinite(value):
        return "--"
    magnitude = abs(value)
    if magnitude >= 1000:
        return f"{value:,.0f}"
    if magnitude >= 1:
        return f"{value:,.2f}"
    return f"{value:.3f}"


@dataclass(frozen=True)
class Evaluation:
    """One finding's rule applied to one arm.

    Attributes:
        status: ``"met"``, ``"inconclusive"``, or ``"failed"``.
        text: The numbers behind the status, for the comparison table.
        effects: The stated effect sizes (one per profile where the claim
            spans both), on the scale the magnitude band is defined on; empty
            when the claim carries no magnitude.
        parts: For a claim made of several rules that must all hold, each
            rule's status; empty for a single rule.
    """

    status: str
    text: str
    effects: tuple[float, ...] = ()
    parts: tuple[str, ...] = ()


@dataclass(frozen=True)
class Finding:
    """One pre-registered claim.

    Attributes:
        key: The ID in the criteria file (``"W1"``, ``"F2b"``, ...).
        primary: Whether the abstract or conclusion makes the claim.
        claim: The paper's claim, in words.
        rule: The rule in brief.
        evaluate: Applies the rule to an arm.
        effect_offset: Added to an effect and its band for display: ``1.0``
            when the effect is a ratio minus one, ``0.0`` otherwise.
        effect_name: What the displayed effect is called.
    """

    key: str
    primary: bool
    claim: str
    rule: str
    evaluate: Callable[[Arm], Evaluation]
    effect_offset: float = 0.0
    effect_name: str = ""


# ---------------------------------------------------------------------------
# Within-SCF wealth task
# ---------------------------------------------------------------------------


def _wealth_difference(
    target: str, metric: str, other: str
) -> Callable[[Arm], Evaluation]:
    """Difference of ``other`` against the candidate, with its ratio effect."""

    def evaluate(arm: Arm) -> Evaluation:
        c = contrast(arm, "wealth", metric, CANDIDATE, other, target)
        return Evaluation(
            difference_status(c),
            f"{format_number(c.mean_b)} vs {format_number(c.mean_a)}; ratio {c.ratio:.2f}; "
            f"worse on {c.k_pos}/{c.n} seeds",
            (c.ratio - 1.0,),
        )

    return evaluate


def _w3a(arm: Arm) -> Evaluation:
    deltas = {
        key: contrast(arm, "wealth", "wasserstein1", CANDIDATE, key, "networth").dbar
        for key in ABLATIONS
    }
    largest = max(deltas, key=lambda key: deltas[key])
    return Evaluation(
        _level(largest == UNWEIGHTED and deltas[UNWEIGHTED] > 0),
        f"deltas: unweighted {deltas[UNWEIGHTED]:+,.0f}; ungated forest "
        f"{deltas[PLAIN_FOREST]:+,.0f}; unchained {deltas[UNCHAINED]:+,.0f}",
    )


def _w4a(arm: Arm) -> Evaluation:
    c = contrast(arm, "wealth", "wasserstein1", CANDIDATE, HOT_DECK, "debt")
    return Evaluation(
        tie_status(c),
        f"{format_number(c.mean_b)} vs {format_number(c.mean_a)}; ratio {c.ratio:.2f}; hot deck "
        f"worse on {c.k_pos}/{c.n} seeds, better on {c.k_neg}/{c.n}",
    )


def _w5(arm: Arm) -> Evaluation:
    parts, ratios = [], []
    for target in ("debt", "networth"):
        means = {
            method: arm.mean("wealth", method, "wasserstein1", target)
            for method in arm.methods("wealth")
        }
        ratio = means[OLS] / means[CANDIDATE]
        ratios.append(ratio)
        parts.append(max(means, key=lambda method: means[method]) == OLS and ratio >= 5)
    return Evaluation(
        _level(all(parts)),
        f"OLS W1 is {ratios[0]:.1f}x (debt) and {ratios[1]:.1f}x (net worth) "
        "the candidate's"
        + ("; highest on both" if all(parts) else "; not highest by 5x on both"),
    )


def _w6(arm: Arm) -> Evaluation:
    def zero_share(method: str) -> float:
        return arm.mean("wealth", method, "zero_share_error", "debt")

    forests = [zero_share(m) for m in (CANDIDATE, *ABLATIONS)]
    quantreg = zero_share(QUANTREG)
    return Evaluation(
        _level(quantreg >= 0.10 and max(forests) <= 0.03),
        f"quantile regression {quantreg:.3f}; forest-based "
        f"{min(forests):.3f} to {max(forests):.3f}",
    )


def _w7(arm: Arm) -> Evaluation:
    parts, texts = [], []
    for target in ("debt", "networth"):
        means = {
            method: arm.mean("wealth", method, "pinball_loss", target)
            for method in arm.methods("wealth")
        }
        best = min(means, key=lambda method: means[method])
        parts.append(best == MARGINAL)
        texts.append(
            f"{target}: marginal {format_number(means[MARGINAL])}, candidate "
            f"{format_number(means[CANDIDATE])}, lowest: "
            f"{'marginal draw' if best == MARGINAL else best}"
        )
    return Evaluation(_level(all(parts)), "; ".join(texts))


def _g1(arm: Arm) -> Evaluation:
    c = contrast(arm, "wealth", "zero_share_error", CANDIDATE, PLAIN_FOREST, "debt")
    return Evaluation(
        tie_status(c, ratio_band=False),
        f"ungated forest {c.mean_b:.3f} vs candidate {c.mean_a:.3f}; forest "
        f"worse on {c.k_pos}/{c.n} seeds, better on {c.k_neg}/{c.n}",
    )


# ---------------------------------------------------------------------------
# Population view
# ---------------------------------------------------------------------------

_ENERGY = "energy_distance"
_COVERAGE = "prdc_coverage"
_AUC = "c2st_auc"
_TAIL_W1 = "w1_over_sd.networth"
_Q99 = "q99_ratio.networth"

_SHORT = {
    _ENERGY: "energy",
    _COVERAGE: "coverage",
    _AUC: "AUC",
    _TAIL_W1: "W1/sd",
    _Q99: "q99 ratio",
}


def _ties(
    pairs: Sequence[tuple[str, str, str, str]],
) -> Callable[[Arm], Evaluation]:
    """Several population-view ties that must all hold.

    Args:
        pairs: ``(profile, metric, a, b)`` for each tie.
    """

    def evaluate(arm: Arm) -> Evaluation:
        contrasts = [
            (profile, metric, contrast(arm, profile, metric, a, b))
            for profile, metric, a, b in pairs
        ]
        statuses = [tie_status(c) for _, _, c in contrasts]
        profile, metric, widest = max(contrasts, key=lambda item: item[2].gap)
        if len(contrasts) == 1:
            text = (
                f"{format_number(widest.mean_b)} vs {format_number(widest.mean_a)}; gap "
                f"{widest.gap:.2f} floor sd"
            )
        else:
            text = (
                f"largest gap {widest.gap:.2f} floor sd ({_SHORT[metric]}, "
                f"{PROFILE_NAMES[profile]}: {format_number(widest.mean_b)} vs "
                f"{format_number(widest.mean_a)}); {statuses.count(MET)}/{len(statuses)} "
                "ties met"
            )
        return Evaluation(_all_of(statuses), text, parts=tuple(statuses))

    return evaluate


def _harness_difference(
    profile: str, metric: str, other: str, *, level_at_least: float | None = None
) -> Callable[[Arm], Evaluation]:
    """Difference of ``other`` against the candidate; the effect is ``dbar``."""

    def evaluate(arm: Arm) -> Evaluation:
        c = contrast(arm, profile, metric, CANDIDATE, other)
        statuses = [difference_status(c)]
        text = (
            f"{format_number(c.mean_b)} vs {format_number(c.mean_a)}; gap {c.dbar:+.3f} "
            f"({c.gap:.2f} floor sd); higher on {c.k_pos}/{c.n} seeds"
        )
        if level_at_least is not None:
            statuses.append(_level(c.mean_b >= level_at_least))
        return Evaluation(_all_of(statuses), text, (c.dbar,))

    return evaluate


def _f0(arm: Arm) -> Evaluation:
    parts, texts = [], []
    for profile in ("min", "scale"):
        others = [m for m in arm.methods(profile) if m != FLOOR]
        auc = arm.mean(profile, FLOOR, _AUC)
        energy = arm.mean(profile, FLOOR, _ENERGY)
        coverage = arm.mean(profile, FLOOR, _COVERAGE)
        q99 = arm.mean(profile, FLOOR, _Q99)
        parts += [
            abs(auc - 0.5) <= 2 * arm.floor_sd(profile, _AUC),
            energy < min(arm.mean(profile, m, _ENERGY) for m in others) / 5,
            coverage >= max(arm.mean(profile, m, _COVERAGE) for m in others),
            abs(q99 - 1.0) <= 2 * arm.floor_sd(profile, _Q99),
        ]
        texts.append(
            f"{PROFILE_NAMES[profile]}: AUC {auc:.3f}, energy {energy:.3f}, "
            f"coverage {coverage:.3f}, q99 ratio {q99:.2f}"
        )
    return Evaluation(_level(all(parts)), "; ".join(texts))


def _f1b(arm: Arm) -> Evaluation:
    statuses, texts, effects = [], [], []
    for profile in ("min", "scale"):
        c = contrast(arm, profile, _Q99, CANDIDATE, UNWEIGHTED)
        statuses += [difference_status(c), _level(c.mean_b > 1.5)]
        effects.append(c.mean_b - 1.0)
        texts.append(
            f"{PROFILE_NAMES[profile]}: {c.mean_b:.2f} vs candidate "
            f"{c.mean_a:.2f} ({c.gap:.1f} floor sd; {c.k_pos}/{c.n} seeds)"
        )
    return Evaluation(_all_of(statuses), "; ".join(texts), tuple(effects))


def _f1c(arm: Arm) -> Evaluation:
    statuses, texts, effects = [], [], []
    for profile in ("min", "scale"):
        c = contrast(arm, profile, _TAIL_W1, CANDIDATE, UNWEIGHTED)
        statuses.append(difference_status(c))
        effects.append(c.ratio - 1.0)
        texts.append(
            f"{PROFILE_NAMES[profile]}: {c.mean_b:.3f} vs {c.mean_a:.3f}, ratio "
            f"{c.ratio:.2f} ({c.k_pos}/{c.n} seeds)"
        )
    return Evaluation(_all_of(statuses), "; ".join(texts), tuple(effects))


def _f3b(arm: Arm) -> Evaluation:
    minimal = arm.mean("min", MARGINAL, _AUC)
    scale = arm.mean("scale", MARGINAL, _AUC)
    return Evaluation(
        _level(scale - minimal > 0.060),
        f"{minimal:.3f} (minimal) to {scale:.3f} (scale): rise {scale - minimal:+.3f}",
    )


def _f3c(arm: Arm) -> Evaluation:
    parts, texts = [], []
    for profile in ("min", "scale"):
        means = {
            method: arm.mean(profile, method, _AUC)
            for method in (CANDIDATE, OLS, QUANTREG, HOT_DECK, MARGINAL)
        }
        best = min(means, key=lambda method: means[method])
        parts.append(best == CANDIDATE)
        runner_up = min(v for k, v in means.items() if k != CANDIDATE)
        texts.append(
            f"{PROFILE_NAMES[profile]}: candidate {means[CANDIDATE]:.3f}, lowest "
            f"baseline {runner_up:.3f}"
        )
    return Evaluation(_level(all(parts)), "; ".join(texts))


def _f4c(arm: Arm) -> Evaluation:
    c = contrast(arm, "scale", _TAIL_W1, CANDIDATE, HOT_DECK)
    return Evaluation(
        _level(c.dbar <= arm.floor_sd("scale", _TAIL_W1)),
        f"hot deck {c.mean_b:.3f} vs candidate {c.mean_a:.3f} ({c.dbar:+.4f}, "
        f"{c.gap:.2f} floor sd)",
    )


def _f5a(arm: Arm) -> Evaluation:
    parts, texts = [], []
    for profile in ("min", "scale"):
        values = arm.values(profile, CANDIDATE, _Q99)
        below = int((values < 1).sum())
        mean = float(values.mean())
        parts.append(below >= MIN_CONSISTENT_SEEDS and 0.60 <= mean <= 0.90)
        texts.append(
            f"{PROFILE_NAMES[profile]}: {mean:.2f}, below 1 on {below}/{len(values)} "
            "seeds"
        )
    return Evaluation(_level(all(parts)), "; ".join(texts))


def _f5b(arm: Arm) -> Evaluation:
    statuses, texts = [], []
    for profile in ("min", "scale"):
        hot_deck = contrast(arm, profile, _Q99, CANDIDATE, HOT_DECK)
        marginal = contrast(arm, profile, _Q99, FLOOR, MARGINAL)
        statuses += [tie_status(hot_deck), tie_status(marginal)]
        texts.append(
            f"{PROFILE_NAMES[profile]}: hot deck {hot_deck.mean_b:.2f} vs candidate "
            f"{hot_deck.mean_a:.2f} ({hot_deck.gap:.2f} floor sd); marginal "
            f"{marginal.mean_b:.2f} vs floor {marginal.mean_a:.2f} "
            f"({marginal.gap:.2f} floor sd)"
        )
    return Evaluation(_all_of(statuses), "; ".join(texts), parts=tuple(statuses))


#: Every pre-registered finding, in the criteria file's order.
FINDINGS: tuple[Finding, ...] = (
    Finding(
        "W1",
        True,
        "Fit unweighted, the same estimator is six times worse on net-worth W1.",
        "Difference, unweighted vs candidate, net-worth W1; ratio within a "
        "factor of two (on ratio minus one).",
        _wealth_difference("networth", "wasserstein1", UNWEIGHTED),
        1.0,
        "ratio",
    ),
    Finding(
        "W2",
        False,
        "Unweighted is 2.5 times worse on debt W1.",
        "Difference, unweighted vs candidate, debt W1; ratio within a factor of two.",
        _wealth_difference("debt", "wasserstein1", UNWEIGHTED),
        1.0,
        "ratio",
    ),
    Finding(
        "W3a",
        False,
        "Weighting is the largest of the three ablation effects on net-worth W1.",
        "Level: the unweighted ablation's mean paired delta is the largest of "
        "the three and positive.",
        _w3a,
    ),
    Finding(
        "W3b",
        False,
        "The ungated, unchained forest is worse than the candidate on "
        "net-worth W1 (1.7 times).",
        "Difference, plain_qrf vs candidate, net-worth W1; ratio within a "
        "factor of two.",
        _wealth_difference("networth", "wasserstein1", PLAIN_FOREST),
        1.0,
        "ratio",
    ),
    Finding(
        "W3c",
        False,
        "Removing chaining raises net-worth W1 by 17 percent.",
        "Difference, unchained vs candidate, net-worth W1; ratio within a "
        "factor of two.",
        _wealth_difference("networth", "wasserstein1", UNCHAINED),
        1.0,
        "ratio",
    ),
    Finding(
        "W4a",
        False,
        "Hot deck nearly ties the candidate on debt W1.",
        "Tie, hot deck vs candidate, debt W1.",
        _w4a,
    ),
    Finding(
        "W4b",
        False,
        "Hot deck is 3.7 times worse on net-worth W1.",
        "Difference, hot deck vs candidate, net-worth W1; ratio within a "
        "factor of two.",
        _wealth_difference("networth", "wasserstein1", HOT_DECK),
        1.0,
        "ratio",
    ),
    Finding(
        "W5",
        False,
        "OLS is catastrophic on both targets.",
        "Level: OLS has the highest mean W1 on both targets, at least five "
        "times the candidate's.",
        _w5,
    ),
    Finding(
        "W6",
        False,
        "Linear quantile regression leaves 15 percent of the debt zero mass "
        "unaccounted for, against at most 2 percent for forest-based methods.",
        "Level: quantile regression's debt zero-share error is at least 0.10; "
        "the candidate, its ablations, and plain_qrf are each at most 0.03.",
        _w6,
    ),
    Finding(
        "W7",
        False,
        "The unconditional weighted marginal draw has the best pinball loss "
        "on both targets.",
        "Level: weighted_marginal has the lowest mean pinball loss on debt "
        "and on net worth.",
        _w7,
    ),
    Finding(
        "G1",
        False,
        "Gating shows no advantage on SCF debt zero share.",
        "Tie on seeds alone, plain_qrf vs candidate, debt zero-share error.",
        _g1,
    ),
    Finding(
        "F0",
        False,
        "The sampling floor behaves as designed.",
        "Level, both profiles: floor AUC within 2 floor sd of 0.5; energy "
        "below a fifth of every method's; coverage at least every method's; "
        "q99 ratio within 2 floor sd of 1.",
        _f0,
    ),
    Finding(
        "F1a",
        True,
        "The unweighted ablation ties the candidate on energy distance, "
        "coverage, and C2ST AUC in both profiles.",
        "Six ties, unweighted vs candidate; all must be met.",
        _ties(
            [
                (profile, metric, CANDIDATE, UNWEIGHTED)
                for profile in ("min", "scale")
                for metric in (_ENERGY, _COVERAGE, _AUC)
            ]
        ),
    ),
    Finding(
        "F1b",
        True,
        "Its imputed net-worth q99 is about twice the holdout's (1.93 and 2.25).",
        "Difference, unweighted vs candidate, net-worth q99 ratio, and the "
        "unweighted mean above 1.5, both profiles; excess over 1 within a "
        "factor of two.",
        _f1b,
        1.0,
        "q99 ratio",
    ),
    Finding(
        "F1c",
        False,
        "Its tail W1/sd is five times the candidate's.",
        "Difference, unweighted vs candidate, net-worth W1/sd, both profiles; "
        "ratio within a factor of two.",
        _f1c,
        1.0,
        "ratio",
    ),
    Finding(
        "F2a",
        True,
        "With two targets the unchained ablation is indistinguishable from "
        "the candidate.",
        "Tie, unchained vs candidate, C2ST AUC, minimal profile.",
        _ties([("min", _AUC, CANDIDATE, UNCHAINED)]),
    ),
    Finding(
        "F2b",
        True,
        "On the four-component balance sheet, fitting targets independently "
        "costs 5.8 AUC points.",
        "Difference, unchained vs candidate, C2ST AUC, scale profile; gap "
        "within a factor of two.",
        _harness_difference("scale", _AUC, UNCHAINED),
        0.0,
        "AUC gap",
    ),
    Finding(
        "F2c",
        False,
        "Unchaining leaves the marginal and tail statistics essentially "
        "unchanged at scale.",
        "Four ties at scale, unchained vs candidate (energy, coverage, W1/sd, "
        "q99 ratio); all must be met.",
        _ties(
            [
                ("scale", metric, CANDIDATE, UNCHAINED)
                for metric in (_ENERGY, _COVERAGE, _TAIL_W1, _Q99)
            ]
        ),
    ),
    Finding(
        "F2d",
        False,
        "The ungated forest loses 5.6 AUC points at scale.",
        "Difference, plain_qrf vs candidate, C2ST AUC, scale profile; gap "
        "within a factor of two.",
        _harness_difference("scale", _AUC, PLAIN_FOREST),
        0.0,
        "AUC gap",
    ),
    Finding(
        "F3a",
        True,
        "At scale the classifier separates the unconditional marginal draw "
        "almost perfectly (AUC 0.97 against the candidate's 0.76).",
        "Difference, weighted_marginal vs candidate, C2ST AUC, scale profile, "
        "and the marginal draw's mean AUC at least 0.95; gap within a factor "
        "of two.",
        _harness_difference("scale", _AUC, MARGINAL, level_at_least=0.95),
        0.0,
        "AUC gap",
    ),
    Finding(
        "F3b",
        False,
        "Conditioning richness makes the test decisive: the marginal draw's "
        "AUC rises from the minimal profile to scale.",
        "Level: the marginal draw's mean AUC at scale exceeds its "
        "minimal-profile mean by more than 0.060.",
        _f3b,
    ),
    Finding(
        "F3c",
        False,
        "The candidate stays closest to the floor.",
        "Level, both profiles: the candidate has the lowest mean AUC among "
        "the candidate and the four baselines.",
        _f3c,
    ),
    Finding(
        "F4a",
        True,
        "Hot deck is within noise of the candidate on energy distance and "
        "coverage in both profiles.",
        "Four ties, hot deck vs candidate; all must be met.",
        _ties(
            [
                (profile, metric, CANDIDATE, HOT_DECK)
                for profile in ("min", "scale")
                for metric in (_ENERGY, _COVERAGE)
            ]
        ),
    ),
    Finding(
        "F4b",
        True,
        "Hot deck is within 2.1 and 0.7 AUC points of the candidate.",
        "Tie, hot deck vs candidate, C2ST AUC, each profile.",
        _ties([(profile, _AUC, CANDIDATE, HOT_DECK) for profile in ("min", "scale")]),
    ),
    Finding(
        "F4c",
        False,
        "At scale hot deck slightly beats the candidate on tail W1/sd.",
        "Level, scale: hot deck's net-worth W1/sd is no more than one floor "
        "sd above the candidate's.",
        _f4c,
    ),
    Finding(
        "F5a",
        False,
        "The candidate understates the extreme tail (net-worth q99 ratio "
        "0.70 and 0.77).",
        "Level, both profiles: below 1 on at least 8 of 10 seeds and mean in "
        "[0.60, 0.90].",
        _f5a,
    ),
    Finding(
        "F5b",
        False,
        "Hot deck does not fix the tail either, while the marginal draw sits "
        "near the floor (read from the 2022 tables, not the text).",
        "Ties on the net-worth q99 ratio, each profile: hot deck vs "
        "candidate, and marginal draw vs floor.",
        _f5b,
    ),
)


@dataclass(frozen=True)
class Judgement:
    """A finding's verdict: the replication arm judged against the baseline.

    Attributes:
        finding: The finding judged.
        baseline: The rule applied to the baseline (2022) arm.
        replication: The rule applied to the replication (2025) arm.
        bands: Per effect, the ``(low, high)`` band the replication's effect
            must fall in, on the display scale; empty without a magnitude.
        effects: The replication's effects on the display scale.
        magnitude_holds: Whether every effect is in its band (``None`` when
            the finding has no magnitude or its rule was not met).
        direction: ``"larger"``, ``"smaller"``, or ``"mixed"`` when the
            magnitude does not hold.
        verdict: One of the four verdicts of the criteria file.
    """

    finding: Finding
    baseline: Evaluation
    replication: Evaluation
    bands: tuple[tuple[float, float], ...]
    effects: tuple[float, ...]
    magnitude_holds: bool | None
    direction: str | None
    verdict: str


REPLICATED = "Replicated"
DIRECTION_ONLY = "Replicated in direction, magnitude differs"
VERDICT_INCONCLUSIVE = "Inconclusive"
NOT_REPLICATED = "Not replicated"


def judge(finding: Finding, baseline: Arm, replication: Arm) -> Judgement:
    """Apply a finding's rule to both arms and return the verdict.

    The verdict follows the replication arm's status, with one cap from the
    criteria file: a claim the baseline arm does not itself meet can at best
    be inconclusive. The magnitude band is the baseline's effect halved and
    doubled; it matters only when the rule is met on both arms.
    """
    base = finding.evaluate(baseline)
    repl = finding.evaluate(replication)
    offset = finding.effect_offset
    bands = tuple((offset + e / 2, offset + 2 * e) for e in base.effects)
    effects = tuple(offset + e for e in repl.effects)

    magnitude_holds: bool | None = None
    direction: str | None = None
    if repl.status == MET and base.effects:
        above = [b > 2 * a for a, b in zip(base.effects, repl.effects, strict=True)]
        below = [b < a / 2 for a, b in zip(base.effects, repl.effects, strict=True)]
        magnitude_holds = not any(above) and not any(below)
        if not magnitude_holds:
            if any(above) and any(below):
                direction = "mixed"
            else:
                direction = "larger" if any(above) else "smaller"

    if repl.status == FAILED:
        verdict = NOT_REPLICATED
    elif repl.status == INCONCLUSIVE or base.status != MET:
        # The criteria file: "A claim that was within seed noise in 2022 can
        # at best come out inconclusive." Meeting the rule on the new wave
        # does not replicate a finding the old wave never established.
        verdict = VERDICT_INCONCLUSIVE
    elif magnitude_holds is False:
        verdict = f"{DIRECTION_ONLY} ({direction})"
    else:
        verdict = REPLICATED
    return Judgement(
        finding, base, repl, bands, effects, magnitude_holds, direction, verdict
    )


def _goodness(metric: str, means: pd.Series) -> pd.Series:
    """Distance from a column's notion of best, so lower always ranks better."""
    if metric == _AUC:
        return (means - 0.5).abs()
    if metric.startswith(("q90_ratio", "q99_ratio")):
        return (means - 1.0).abs()
    if metric == _COVERAGE:
        return -means
    return means


def ranking_agreement(
    baseline: Arm,
    replication: Arm,
    columns: dict[str, Sequence[tuple[str | None, str, str]]],
) -> pd.DataFrame:
    """Spearman agreement of method rankings, per displayed table column.

    Args:
        baseline: The 2022 arm.
        replication: The 2025 arm.
        columns: Run key -> ``(target, metric, header)`` for each displayed
            column (``target`` is ignored for the population-view runs).

    Returns:
        One row per column: the Spearman correlation between the two arms'
        rankings of :data:`DISPLAYED_METHODS`, each arm's best method, and the
        criteria file's verdict (at least 0.8 replicates; 0.5 to 0.8 partly;
        below 0.5 does not).
    """
    from scipy.stats import spearmanr

    rows = []
    for run, specs in columns.items():
        for target, metric, header in specs:
            wealth_target = target if run == "wealth" else None
            scores = []
            for arm in (baseline, replication):
                means = pd.Series(
                    {
                        method: arm.mean(run, method, metric, wealth_target)
                        for method in DISPLAYED_METHODS
                    }
                )
                scores.append(_goodness(metric, means))
            # Rounded so tied ranks (the candidate and its unchained ablation
            # share every debt column) do not leave 0.9999999999999999.
            rho = round(float(spearmanr(scores[0], scores[1]).statistic), 10)
            best = [score.idxmin() for score in scores]
            if rho >= 0.8:
                verdict = "Replicated"
            elif rho >= 0.5:
                verdict = "Partly replicated"
            else:
                verdict = "Not replicated"
            rows.append(
                {
                    "run": run,
                    "column": header,
                    "metric": metric if wealth_target is None else f"{target}.{metric}",
                    "spearman": rho,
                    "best_baseline": best[0],
                    "best_replication": best[1],
                    "same_best": best[0] == best[1],
                    "verdict": verdict,
                }
            )
    return pd.DataFrame(rows)
