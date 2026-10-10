"""The pre-registered SCF 2025 replication rules: pinned, with their invariants.

``docs/scf-2025-replication-criteria.md`` was committed before any method was
fit on the 2025 wave. These tests hold the code to it:

* the criteria file itself is frozen (its hash is pinned);
* the code's findings are the file's findings, in order, with the same
  primaries;
* applied to the committed 2022 runs, the code returns exactly the "2022 under
  the same rule" column the file states, and the magnitude bands it states;
* the report, the appendix's generated inputs, and the appendix prose are
  current with the committed runs.

They also state the rules' invariants and check them on generated inputs:

* a difference that is met one way is failed the other way;
* a tie is symmetric in its two methods;
* no contrast is both a met tie and a met difference;
* statuses do not depend on the metric's units or on how seeds are labelled;
* an arm judged against itself is never "not replicated" on a rule it meets,
  and its magnitude always holds;
* shrinking a met effect moves the verdict through "magnitude differs" to
  "not replicated", never the other way.

Everything here runs from the committed long files; nothing touches the
network or needs the ``methods`` extra.
"""

from __future__ import annotations

import hashlib
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from imputation_paper.cli import replication as report_cli
from imputation_paper.cli.tables import HARNESS_COLUMNS, SCF_WEALTH_COLUMNS, make_tables
from imputation_paper.experiments import replication as rules
from imputation_paper.experiments.replication import (
    CANDIDATE,
    FAILED,
    FINDINGS,
    FLOOR,
    INCONCLUSIVE,
    MET,
    Arm,
    Contrast,
)

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
CRITERIA = ROOT / "docs" / "scf-2025-replication-criteria.md"
APPENDIX = ROOT / "paper" / "sections" / "appendix_scf2025.tex"

#: SHA-256 of the criteria file as committed before the 2025 runs (f1e4827).
#: The file says nothing in it is revised after the runs; this holds it to that.
CRITERIA_SHA256 = "63f25cfbf58d3df897cb8c526b7a9fc27f62e6b6496cf7f3649fec4f0cbd83b6"

#: The criteria file's "2022 under the same rule" column.
STATUS_2022 = {key: MET for key in (f.key for f in FINDINGS)} | {
    "W3b": INCONCLUSIVE,
    "W3c": INCONCLUSIVE,
}

#: The magnitude bands the criteria file states, on the scale it states them.
BANDS = {
    "W1": [(3.55, 11.21)],
    "W2": [(1.74, 3.96)],
    "W3b": [(1.36, 2.45)],
    "W3c": [(1.08, 1.33)],
    "W4b": [(2.37, 6.49)],
    "F1b": [(1.47, 2.86), (1.62, 3.50)],
    "F1c": [(3.04, 9.16), (2.97, 8.89)],
    "F2b": [(0.029, 0.115)],
    "F2d": [(0.028, 0.112)],
    "F3a": [(0.103, 0.413)],
}


@pytest.fixture(scope="module")
def arm_2022() -> Arm:
    return rules.load_arm(RUNS, "", "2022")


# ---------------------------------------------------------------------------
# The code is the criteria file
# ---------------------------------------------------------------------------


def test_criteria_file_is_frozen() -> None:
    digest = hashlib.sha256(CRITERIA.read_bytes()).hexdigest()
    assert digest == CRITERIA_SHA256, (
        "docs/scf-2025-replication-criteria.md changed after it was "
        "pre-registered. It is not to be revised; report any ill-posed rule "
        "in the comparison document instead."
    )


def test_findings_are_the_criteria_files_findings() -> None:
    """Same IDs, same order, same primaries as the two findings tables."""
    rows = re.findall(
        r"^\| ([WGF]\d[a-d]?)( \(primary\))? \|", CRITERIA.read_text(), re.M
    )
    assert [key for key, _ in rows] == [f.key for f in FINDINGS]
    assert {key for key, primary in rows if primary} == {
        f.key for f in FINDINGS if f.primary
    }


def test_rules_reproduce_the_stated_2022_statuses(arm_2022: Arm) -> None:
    statuses = {f.key: f.evaluate(arm_2022).status for f in FINDINGS}
    assert statuses == STATUS_2022


def test_rules_reproduce_the_stated_magnitude_bands(arm_2022: Arm) -> None:
    computed = {
        j.finding.key: j.bands
        for j in (rules.judge(f, arm_2022, arm_2022) for f in FINDINGS)
        if j.bands
    }
    assert set(computed) == set(BANDS)
    for key, bands in BANDS.items():
        for (low, high), (stated_low, stated_high) in zip(
            computed[key], bands, strict=True
        ):
            tolerance = 0.0006 if stated_high < 1 else 0.006
            assert low == pytest.approx(stated_low, abs=tolerance), key
            assert high == pytest.approx(stated_high, abs=tolerance), key


def test_displayed_columns_are_the_papers_fifteen() -> None:
    assert len(SCF_WEALTH_COLUMNS) == 5
    assert len(HARNESS_COLUMNS) == 5
    assert len(rules.DISPLAYED_METHODS) == 8


# ---------------------------------------------------------------------------
# Rule thresholds
# ---------------------------------------------------------------------------


def _contrast(
    dbar: float, k_pos: int, *, floor_sd: float | None, a: float = 1.0
) -> Contrast:
    return Contrast(a, a + dbar, dbar, k_pos, 10 - k_pos, 10, floor_sd)


@pytest.mark.parametrize(
    ("dbar", "k_pos", "expected"),
    [
        (0.5, 8, MET),
        (0.5, 7, INCONCLUSIVE),
        (0.0, 10, FAILED),
        (-0.5, 2, FAILED),
    ],
)
def test_within_scf_difference_thresholds(
    dbar: float, k_pos: int, expected: str
) -> None:
    assert rules.difference_status(_contrast(dbar, k_pos, floor_sd=None)) == expected


@pytest.mark.parametrize(
    ("dbar", "k_pos", "expected"),
    [
        (2.01, 8, MET),
        (2.01, 7, INCONCLUSIVE),
        (2.0, 10, INCONCLUSIVE),  # must exceed two floor sds
        (1.5, 10, INCONCLUSIVE),
        (1.0, 10, FAILED),  # at most one floor sd
        (-3.0, 0, FAILED),
    ],
)
def test_population_view_difference_thresholds(
    dbar: float, k_pos: int, expected: str
) -> None:
    assert rules.difference_status(_contrast(dbar, k_pos, floor_sd=1.0)) == expected


@pytest.mark.parametrize(
    ("dbar", "expected"),
    [
        (0.0, MET),
        (1.0, MET),
        (-1.0, MET),
        (1.5, INCONCLUSIVE),
        (2.0, INCONCLUSIVE),
        (2.01, FAILED),
        (-2.01, FAILED),
    ],
)
def test_population_view_tie_thresholds(dbar: float, expected: str) -> None:
    assert rules.tie_status(_contrast(dbar, 5, floor_sd=1.0)) == expected


@pytest.mark.parametrize(
    ("dbar", "k_pos", "expected"),
    [
        (0.05, 5, MET),  # even seeds, ratio 1.05
        (0.05, 8, INCONCLUSIVE),  # lopsided seeds, ratio inside the band
        (0.50, 5, INCONCLUSIVE),  # even seeds, ratio outside the band
        (0.50, 9, FAILED),
        (-0.30, 1, FAILED),  # ratio 0.70, the other method wins 9 of 10
    ],
)
def test_within_scf_tie_thresholds(dbar: float, k_pos: int, expected: str) -> None:
    assert rules.tie_status(_contrast(dbar, k_pos, floor_sd=None)) == expected


def test_seeds_only_tie_ignores_the_ratio() -> None:
    near_zero = Contrast(0.001, 0.004, 0.003, 6, 4, 10, None)
    assert rules.tie_status(near_zero, ratio_band=False) == MET
    assert rules.tie_status(near_zero) == INCONCLUSIVE
    lopsided = Contrast(0.001, 0.004, 0.003, 9, 1, 10, None)
    assert rules.tie_status(lopsided, ratio_band=False) == FAILED


# ---------------------------------------------------------------------------
# Invariants of the rules, on generated runs
# ---------------------------------------------------------------------------

#: Dyadic values: sums and power-of-two rescalings of these are exact in
#: floating point, so an invariance can be asserted as equality.
_dyadic = st.integers(min_value=1, max_value=4096).map(lambda n: n / 64)
_ten = st.lists(_dyadic, min_size=10, max_size=10)


def _harness_arm(a: list[float], b: list[float], floor: list[float]) -> Arm:
    rows = [
        {"method": method, "seed": seed, "view": "scf", "metric": "m", "value": value}
        for method, values in (("a", a), ("b", b), (FLOOR, floor))
        for seed, value in enumerate(values)
    ]
    frame = pd.DataFrame(rows)
    empty = pd.DataFrame(columns=["method", "seed", "target", "metric", "value"])
    return Arm("generated", empty, frame, frame)


def _wealth_arm(a: list[float], b: list[float]) -> Arm:
    rows = [
        {"method": method, "seed": seed, "target": "t", "metric": "m", "value": value}
        for method, values in (("a", a), ("b", b))
        for seed, value in enumerate(values)
    ]
    frame = pd.DataFrame(rows)
    empty = pd.DataFrame(columns=["method", "seed", "view", "metric", "value"])
    return Arm("generated", frame, empty, empty)


def _both_ways(arm: Arm, run: str) -> tuple[Contrast, Contrast]:
    target = "t" if run == "wealth" else None
    return (
        rules.contrast(arm, run, "m", "a", "b", target),
        rules.contrast(arm, run, "m", "b", "a", target),
    )


@settings(max_examples=200, deadline=None)
@given(a=_ten, b=_ten, floor=_ten)
def test_population_view_rules_are_coherent(
    a: list[float], b: list[float], floor: list[float]
) -> None:
    forward, backward = _both_ways(_harness_arm(a, b, floor), "min")
    # A difference met one way is failed the other way.
    if rules.difference_status(forward) == MET:
        assert rules.difference_status(backward) == FAILED
    # A tie does not care which method is called A.
    assert rules.tie_status(forward) == rules.tie_status(backward)
    # No contrast is both a tie and a difference.
    assert not (
        rules.tie_status(forward) == MET and rules.difference_status(forward) == MET
    )


@settings(max_examples=200, deadline=None)
@given(a=_ten, b=_ten)
def test_within_scf_rules_are_coherent(a: list[float], b: list[float]) -> None:
    forward, backward = _both_ways(_wealth_arm(a, b), "wealth")
    if rules.difference_status(forward) == MET:
        assert rules.difference_status(backward) == FAILED
    assert rules.tie_status(forward, ratio_band=False) == rules.tie_status(
        backward, ratio_band=False
    )
    assert not (
        rules.tie_status(forward) == MET and rules.difference_status(forward) == MET
    )
    # The ratio band is its own reciprocal, so a met tie is met both ways.
    low, high = rules.TIE_RATIO_BAND
    assert low * high == 1.0
    assume(not np.isclose(forward.ratio, [low, high], rtol=1e-9).any())
    assert (rules.tie_status(forward) == MET) == (rules.tie_status(backward) == MET)


@settings(max_examples=200, deadline=None)
@given(
    a=_ten,
    b=_ten,
    floor=_ten,
    scale=st.sampled_from([0.25, 0.5, 2.0, 1024.0]),
    order=st.permutations(range(10)),
)
def test_statuses_ignore_units_and_seed_labels(
    a: list[float], b: list[float], floor: list[float], scale: float, order: list[int]
) -> None:
    """Rescaling the metric or relabelling the seeds changes no status."""

    def statuses(arm: Arm, run: str) -> tuple[str, str]:
        forward, _ = _both_ways(arm, run)
        return rules.difference_status(forward), rules.tie_status(forward)

    def moved(values: list[float]) -> list[float]:
        return [scale * values[i] for i in order]

    # Skip inputs that sit exactly on a threshold: there the permuted floor sd
    # may round to the other side.
    forward, _ = _both_ways(_harness_arm(a, b, floor), "min")
    sd = forward.floor_sd or 0.0
    assume(not np.isclose(abs(forward.dbar), [sd, 2 * sd], rtol=1e-9, atol=0).any())

    assert statuses(_harness_arm(a, b, floor), "min") == statuses(
        _harness_arm(moved(a), moved(b), moved(floor)), "min"
    )
    assert statuses(_wealth_arm(a, b), "wealth") == statuses(
        _wealth_arm(moved(a), moved(b)), "wealth"
    )


def test_an_arm_judged_against_itself(arm_2022: Arm) -> None:
    """Self-comparison: met rules replicate, and every magnitude holds."""
    for finding in FINDINGS:
        judgement = rules.judge(finding, arm_2022, arm_2022)
        if judgement.baseline.status == MET:
            assert judgement.verdict == rules.REPLICATED, finding.key
            assert judgement.magnitude_holds in (True, None), finding.key
        else:
            assert judgement.verdict == rules.VERDICT_INCONCLUSIVE, finding.key


def test_a_claim_unmet_in_2022_is_at_best_inconclusive(arm_2022: Arm) -> None:
    """The criteria file's cap: 2022 seed noise cannot be replicated.

    W3c (unchained worse on net-worth W1) is inconclusive on the 2022 runs
    (7 of 10 seeds). Make it clearly met on a perturbed arm, and clearly
    failed on another: the verdicts are inconclusive and not replicated.
    """
    w3c = next(f for f in FINDINGS if f.key == "W3c")
    assert w3c.evaluate(arm_2022).status == INCONCLUSIVE

    worse = _scale_method(arm_2022, rules.UNCHAINED, 3.0)
    assert w3c.evaluate(worse).status == MET
    assert rules.judge(w3c, arm_2022, worse).verdict == rules.VERDICT_INCONCLUSIVE

    better = _scale_method(arm_2022, rules.UNCHAINED, 0.2)
    assert w3c.evaluate(better).status == FAILED
    assert rules.judge(w3c, arm_2022, better).verdict == rules.NOT_REPLICATED


def _scale_method(arm: Arm, method: str, factor: float) -> Arm:
    """The 2022 arm with one method's net-worth W1 multiplied by ``factor``."""
    wealth = arm.wealth.copy()
    mask = (
        wealth["method"].eq(method)
        & wealth["target"].eq("networth")
        & wealth["metric"].eq("wasserstein1")
    )
    wealth.loc[mask, "value"] *= factor
    return Arm("perturbed", wealth, arm.minimal, arm.scale)


@pytest.mark.parametrize(
    ("factor", "verdict"),
    [
        (1.0, "Replicated"),
        (1.5, "Replicated"),  # ratio 9.2, inside [3.55, 11.21]
        (2.0, "Replicated in direction, magnitude differs (larger)"),  # 12.2
        (0.5, "Replicated in direction, magnitude differs (smaller)"),  # 3.05
        (0.1, "Not replicated"),  # ratio 0.61: unweighted now better
    ],
)
def test_verdict_tracks_the_effect(arm_2022: Arm, factor: float, verdict: str) -> None:
    """W1 with the unweighted ablation's net-worth W1 scaled up and down."""
    w1 = FINDINGS[0]
    assert w1.key == "W1"
    perturbed = _scale_method(arm_2022, rules.UNWEIGHTED, factor)
    assert rules.judge(w1, arm_2022, perturbed).verdict == verdict


def test_ranking_agreement_is_one_against_itself_and_ignores_units(
    arm_2022: Arm,
) -> None:
    columns = {
        "wealth": SCF_WEALTH_COLUMNS,
        "min": HARNESS_COLUMNS,
        "scale": HARNESS_COLUMNS,
    }
    same = rules.ranking_agreement(arm_2022, arm_2022, columns)
    assert len(same) == 15
    assert same["spearman"].eq(1.0).all()
    assert same["same_best"].all()

    rebased = arm_2022.wealth.copy()
    dollars = rebased["metric"].isin(["pinball_loss", "wasserstein1"])
    rebased.loc[dollars, "value"] *= report_cli.REBASE_2022_TO_2025
    other = Arm("rebased", rebased, arm_2022.minimal, arm_2022.scale)
    assert rules.ranking_agreement(arm_2022, other, columns)["spearman"].eq(1.0).all()


def test_ranking_agreement_detects_a_reversal(arm_2022: Arm) -> None:
    """Negating a lower-is-better column reverses its ranking exactly."""
    flipped = arm_2022.wealth.copy()
    mask = flipped["target"].eq("networth") & flipped["metric"].eq("wasserstein1")
    flipped.loc[mask, "value"] *= -1
    other = Arm("flipped", flipped, arm_2022.minimal, arm_2022.scale)
    result = rules.ranking_agreement(arm_2022, other, {"wealth": SCF_WEALTH_COLUMNS})
    row = result[result["metric"].eq("networth.wasserstein1")].iloc[0]
    assert row["spearman"] == pytest.approx(-1.0)
    assert row["verdict"] == "Not replicated"


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------


def _copy_arm(source_suffix: str, target_suffix: str, target_root: Path) -> None:
    for stem in ("scf-wealth", "scf-to-cps-harness", "scf-to-cps-harness-scale"):
        shutil.copytree(
            RUNS / f"{stem}{source_suffix}", target_root / f"{stem}{target_suffix}"
        )


def test_report_on_identical_arms(tmp_path: Path) -> None:
    """2022 against a copy of itself: nothing fails, nothing moves."""
    for suffix in ("", "-2025", "-2022-rerun"):
        _copy_arm("", suffix, tmp_path)
    report = report_cli.build_report(tmp_path)
    assert report.baseline_letter == "A"
    assert report.findings["verdict"].value_counts().to_dict() == {
        "Replicated": 25,
        "Inconclusive": 2,
    }
    assert report.reproduction["identical"].all()
    assert report.reproduction["gap_in_sd"].eq(0).all()
    assert report.rankings["spearman"].eq(1.0).all()

    text = report_cli.render_markdown(report)
    for heading in (
        "## Summary",
        "## Verdicts",
        "## Headline numbers",
        "## Method rankings",
        "## Full tables",
        "## Diagnostics",
    ):
        assert heading in text
    macros = report_cli.number_macros(report)
    for headline in report_cli.HEADLINES:
        assert macros[f"{headline.key}:2022"] == macros[f"{headline.key}:2025"]
    assert macros["nw-w1-unweighted-ratio:2022"] == "6.10"
    assert macros["nw-w1-candidate:2022"] == "135,871"
    assert macros["nw-w1-candidate:2022r"] == "148,694"  # 135,871 x 4789/4376


def test_rerun_that_drifts_replaces_the_baseline(tmp_path: Path) -> None:
    """If today's environment does not reproduce the paper, say so and switch."""
    for suffix in ("", "-2025", "-2022-rerun"):
        _copy_arm("", suffix, tmp_path)
    path = tmp_path / "scf-wealth-2022-rerun" / "metrics_long.csv"
    long = pd.read_csv(path)
    drift = long["method"].eq(CANDIDATE) & long["metric"].eq("wasserstein1")
    long.loc[drift, "value"] *= 3.0
    long.to_csv(path, index=False)
    report = report_cli.build_report(tmp_path)
    assert report.baseline_letter == "D"
    assert "rerun replaces the committed runs as the baseline" in (
        report_cli.render_markdown(report)
    )


def test_headline_keys_are_unique_and_macro_safe() -> None:
    keys = [headline.key for headline in report_cli.HEADLINES]
    assert len(keys) == len(set(keys))
    assert all(re.fullmatch(r"[a-z0-9-]+", key) for key in keys)


# ---------------------------------------------------------------------------
# The committed outputs are current
# ---------------------------------------------------------------------------


def test_committed_report_and_tables_are_current(tmp_path: Path) -> None:
    """Regenerating from the committed runs changes no committed output."""
    docs, tables = tmp_path / "docs", tmp_path / "tables"
    assert report_cli.make_replication(RUNS, docs, tables) == 0
    assert make_tables(RUNS, tables) == 0
    committed_docs = ROOT / "docs" / "scf-2025-replication"
    for path in sorted(docs.iterdir()):
        assert path.read_bytes() == (committed_docs / path.name).read_bytes(), path.name
    committed_tables = ROOT / "paper" / "tables"
    for name in (
        "scf2025_verdicts_wealth.tex",
        "scf2025_verdicts_view_a.tex",
        "scf2025_verdicts_view_b.tex",
        "scf2025_numbers.tex",
        "scf_wealth_2025.tex",
        "harness_2025.tex",
        "harness_scale_2025.tex",
        "scf_wealth.tex",
        "harness.tex",
        "harness_scale.tex",
    ):
        assert (tables / name).read_bytes() == (committed_tables / name).read_bytes(), (
            name
        )


def test_appendix_cites_only_generated_numbers() -> None:
    r"""Every ``\rep{key}`` in the appendix exists, and no number is typed in.

    The appendix states results only through ``\rep{...}``. Digits may appear
    in years, wave labels, finding IDs, cross-references, and the handful of
    constants of the pre-registered rules listed below.
    """
    text = APPENDIX.read_text()
    defined = set(
        re.findall(
            r"\\csname rep:([^\\]+)\\endcsname\{",
            (ROOT / "paper" / "tables" / "scf2025_numbers.tex").read_text(),
        )
    )
    used = set(re.findall(r"\\rep\{([^}]+)\}", text))
    assert used, "the appendix cites no generated number"
    assert used <= defined, sorted(used - defined)

    prose = re.sub(r"(?m)(?<!\\)%.*$", "", text)  # comments
    prose = re.sub(r"\\(rep|ref|label|input|cite[pt]?)\{[^}]*\}", "", prose)
    allowed = (
        r"20(21|22|23|24|25|26)",  # years and wave labels
        r"\b[WGF]\d[a-d]?\b",  # finding IDs
        r"\b(two|three|four|five|seven|eight|ten)\b",
        r"Forbes 400",
        r"4789/4376",  # the Fed's re-basing factor
        r"0\.5",  # the indistinguishable AUC
        r"q99|q90|W_1",
        r"\\textwidth|\\footnotesize|\\scriptsize",
    )
    for pattern in allowed:
        prose = re.sub(pattern, "", prose)
    stray = re.findall(r"\d[\d,.]*", prose)
    assert not stray, f"numbers typed into the appendix: {stray}"


def test_unit_invariance_is_exact_for_a_pure_change_of_units(arm_2022: Arm) -> None:
    """An arm that differs only in dollar units matches on every metric row."""
    wealth = arm_2022.wealth.copy()
    dollars = wealth["metric"].isin(["pinball_loss", "wasserstein1"])
    wealth.loc[dollars, "value"] *= report_cli.REBASE_2022_TO_2025
    rebased = Arm("rebased", wealth, arm_2022.minimal, arm_2022.scale)
    units = report_cli.unit_invariance(arm_2022, rebased)
    assert len(units) == len(rules.DISPLAYED_METHODS)
    assert (units["rows_matching"] == units["rows"]).all()
    assert units["max_w1_mean_relative_difference"].max() < 1e-12

    # A real difference in one method shows up in that method only.
    wealth.loc[wealth["method"].eq(rules.HOT_DECK) & dollars, "value"] *= 1.01
    shifted = Arm("shifted", wealth, arm_2022.minimal, arm_2022.scale)
    units = report_cli.unit_invariance(arm_2022, shifted).set_index("method")
    assert units.loc[rules.HOT_DECK, "max_w1_mean_relative_difference"] > 0.009
    others = units.drop(index=rules.HOT_DECK)
    assert (others["rows_matching"] == others["rows"]).all()
