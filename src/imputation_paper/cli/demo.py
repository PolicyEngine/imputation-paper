"""``imp demo``: the dependency-free end-to-end path CI exercises.

Runs the trivial ``weighted_marginal`` baseline through the real harness -- toy
weighted data, paired donor/receiver splits, the weighted metrics -- and prints
the per-metric means over seeds. This proves the plumbing (registry ->
condition -> metrics) without importing any heavy method package; it makes no
scientific claim.
"""

from __future__ import annotations

import pandas as pd

from imputation_paper import smoke
from imputation_paper.experiments.conditions import run_condition
from imputation_paper.experiments.holdout import paired_splits

#: Seeds for the demo's repeated splits: enough to show the pairing mechanics,
#: few enough to stay instant.
DEMO_SPLIT_SEEDS: tuple[int, ...] = (0, 1, 2)


def run_demo(*, seed: int = 0, n: int = 800) -> int:
    """Run the toy pipeline and print grid-mean metrics per target.

    Args:
        seed: Seed for the toy-data draw.
        n: Toy-data row count (before the split).

    Returns:
        Process exit code (``0`` on success).
    """
    dataset = smoke.make_toy_dataset(seed=seed, n=n)
    # Pool the toy donor/receiver halves back into one observed table, then
    # re-split it with the paper's paired-split primitive so the demo exercises
    # the exact protocol the sweep uses.
    pooled = pd.concat([dataset.train, dataset.test], ignore_index=True)

    accumulated: dict[str, list[float]] = {}
    for split in paired_splits(pooled, seeds=DEMO_SPLIT_SEEDS):
        result = run_condition(
            "weighted_marginal",
            split.train,
            split.test,
            dataset.predictors,
            dataset.targets,
            weight_column=dataset.weight_column,
            seed=split.seed,
        )
        for name, value in result.metrics.items():
            accumulated.setdefault(name, []).append(value)

    print("imp demo -- weighted_marginal on the toy task")
    print(f"  rows={len(pooled)}, splits={len(DEMO_SPLIT_SEEDS)} (paired seeds)")
    for name in sorted(accumulated):
        values = accumulated[name]
        mean = sum(values) / len(values)
        print(f"  {name}: mean={mean:,.4f} over {len(values)} seeds")
    print(
        "note: weighted_marginal is the dependency-free harness baseline, "
        "not a contender; see PLAN.md for the real method surface."
    )
    return 0
