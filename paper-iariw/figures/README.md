# Figures

Final figures land here. Regenerate experiment figures with:

```bash
uv run --extra viz l0 figures --sweep runs/weighted-loss-3seed --paper-figures
```

Or run the current-paper wrapper:

```bash
uv run --extra data --extra viz l0 paper \
    --consumer-facts data/targets/consumer_facts.jsonl
```

The manuscript currently references:

- `fig:pipeline` (Data): Populace pipeline overview for this build's configuration.
- `fig:objective_frontier` (Results): full-surface Populace objective loss vs. retained-record budget.
- `fig:budget_frontier` (Results): supplemental raw mean/median ARE vs. retained-record budget.
- `fig:usability` (Results): weight concentration and effective-sample-size diagnostics.
- `fig:cost_accuracy` (Results): runtime and accuracy tradeoff.

The pipeline overview is generated separately:

```bash
uv run python paper/figures/populace_pipeline.py
```
