"""OpenML regression tasks: the cross-dataset sweep beyond economic microdata.

Loads the six OpenML AutoML-Benchmark regression datasets carried over from the
microimpute manuscript's appendix, each as an ``openml_<name>``
:class:`~imputation_paper.data.base.TaskFrame`. These tasks test whether the
method's behaviour transfers past survey microdata; they carry no survey design,
so each gets a constant ``weight = 1.0`` column (an unweighted task the weighted
metrics still run over).

Only numeric predictors are kept (categorical/object columns are dropped). To
bound forest cost when a dataset is wide, the predictor set is capped at the 12
columns most correlated with the target (by absolute Pearson correlation, ties
broken by column order) -- a deterministic selection documented on
:func:`load_openml`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from imputation_paper.data.base import TaskFrame

__all__ = ["load_openml", "OPENML_TASKS"]

#: Maximum numeric predictors kept per task, to bound forest training cost.
_MAX_PREDICTORS = 12

#: Constant weight column name for these design-free tasks.
_WEIGHT_COLUMN = "weight"

#: Per-dataset spec: expected row count, the target column, and the OpenML
#: version fetch_openml's *default* (version-agnostic) resolution lands on. The
#: target is named explicitly because some datasets (e.g. ``house_sales``) expose
#: an empty ``target_names`` and carry the target as an ordinary frame column.
#: The version is not passed to fetch_openml (its default already resolves to the
#: value recorded here -- five at version 1, ``brazilian_houses`` at version 4,
#: which has *no* version 1); instead :func:`load_openml` *asserts* the resolved
#: version and row count, so any silent OpenML re-versioning fails loudly rather
#: than swapping the data underneath the sweep.
OPENML_TASKS: dict[str, dict[str, object]] = {
    "space_ga": {"rows": 3107, "target": "ln(VOTES/POP)", "version": 1},
    "elevators": {"rows": 16599, "target": "Goal", "version": 1},
    "brazilian_houses": {"rows": 10692, "target": "total_(BRL)", "version": 4},
    "onlinenewspopularity": {"rows": 39644, "target": "shares", "version": 1},
    "abalone": {"rows": 4177, "target": "Class_number_of_rings", "version": 1},
    "house_sales": {"rows": 21613, "target": "price", "version": 1},
}


def _select_predictors(frame: pd.DataFrame, target: str) -> list[str]:
    """Return the numeric non-target predictor columns, capped and deterministic.

    Keeps every numeric column except the target; if more than
    :data:`_MAX_PREDICTORS` remain, keeps the ``_MAX_PREDICTORS`` with the
    highest absolute Pearson correlation to the target. Ordering and tie-breaks
    are deterministic: candidates are ranked by ``(-abs_corr, original_index)``,
    so the result is stable across runs and independent of dict/hash ordering.
    A column whose correlation is undefined (zero variance) sorts last.
    """
    numeric = frame.select_dtypes(include="number")
    candidates = [c for c in numeric.columns if c != target]
    if len(candidates) <= _MAX_PREDICTORS:
        return candidates

    target_values = frame[target].to_numpy(dtype=np.float64)
    ranked = []
    for position, column in enumerate(candidates):
        values = frame[column].to_numpy(dtype=np.float64)
        if np.std(values) == 0 or np.std(target_values) == 0:
            abs_corr = -1.0  # undefined correlation: rank last
        else:
            abs_corr = abs(float(np.corrcoef(values, target_values)[0, 1]))
            if not np.isfinite(abs_corr):
                abs_corr = -1.0
        ranked.append((-abs_corr, position, column))
    ranked.sort()
    return [column for _, _, column in ranked[:_MAX_PREDICTORS]]


def load_openml(name: str) -> TaskFrame:
    """Load one OpenML regression dataset as an ``openml_<name>`` task.

    Fetches the dataset at fetch_openml's default version (verified to match the
    expected row count), keeps numeric predictors only, selects at most the 12
    most target-correlated of them (see :func:`_select_predictors`), coerces the
    target numeric, and adds a constant ``weight = 1.0`` column.

    Args:
        name: One of the keys of :data:`OPENML_TASKS`.

    Returns:
        A :class:`~imputation_paper.data.base.TaskFrame` named
        ``f"openml_{name}"``.

    Raises:
        KeyError: If ``name`` is not a known OpenML task.
        ValueError: If the fetched version or row count does not match the
            recorded expectation, or the target column is absent.
    """
    from sklearn.datasets import fetch_openml

    if name not in OPENML_TASKS:
        raise KeyError(f"Unknown OpenML task {name!r}; known: {sorted(OPENML_TASKS)}.")
    spec = OPENML_TASKS[name]
    target = str(spec["target"])
    expected_rows = int(spec["rows"])  # type: ignore[call-overload]
    expected_version = int(spec["version"])  # type: ignore[call-overload]

    dataset = fetch_openml(name=name, as_frame=True, parser="auto")
    resolved_version = int(dataset.details.get("version", -1))
    if resolved_version != expected_version:
        raise ValueError(
            f"OpenML {name!r} resolved to version {resolved_version}, expected "
            f"{expected_version}; OpenML may have re-versioned the dataset."
        )
    frame = dataset.frame
    if frame.shape[0] != expected_rows:
        raise ValueError(
            f"OpenML {name!r} returned {frame.shape[0]} rows, expected "
            f"{expected_rows}; the default version may have changed."
        )
    if target not in frame.columns:
        raise ValueError(
            f"OpenML {name!r} has no target column {target!r}; columns: "
            f"{list(frame.columns)}."
        )

    predictors = _select_predictors(frame, target)
    # Coerce target numeric (some targets arrive as object/categorical).
    target_series = pd.to_numeric(frame[target], errors="coerce")

    used = pd.DataFrame({column: frame[column] for column in predictors})
    used[target] = target_series
    used[_WEIGHT_COLUMN] = 1.0
    used = used.astype(np.float64).dropna().reset_index(drop=True)

    notes = (
        f"Source: OpenML dataset {name!r} via sklearn.fetch_openml "
        "(default version; row count verified).",
        f"Kept {len(predictors)} numeric predictor(s) "
        f"(cap {_MAX_PREDICTORS}, ranked by |corr| with the target).",
        "Design-free task: constant weight = 1.0.",
    )
    return TaskFrame(
        name=f"openml_{name}",
        frame=used,
        predictors=tuple(predictors),
        targets=(target,),
        weight_column=_WEIGHT_COLUMN,
        notes=notes,
    )
