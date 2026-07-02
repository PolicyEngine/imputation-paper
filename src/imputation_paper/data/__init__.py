"""Real-data loaders for the paper's tasks, behind the ``data`` optional extra.

Each loader returns a :class:`~imputation_paper.data.base.TaskFrame` (the SCF->CPS
receiver, :func:`load_cps_households`, returns a plain household
:class:`~pandas.DataFrame`). The tasks:

======================  =========================  ==============================
Loader                  Task name                  Source
======================  =========================  ==============================
:func:`load_scf`        ``scf_wealth``             Fed SCF 2022 summary extract
:func:`load_cps`        ``cps_components``         raw ``cps_2023.h5`` (the Hub)
:func:`load_cps_households`  (SCF->CPS receiver)   raw ``cps_2023.h5`` (the Hub)
:func:`load_openml`     ``openml_<name>``          six OpenML regression sets
:func:`load_sipp`       ``sipp_components``        local SIPP 2022 PU (optional)
======================  =========================  ==============================

**Lazy by construction.** Loaders import their network/format dependencies
(``huggingface_hub``, ``h5py``) *inside their own bodies*, so ``import
imputation_paper.data`` and the names below resolve on the base install without
the ``data`` extra -- a loader only raises :class:`ModuleNotFoundError` if it is
*called* without its dependency present. The re-exports here go through a
module-level ``__getattr__`` so even importing a specific loader stays lazy.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from imputation_paper.data.base import TaskFrame, cache_dir, download

if TYPE_CHECKING:  # pragma: no cover - typing only
    from imputation_paper.data.cps import load_cps, load_cps_households
    from imputation_paper.data.openml import OPENML_TASKS, load_openml
    from imputation_paper.data.scf import load_scf
    from imputation_paper.data.sipp import load_sipp

__all__ = [
    "TaskFrame",
    "cache_dir",
    "download",
    "load_scf",
    "load_cps",
    "load_cps_households",
    "load_openml",
    "OPENML_TASKS",
    "load_sipp",
]

#: Map re-exported name -> (submodule, attribute) for lazy resolution.
_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "load_scf": ("scf", "load_scf"),
    "load_cps": ("cps", "load_cps"),
    "load_cps_households": ("cps", "load_cps_households"),
    "load_openml": ("openml", "load_openml"),
    "OPENML_TASKS": ("openml", "OPENML_TASKS"),
    "load_sipp": ("sipp", "load_sipp"),
}


def __getattr__(name: str) -> object:
    """Resolve loader names lazily from their submodules (PEP 562)."""
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    module = importlib.import_module(f"{__name__}.{target[0]}")
    return getattr(module, target[1])


def __dir__() -> list[str]:
    """Include the lazily-exported names in ``dir()`` for discoverability."""
    return sorted(set(globals()) | set(__all__))
