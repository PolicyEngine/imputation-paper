"""Shared data-loader primitives: the ``TaskFrame`` and a caching downloader.

This module is deliberately lightweight -- it imports only the standard library
plus numpy/pandas -- so the whole ``data`` package can be imported (and its
registry listed) on the base install, without the ``data`` optional extra. The
network- and format-heavy dependencies (``huggingface_hub``, ``h5py``,
``zipfile`` over multi-hundred-megabyte files) live *inside* the individual
loader functions in the sibling modules, so importing this module never pulls
them in.

Every loader returns a :class:`TaskFrame`: one task's observed table, already
reduced to the numeric predictor/target columns the sweep will fit and score
over, plus the survey weight column. The contract the sweep relies on -- float64
columns, no NaNs in used columns, a positive weight column, a fresh
``RangeIndex`` -- is documented on the dataclass and enforced by each loader (and
checked by ``tests/test_data.py``).
"""

from __future__ import annotations

import hashlib
import os
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

__all__ = ["TaskFrame", "cache_dir", "download"]

#: Environment variable that overrides the on-disk cache location.
CACHE_ENV_VAR = "IMPUTATION_PAPER_CACHE"

#: Default cache location when :data:`CACHE_ENV_VAR` is unset.
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "imputation-paper"

#: Read size for streaming a download to disk while hashing it.
_DOWNLOAD_CHUNK_BYTES = 1 << 20  # 1 MiB

#: User-Agent sent with downloads. Some hosts (e.g. the Federal Reserve behind
#: Cloudflare) reject urllib's default agent with HTTP 403, so we present a
#: conventional browser agent.
_DOWNLOAD_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)


@dataclass(frozen=True)
class TaskFrame:
    """One task's observed table: rows, predictors, targets, weight column.

    A ``TaskFrame`` is the frozen, self-describing input the sweep fits and
    scores over. Loaders guarantee the following invariants (all checked in
    ``tests/test_data.py``):

    * ``frame`` holds only the columns named in ``predictors``, ``targets`` and
      ``weight_column`` (in that order), every one of them ``float64``;
    * no NaNs remain in any used column (rows with a NaN in a used column are
      dropped by the loader);
    * ``weight_column`` is strictly positive;
    * ``frame`` carries a fresh ``RangeIndex`` (``reset_index(drop=True)``).

    Attributes:
        name: Stable task identifier (e.g. ``"scf_wealth"``); used in run
            directories and result tables.
        frame: The observed table -- numeric columns only, no NaNs in used
            columns.
        predictors: Column names used as imputation inputs.
        targets: Column names to impute (the held-out variables).
        weight_column: The survey weight column carried by ``frame``.
    """

    name: str
    frame: pd.DataFrame
    predictors: tuple[str, ...]
    targets: tuple[str, ...]
    weight_column: str
    #: Free-form provenance notes (source, derived-variable mappings). Excluded
    #: from equality so two loads of the same task compare equal on their data.
    notes: tuple[str, ...] = field(default=(), compare=False)

    @property
    def used_columns(self) -> tuple[str, ...]:
        """The columns the sweep touches: predictors, then targets, then weight."""
        return (*self.predictors, *self.targets, self.weight_column)


def cache_dir() -> Path:
    """Return the on-disk cache directory, creating it on demand.

    Honours the :data:`CACHE_ENV_VAR` environment variable
    (``IMPUTATION_PAPER_CACHE``); otherwise falls back to
    ``~/.cache/imputation-paper``. The directory (and any parents) is created if
    it does not exist.

    Returns:
        The cache directory as a :class:`~pathlib.Path`.
    """
    override = os.environ.get(CACHE_ENV_VAR)
    path = Path(override).expanduser() if override else DEFAULT_CACHE_DIR
    path.mkdir(parents=True, exist_ok=True)
    return path


def _sha256_of_file(path: Path) -> str:
    """Return the hex SHA-256 of a file, read in chunks (no full slurp)."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_DOWNLOAD_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, filename: str) -> Path:
    """Download ``url`` to ``cache_dir()/filename`` (idempotently), hashing it.

    On download the file's SHA-256 is written to a ``<filename>.sha256`` sidecar
    and logged. The hash is *recorded*, not *checked*: pinning expected hashes is
    deferred to when the run configs freeze their inputs, so this helper only
    makes the hash of whatever was fetched auditable.

    If the target file already exists it is reused as-is (no re-download); a
    missing sidecar is regenerated from the cached file so the hash is always
    available.

    Args:
        url: The URL to fetch.
        filename: The basename to store it under in :func:`cache_dir`.

    Returns:
        The path to the cached file.
    """
    import logging

    logger = logging.getLogger(__name__)
    destination = cache_dir() / filename
    sidecar = destination.with_name(destination.name + ".sha256")

    if not destination.exists():
        logger.info("Downloading %s -> %s", url, destination)
        # Stream to a temp file, then atomically rename, so an interrupted
        # download never leaves a truncated file masquerading as complete.
        tmp = destination.with_name(destination.name + ".part")
        request = urllib.request.Request(
            url, headers={"User-Agent": _DOWNLOAD_USER_AGENT}
        )
        with urllib.request.urlopen(request) as response, tmp.open("wb") as out:  # noqa: S310 - trusted federalreserve.gov / hub URLs
            while True:
                chunk = response.read(_DOWNLOAD_CHUNK_BYTES)
                if not chunk:
                    break
                out.write(chunk)
        tmp.replace(destination)

    digest = _sha256_of_file(destination)
    sidecar.write_text(digest + "\n")
    logger.info("SHA-256(%s) = %s", filename, digest)
    return destination
