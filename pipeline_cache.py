"""
pipeline_cache.py — SHA-256 Input-Hash Caching for the Ember Pipeline
======================================================================
Design pattern : Repository pattern + dataclasses
Formatting     : RUFF-compliant (ruff check + ruff format)

How it works:
    Before running a step, hash:
      - The script file
      - All declared input files / directories
      - The relevant params from params.yaml

    Hash stored in .cache/pipeline/cache.json
    Match  → step skipped
    Differ → step runs, new hash saved on success

Usage:
    from pipeline_cache import StepCache, StepDefinition

    cache = StepCache()
    step  = StepDefinition(
        name    = "eda",
        script  = "src/01_eda.py",
        deps    = ["data/raw/ember.csv"],
        params  = ["data", "countries"],
        outputs = ["outputs/eda/ember_filtered.csv"],
    )

    if cache.is_cached(step):
        log.info("Skipping eda — inputs unchanged")
    else:
        run_step(...)
        cache.mark_done(step)
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import logging
from pathlib import Path

import yaml

log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
CACHE_DIR = Path(".cache/pipeline")
LOCK_FILE = CACHE_DIR / "cache.json"
PARAMS_FILE = Path("params.yaml")


# ── Data classes ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class StepDefinition:
    """
    Immutable definition of a pipeline step and its cache dependencies.

    Attributes
    ----------
    name    : Unique step identifier (e.g. "eda", "modeling")
    script  : Path to the .py script file
    deps    : Input files/dirs that affect the cache hash
    params  : Top-level keys from params.yaml that affect the hash
    outputs : Files that must exist for cache to be considered valid
    """

    name: str
    script: str
    deps: list[str] = field(default_factory=list)
    params: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)


@dataclass
class CacheEntry:
    """A single cached step — stores the fingerprint hash."""

    step: str
    fingerprint: str

    def is_valid(self, current: str) -> bool:
        return self.fingerprint == current


# ── Hash helpers ──────────────────────────────────────────────────────────────


def _hash_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """SHA-256 of a single file — streamed for large CSVs."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while data := f.read(chunk_size):
            h.update(data)
    return h.hexdigest()


def _hash_dir(path: Path) -> str:
    """SHA-256 over all files in a directory (sorted for determinism)."""
    h = hashlib.sha256()
    for fp in sorted(path.rglob("*")):
        if fp.is_file():
            h.update(fp.name.encode())
            h.update(_hash_file(fp).encode())
    return h.hexdigest()


def _hash_path(p: str | Path) -> str:
    """Hash a file or directory. Returns 'missing' if path doesn't exist."""
    path = Path(p)
    if not path.exists():
        return "missing"
    return _hash_dir(path) if path.is_dir() else _hash_file(path)


def _hash_params(keys: list[str]) -> str:
    """Hash specific top-level keys from params.yaml."""
    if not PARAMS_FILE.exists():
        return "no-params"
    with PARAMS_FILE.open() as f:
        all_params = yaml.safe_load(f) or {}
    selected = {k: all_params.get(k) for k in keys}
    blob = json.dumps(selected, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()


def _hash_script(script: str) -> str:
    """Hash the script file so code changes invalidate the cache."""
    path = Path(script)
    return _hash_file(path) if path.exists() else "no-script"


def _compute_fingerprint(step: StepDefinition) -> str:
    """
    Combine hashes of script + deps + params into one fingerprint.
    Any change to any input produces a different fingerprint.
    """
    h = hashlib.sha256()
    h.update(_hash_script(step.script).encode())
    for dep in sorted(step.deps):
        h.update(f"{dep}:{_hash_path(dep)}".encode())
    h.update(_hash_params(step.params).encode())
    return h.hexdigest()


# ── Cache repository ──────────────────────────────────────────────────────────


class StepCacheRepository:
    """
    Low-level JSON file repository for cache entries.
    Follows the Repository pattern — separates storage from business logic.
    """

    def __init__(self, lock_file: Path = LOCK_FILE) -> None:
        self._lock_file = lock_file
        self._store: dict[str, str] = {}
        self._load()

    def _load(self) -> None:
        self._lock_file.parent.mkdir(parents=True, exist_ok=True)
        if self._lock_file.exists():
            with self._lock_file.open() as f:
                self._store = json.load(f)

    def _save(self) -> None:
        with self._lock_file.open("w") as f:
            json.dump(self._store, f, indent=2)

    def get(self, step_name: str) -> CacheEntry | None:
        fp = self._store.get(step_name)
        return CacheEntry(step=step_name, fingerprint=fp) if fp else None

    def put(self, entry: CacheEntry) -> None:
        self._store[entry.step] = entry.fingerprint
        self._save()

    def delete(self, step_name: str) -> None:
        self._store.pop(step_name, None)
        self._save()

    def clear(self) -> None:
        self._store.clear()
        self._save()

    def all_entries(self) -> dict[str, str]:
        return {k: v[:12] for k, v in self._store.items()}


# ── Public API ────────────────────────────────────────────────────────────────


class StepCache:
    """
    High-level cache manager for pipeline steps.

    Uses StepCacheRepository for storage and StepDefinition for step metadata.

    Example
    -------
    cache = StepCache()

    step = StepDefinition(
        name    = "eda",
        script  = "src/01_eda.py",
        deps    = ["data/raw/ember.csv"],
        params  = ["data", "countries"],
        outputs = ["outputs/eda/ember_filtered.csv"],
    )

    if cache.is_cached(step):
        log.info("Skipping eda")
    else:
        run_step(...)
        cache.mark_done(step)
    """

    def __init__(self, cache_dir: Path = CACHE_DIR) -> None:
        self._repo = StepCacheRepository(cache_dir / "cache.json")

    def is_cached(self, step: StepDefinition) -> bool:
        """
        Return True if:
          - All declared outputs exist on disk, AND
          - The stored fingerprint matches the current input hash.
        """
        # Output files must exist — if missing, always re-run
        for out in step.outputs:
            if not Path(out).exists():
                log.debug("Cache MISS (%s): output missing → %s", step.name, out)
                return False

        current = _compute_fingerprint(step)
        entry = self._repo.get(step.name)

        if entry and entry.is_valid(current):
            log.debug("Cache HIT  (%s): %s", step.name, current[:12])
            return True

        stored = entry.fingerprint[:12] if entry else "none"
        log.debug(
            "Cache MISS (%s): %s → %s",
            step.name,
            stored,
            current[:12],
        )
        return False

    def mark_done(self, step: StepDefinition) -> None:
        """Save the current fingerprint after a successful run."""
        fingerprint = _compute_fingerprint(step)
        self._repo.put(CacheEntry(step=step.name, fingerprint=fingerprint))
        log.debug("Cache SAVE (%s): %s", step.name, fingerprint[:12])

    def invalidate(self, step_name: str | None = None) -> None:
        """Invalidate one step or the entire cache."""
        if step_name:
            self._repo.delete(step_name)
            log.info("Cache invalidated: %s", step_name)
        else:
            self._repo.clear()
            log.info("Entire cache invalidated.")

    def status(self) -> dict[str, str]:
        """Return {step_name: short_hash} for all cached steps."""
        return self._repo.all_entries()
