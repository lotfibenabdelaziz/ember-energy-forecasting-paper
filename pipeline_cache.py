"""
pipeline_cache.py — Input-hash caching for the Ember pipeline.

How it works:
  Before running a step, we hash:
    - The script file itself
    - All declared input files / directories
    - The relevant params from params.yaml

  The hash is stored in  .cache/pipeline/<step>.hash
  If it matches → step is skipped.
  If it differs  → step runs, new hash saved on success.

Usage (inside pipeline.py):
    from pipeline_cache import StepCache
    cache = StepCache()
    if cache.is_cached("eda", deps=[csv_path], params=["data", "countries"]):
        log.info("── CACHED: eda (inputs unchanged)")
        continue
    run_step(...)
    cache.save("eda", deps=[csv_path], params=["data", "countries"])
"""

import hashlib
import json
import logging
import os
from pathlib import Path

import yaml

log = logging.getLogger(__name__)

CACHE_DIR = Path(os.getenv("CACHE_DIR", ".cache/pipeline"))
PARAMS_FILE = Path("params.yaml")
LOCK_FILE = CACHE_DIR / "cache.json"  # stores {step: hash}


# ── Helpers ───────────────────────────────────────────────────────────────────


def _hash_file(path: Path, chunk: int = 1 << 20) -> str:
    """SHA-256 of a single file (streamed — safe for large CSVs)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while data := f.read(chunk):
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


def _hash_path(p: str | Path) -> str | None:
    """Hash a file or directory; return None if it doesn't exist."""
    p = Path(p)
    if not p.exists():
        return None
    return _hash_dir(p) if p.is_dir() else _hash_file(p)


def _hash_params(keys: list[str]) -> str:
    """Hash specific top-level keys from params.yaml."""
    if not PARAMS_FILE.exists():
        return "no-params"
    with open(PARAMS_FILE) as f:
        all_params = yaml.safe_load(f) or {}
    selected = {k: all_params.get(k) for k in keys}
    blob = json.dumps(selected, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()


def _hash_script(script: str) -> str:
    """Hash the .py script so code changes invalidate the cache."""
    p = Path(script)
    return _hash_file(p) if p.exists() else "no-script"


def _compute_step_hash(
    script: str,
    deps: list[str],
    params: list[str],
) -> str:
    """Combine hashes of script + deps + params into one fingerprint."""
    h = hashlib.sha256()
    h.update(_hash_script(script).encode())
    for dep in sorted(deps):
        dep_hash = _hash_path(dep) or "missing"
        h.update(f"{dep}:{dep_hash}".encode())
    h.update(_hash_params(params).encode())
    return h.hexdigest()


# ── StepCache ─────────────────────────────────────────────────────────────────


class StepCache:
    """
    Manages a simple JSON lock-file that stores per-step input hashes.

    Example
    -------
    cache = StepCache()

    if cache.is_cached("eda", script="src/01_eda.py",
                        deps=["data/raw/ember.csv"],
                        params=["data", "countries"]):
        log.info("Skipping eda — inputs unchanged")
    else:
        run_step(...)
        cache.mark_done("eda", script="src/01_eda.py",
                        deps=["data/raw/ember.csv"],
                        params=["data", "countries"])
    """

    def __init__(self, cache_dir: Path = CACHE_DIR) -> None:
        self.cache_dir = cache_dir
        self.lock_file = cache_dir / "cache.json"
        self._store: dict[str, str] = {}
        self._load()

    # ── private ───────────────────────────────────────────────────────────────

    def _load(self) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        if self.lock_file.exists():
            with open(self.lock_file) as f:
                self._store = json.load(f)

    def _save(self) -> None:
        with open(self.lock_file, "w") as f:
            json.dump(self._store, f, indent=2)

    def _fingerprint(
        self,
        step: str,
        script: str,
        deps: list[str],
        params: list[str],
    ) -> str:
        return _compute_step_hash(script, deps, params)

    # ── public ────────────────────────────────────────────────────────────────

    def is_cached(
        self,
        step: str,
        script: str,
        deps: list[str],
        params: list[str],
        outputs: list[str] | None = None,
    ) -> bool:
        """
        Return True if:
          - The stored hash matches the current input hash, AND
          - All declared output files/dirs exist.
        """
        # If any output is missing, always re-run regardless of hash
        if outputs:
            for out in outputs:
                if not Path(out).exists():
                    log.debug("Cache MISS (%s): output missing → %s", step, out)
                    return False

        current = self._fingerprint(step, script, deps, params)
        stored = self._store.get(step)

        if stored == current:
            log.debug("Cache HIT  (%s): hash=%s", step, current[:12])
            return True

        log.debug(
            "Cache MISS (%s): hash changed %s → %s", step, (stored or "none")[:12], current[:12]
        )
        return False

    def mark_done(
        self,
        step: str,
        script: str,
        deps: list[str],
        params: list[str],
    ) -> None:
        """Save the current fingerprint after a successful run."""
        self._store[step] = self._fingerprint(step, script, deps, params)
        self._save()
        log.debug("Cache SAVE (%s): hash=%s", step, self._store[step][:12])

    def invalidate(self, step: str | None = None) -> None:
        """Invalidate one step or the entire cache."""
        if step:
            self._store.pop(step, None)
            log.info("Cache invalidated for step: %s", step)
        else:
            self._store.clear()
            log.info("Entire pipeline cache invalidated.")
        self._save()

    def status(self) -> dict[str, str]:
        """Return {step: short_hash} for all cached steps."""
        return {k: v[:12] for k, v in self._store.items()}
