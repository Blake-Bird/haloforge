"""Compressed local cache for slow matter-power calculations."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

DATA_ROOT = Path(os.environ.get("HALOFORGE_DATA_DIR", "data")).expanduser().resolve()
CACHE_DIR = DATA_ROOT / "cache"
ENGINE_VERSION = "strict-axiclass-v3"
SLOW_KEYS = [
    "A_s", "n_s", "k_pivot", "H0", "Omega_m", "Omega_b", "Omega_k", "tau_reio",
    "Tcmb", "N_eff", "enable_ede", "f_EDE", "log10_a_c", "n_EDE", "k_min",
    "k_max", "k_points", "z_values", "single_z", "scf_parameters",
]
ARRAY_KEYS = {"k", "P", "P_by_z", "redshifts", "growth_class"}


def slow_parameter_payload(params: dict) -> dict:
    return {"engine_version": ENGINE_VERSION, **{key: params[key] for key in SLOW_KEYS}}


def cache_key(params: dict) -> str:
    payload = json.dumps(slow_parameter_payload(params), sort_keys=True, default=str).encode()
    return hashlib.sha256(payload).hexdigest()[:20]


def _metadata_from_result(result: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in result.items() if key not in ARRAY_KEYS}


def cache_path(params: dict) -> Path:
    return CACHE_DIR / f"{cache_key(params)}.npz"


def load_cached_power(params: dict) -> dict[str, Any] | None:
    path = cache_path(params)
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata"].item()))
            arrays = {key: data[key] for key in ARRAY_KEYS if key in data.files}
            result = {**arrays, **metadata, "from_cache": True}
            if "k" not in result or "P" not in result:
                return None
            result.setdefault("P_by_z", np.asarray([result["P"]]))
            result.setdefault("redshifts", np.asarray([0.0]))
            result.setdefault("growth_class", np.asarray([1.0]))
            return result
    except Exception:
        path.unlink(missing_ok=True)
        return None


def save_cached_power(params: dict, result: dict[str, Any]) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = cache_path(params)
    metadata = _metadata_from_result(result)
    arrays = {key: np.asarray(result[key]) for key in ARRAY_KEYS if key in result}
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=CACHE_DIR)
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            np.savez_compressed(handle, **arrays, metadata=json.dumps(metadata))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    return path


def list_cache_files() -> list[Path]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return sorted(CACHE_DIR.glob("*.npz"), key=lambda path: path.stat().st_mtime, reverse=True)


def cache_size_bytes() -> int:
    return sum(path.stat().st_size for path in list_cache_files())


def clear_cache() -> int:
    removed = 0
    for path in list_cache_files():
        path.unlink(missing_ok=True)
        removed += 1
    return removed
