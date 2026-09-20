"""Compressed local cache for slow matter-power calculations."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from state.storage_policy import default_data_root
from engine.power_contract import validate_power_arrays

DATA_ROOT = default_data_root()
CACHE_DIR = DATA_ROOT / "cache"
ENGINE_VERSION = "strict-axiclass-v5-zero-ede-closed-background"
CACHE_SCHEMA_VERSION = "haloforge-power-cache-v2"
SLOW_KEYS = [
    "A_s",
    "n_s",
    "k_pivot",
    "H0",
    "Omega_m",
    "Omega_b",
    "Omega_k",
    "tau_reio",
    "Tcmb",
    "N_eff",
    "enable_ede",
    "f_EDE",
    "log10_a_c",
    "n_EDE",
    "k_min",
    "k_max",
    "k_points",
    "z_values",
    "single_z",
    "scf_parameters",
]
ARRAY_KEYS = {
    "k",
    "P",
    "P_by_z",
    "redshifts",
    "growth_class",
    "background_omega_m_by_z",
}


def slow_parameter_payload(params: dict) -> dict:
    return {"engine_version": ENGINE_VERSION, **{key: params[key] for key in SLOW_KEYS}}


def cache_key(params: dict) -> str:
    payload = json.dumps(
        slow_parameter_payload(params), sort_keys=True, default=str
    ).encode()
    return hashlib.sha256(payload).hexdigest()[:20]


def _metadata_from_result(result: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in result.items() if key not in ARRAY_KEYS}


def _integrity_digest(arrays: dict[str, np.ndarray], metadata: dict[str, Any]) -> str:
    """Hash scientific cache content independently of NPZ container details."""
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            metadata, sort_keys=True, separators=(",", ":"), default=str
        ).encode()
    )
    for key in sorted(arrays):
        value = np.ascontiguousarray(np.asarray(arrays[key]))
        digest.update(key.encode())
        digest.update(str(value.dtype).encode())
        digest.update(json.dumps(value.shape).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def cache_path(params: dict) -> Path:
    return CACHE_DIR / f"{cache_key(params)}.npz"


def load_cached_power(params: dict) -> dict[str, Any] | None:
    path = cache_path(params)
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata"].item()))
            if not isinstance(metadata, dict) or ARRAY_KEYS.intersection(metadata):
                raise ValueError("Cache metadata must not redefine scientific arrays")
            arrays = {key: data[key] for key in ARRAY_KEYS if key in data.files}
            envelope = metadata.pop("cache_integrity", None)
            if (
                not isinstance(envelope, dict)
                or envelope.get("schema_version") != CACHE_SCHEMA_VERSION
            ):
                raise ValueError("missing or incompatible cache integrity envelope")
            if (
                envelope.get("cache_key") != cache_key(params)
                or envelope.get("engine_version") != ENGINE_VERSION
            ):
                raise ValueError(
                    "cache envelope does not match the requested calculation"
                )
            if envelope.get("content_sha256") != _integrity_digest(arrays, metadata):
                raise ValueError("cache content checksum mismatch")
            validate_power_arrays(arrays, params)
            result = {**arrays, **metadata, "from_cache": True}
            return result
    except Exception:
        path.unlink(missing_ok=True)
        return None


def save_cached_power(params: dict, result: dict[str, Any]) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = cache_path(params)
    metadata = _metadata_from_result(result)
    arrays = {key: np.asarray(result[key]) for key in ARRAY_KEYS if key in result}
    try:
        validate_power_arrays(arrays, params)
    except ValueError as exc:
        raise ValueError(
            f"Refusing to cache an invalid matter power spectrum: {exc}"
        ) from exc
    content_sha256 = _integrity_digest(arrays, metadata)
    metadata["cache_integrity"] = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "cache_key": cache_key(params),
        "engine_version": ENGINE_VERSION,
        "content_sha256": content_sha256,
    }
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=CACHE_DIR
    )
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
    return sorted(
        CACHE_DIR.glob("*.npz"), key=lambda path: path.stat().st_mtime, reverse=True
    )


def cache_size_bytes() -> int:
    return sum(path.stat().st_size for path in list_cache_files())


def clear_cache() -> int:
    removed = 0
    for path in list_cache_files():
        path.unlink(missing_ok=True)
        removed += 1
    return removed
