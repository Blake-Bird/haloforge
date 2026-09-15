"""Versioned, machine-readable provenance for reproducible HaloForge runs."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path


SCHEMA_VERSION = "haloforge-run-v2"
APP_VERSION = "0.2.0-dev"
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_revision() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
            capture_output=True,
            timeout=2,
            check=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unavailable"


def _axiclass_commit() -> str:
    path = Path("/opt/AXICLASS_COMMIT")
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return os.environ.get("HALOFORGE_AXICLASS_COMMIT", "unavailable")


def _package_versions() -> dict[str, str]:
    result = {}
    for package in ("streamlit", "plotly", "numpy", "pandas", "scipy", "sympy"):
        try:
            result[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            result[package] = "unavailable"
    return result


def software_provenance() -> dict:
    requirements = PROJECT_ROOT / "requirements.txt"
    requirements_hash = (
        file_sha256(requirements) if requirements.is_file() else "unavailable"
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "app_version": APP_VERSION,
        "git_revision": _git_revision(),
        "image_digest": os.environ.get("HALOFORGE_IMAGE_DIGEST", "unavailable"),
        "axiclass_commit": _axiclass_commit(),
        "python": sys.version.split()[0],
        "platform": sys.platform,
        "package_versions": _package_versions(),
        "requirements_sha256": requirements_hash,
    }


def reproducibility_hash(params: dict, class_settings: dict, provenance: dict) -> str:
    """Stable ID for the declared calculation, excluding timestamps and arrays."""
    payload = {
        "schema_version": provenance["schema_version"],
        "app_version": provenance["app_version"],
        "git_revision": provenance["git_revision"],
        "axiclass_commit": provenance["axiclass_commit"],
        "params": params,
        "class_settings": class_settings,
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), default=str
    ).encode()
    return _sha256_bytes(encoded)


def run_provenance(params: dict, class_settings: dict) -> dict:
    provenance = software_provenance()
    provenance["created_at"] = datetime.now(timezone.utc).isoformat()
    provenance["reproducibility_hash"] = reproducibility_hash(
        params, class_settings, provenance
    )
    return provenance
