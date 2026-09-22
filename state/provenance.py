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


SCHEMA_VERSION = "haloforge-run-v3"
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
    for package in (
        "streamlit",
        "plotly",
        "numpy",
        "pandas",
        "scipy",
        "sympy",
        "classy",
        "pyarrow",
        "reportlab",
        "pypdf",
        "kaleido",
    ):
        try:
            result[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            result[package] = "unavailable"
    return result


def solver_binding_provenance(solver_execution: dict | None = None) -> dict:
    """Record the exact locally observed Python/binding identity when available."""
    execution = solver_execution or {}
    worker_python = str(execution.get("worker_python", sys.executable))
    binding_path = str(execution.get("classy_path", ""))
    if not binding_path:
        try:
            import classy  # type: ignore

            binding_path = str(getattr(classy, "__file__", ""))
        except Exception:
            binding_path = ""
    path = Path(binding_path) if binding_path else None
    return {
        "worker_python": worker_python,
        "worker_python_sha256": (
            file_sha256(Path(worker_python))
            if Path(worker_python).is_file()
            else "unavailable"
        ),
        "classy_binding_path": binding_path or "unavailable",
        "classy_binding_sha256": (
            file_sha256(path) if path and path.is_file() else "unavailable"
        ),
    }


def source_tree_sha256(root: Path = PROJECT_ROOT) -> str:
    """Identify shipped source, including local edits and builds without Git.

    Only application inputs are included: user experiments, caches, tests,
    and generated artifacts cannot affect or disclose themselves in this hash.
    Paths and file digests are length-delimited to avoid ambiguous concatenation.
    """
    files = [root / "app.py", root / "requirements.txt"]
    for directory in ("config", "content", "engine", "state", "assets"):
        files.extend(
            path
            for path in (root / directory).rglob("*")
            if path.is_file() and path.suffix in {".py", ".css", ".svg", ".json"}
        )
    digest = hashlib.sha256()
    for path in sorted(files):
        if not path.is_file():
            continue
        name = path.relative_to(root).as_posix().encode()
        digest.update(len(name).to_bytes(8, "big"))
        digest.update(name)
        digest.update(bytes.fromhex(file_sha256(path)))
    return digest.hexdigest()


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
        "source_tree_sha256": source_tree_sha256(),
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
    # v1/v2 hashes remain verifiable under their original, narrower identity.
    if provenance["schema_version"] == "haloforge-run-v3":
        payload["software"] = {
            key: provenance.get(key, "unavailable")
            for key in (
                "source_tree_sha256",
                "requirements_sha256",
                "package_versions",
                "python",
                "platform",
                "image_digest",
            )
        }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), default=str
    ).encode()
    return _sha256_bytes(encoded)


def run_provenance(
    params: dict, class_settings: dict, solver_execution: dict | None = None
) -> dict:
    provenance = software_provenance()
    provenance["solver_binding"] = solver_binding_provenance(solver_execution)
    provenance["created_at"] = datetime.now(timezone.utc).isoformat()
    provenance["reproducibility_hash"] = reproducibility_hash(
        params, class_settings, provenance
    )
    return provenance
