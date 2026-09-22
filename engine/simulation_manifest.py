"""Immutable linkage between a saved linear run and a GADGET-4 snapshot.

The manifest deliberately proves only what it records: a particular snapshot
byte stream was associated with a particular integrity-checked HaloForge run
and matching snapshot headers.  It does not establish that GADGET-4 dynamics,
EDE evolution, or the halo finder have been externally validated.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from engine.gadget_snapshot import GadgetSnapshotParticles


SIMULATION_MANIFEST_SCHEMA = "haloforge-snapshot-link-v1"
SPLIT_SIMULATION_MANIFEST_SCHEMA = "haloforge-snapshot-link-v2"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _snapshot_files(snapshot: GadgetSnapshotParticles) -> tuple[Path, ...]:
    return tuple(
        Path(path) for path in (snapshot.source_paths or (snapshot.source_path,))
    )


def _file_evidence(snapshot: GadgetSnapshotParticles) -> tuple[list[dict], str]:
    files = _snapshot_files(snapshot)
    if any(not path.is_file() for path in files):
        raise ValueError("Snapshot file no longer exists")
    evidence = [{"file_name": path.name, "sha256": _sha256(path)} for path in files]
    if len(files) == 1:
        return evidence, evidence[0]["sha256"]
    combined = hashlib.sha256(
        json.dumps(evidence, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return evidence, combined


def _require_integrity_checked_run(run: dict) -> None:
    if not run.get("run_id") or not run.get("reproducibility_hash"):
        raise ValueError("A saved run with a reproducibility hash is required")
    integrity = run.get("integrity_status", {})
    if integrity.get("state") != "verified":
        raise ValueError("The saved run must pass its integrity checks before binding")


def build_snapshot_manifest(snapshot: GadgetSnapshotParticles, run: dict) -> dict:
    """Build a serializable exact-byte snapshot association for one saved run."""
    _require_integrity_checked_run(run)
    source = Path(snapshot.source_path)
    evidence, digest = _file_evidence(snapshot)
    split = len(evidence) > 1
    return {
        "schema_version": SPLIT_SIMULATION_MANIFEST_SCHEMA
        if split
        else SIMULATION_MANIFEST_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scientific_status": "simulation_output",
        "scope_limit": (
            "This establishes an immutable file-to-saved-run association and matching "
            "headers only. It does not validate GADGET-4 dynamics, EDE treatment, "
            "initial conditions, halo finding, or empirical HMF calibration."
        ),
        "linear_run": {
            "run_id": str(run["run_id"]),
            "reproducibility_hash": str(run["reproducibility_hash"]),
            "integrity_state": str(run["integrity_status"]["state"]),
        },
        "snapshot": {
            "file_name": source.name,
            "sha256": digest,
            **({"files": evidence} if split else {}),
            "redshift": float(snapshot.redshift),
            "scale_factor": float(snapshot.scale_factor),
            "box_size_mpc_h": float(snapshot.box_size_mpc_h),
            "particle_mass_msun_h": float(snapshot.particle_mass_msun_h),
            "omega_m": float(snapshot.omega_m),
            "h": float(snapshot.h),
            "particle_count": int(snapshot.particle_ids.size),
        },
    }


def manifest_path_for_snapshot(snapshot_path: str | Path) -> Path:
    source = Path(snapshot_path)
    return source.with_name(source.name + ".haloforge-manifest.json")


def write_snapshot_manifest(snapshot: GadgetSnapshotParticles, run: dict) -> Path:
    """Atomically save a manifest beside the snapshot after explicit user action."""
    target = manifest_path_for_snapshot(snapshot.source_path)
    payload = build_snapshot_manifest(snapshot, run)
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, target)
    return target


def load_and_validate_snapshot_manifest(
    snapshot: GadgetSnapshotParticles, run: dict, path: str | Path | None = None
) -> dict:
    """Fail closed unless a sidecar manifest binds these exact bytes to this run."""
    _require_integrity_checked_run(run)
    manifest_path = (
        Path(path)
        if path is not None
        else manifest_path_for_snapshot(snapshot.source_path)
    )
    if not manifest_path.is_file():
        raise ValueError(
            "No immutable snapshot manifest was found beside this snapshot"
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Snapshot manifest is unreadable: {exc}") from exc
    schema = manifest.get("schema_version")
    if schema not in {SIMULATION_MANIFEST_SCHEMA, SPLIT_SIMULATION_MANIFEST_SCHEMA}:
        raise ValueError("Snapshot manifest schema is unsupported")
    linear = manifest.get("linear_run", {})
    if linear.get("run_id") != run["run_id"]:
        raise ValueError("Snapshot manifest belongs to a different saved linear run")
    if linear.get("reproducibility_hash") != run["reproducibility_hash"]:
        raise ValueError(
            "Snapshot manifest has a different linear-run reproducibility hash"
        )
    recorded = manifest.get("snapshot", {})
    evidence, digest = _file_evidence(snapshot)
    if len(evidence) > 1 and schema != SPLIT_SIMULATION_MANIFEST_SCHEMA:
        raise ValueError("Split snapshot requires a manifest binding every shard")
    if schema == SPLIT_SIMULATION_MANIFEST_SCHEMA and recorded.get("files") != evidence:
        raise ValueError("Snapshot shard bytes differ from the manifest SHA-256")
    if recorded.get("sha256") != digest:
        raise ValueError("Snapshot bytes differ from the manifest SHA-256")
    expected = {
        "redshift": snapshot.redshift,
        "scale_factor": snapshot.scale_factor,
        "box_size_mpc_h": snapshot.box_size_mpc_h,
        "particle_mass_msun_h": snapshot.particle_mass_msun_h,
        "omega_m": snapshot.omega_m,
        "h": snapshot.h,
    }
    for key, actual in expected.items():
        if not np.isclose(
            float(recorded.get(key, np.nan)), actual, rtol=0.0, atol=1e-10
        ):
            raise ValueError(
                f"Snapshot manifest {key} does not match the loaded header"
            )
    if int(recorded.get("particle_count", -1)) != snapshot.particle_ids.size:
        raise ValueError(
            "Snapshot manifest particle count does not match the loaded snapshot"
        )
    return manifest
