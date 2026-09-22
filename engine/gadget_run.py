"""Bounded local GADGET-4 job folder and persistent single-job controller."""

from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from engine.gadget4_adapter import (
    GADGET4_PINNED_COMMIT,
    generate_gadget4_parameter_file,
    generate_output_times_file,
)
from engine.ic_generator import export_gadget_hdf5_ic


SCHEMA = "haloforge-gadget4-local-run-v1"
RUN_ROOT = (
    Path(os.environ.get("HALOFORGE_DATA_DIR", "/var/lib/haloforge")) / "simulations"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _save(root: Path, record: dict[str, Any]) -> None:
    temporary = root / "run.json.tmp"
    temporary.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, root / "run.json")


def read_run(root: str | Path) -> dict[str, Any]:
    path = Path(root).resolve()
    if path.parent != RUN_ROOT.resolve() or not (path / "run.json").is_file():
        raise ValueError("Run folder is outside HaloForge's local simulation directory")
    record = json.loads((path / "run.json").read_text())
    if record.get("schema_version") != SCHEMA or record.get("run_id") != path.name:
        raise ValueError("Run manifest is invalid")
    return record


def list_runs(*, limit: int = 50) -> list[Path]:
    """List saved local runs so a browser restart does not hide their outputs."""
    if not RUN_ROOT.is_dir():
        return []
    candidates = sorted(
        (item for item in RUN_ROOT.iterdir() if item.is_dir() and (item / "run.json").is_file()),
        key=lambda item: (item / "run.json").stat().st_mtime,
        reverse=True,
    )
    return candidates[:limit]


def verify_completed_run(root: str | Path) -> dict[str, Any]:
    """Check recorded output bytes before reopening a finished run."""
    record = read_run(root)
    if record["state"] != "completed":
        raise ValueError("Run has not completed")
    for item in record.get("snapshots", []):
        for path_key, hash_key in (
            ("path", "sha256"), ("catalogue_path", "catalogue_sha256")
        ):
            source = Path(item[path_key]).resolve()
            if not source.is_relative_to(Path(root).resolve()) or _sha256(source) != item[hash_key]:
                raise ValueError("Completed run output changed or escaped its run folder")
    if not record.get("snapshots"):
        raise ValueError("Completed run has no verified snapshots")
    return record


def create_run(
    *,
    positions: np.ndarray,
    velocities: np.ndarray,
    ids: np.ndarray,
    box_size_mpc_h: float,
    start_redshift: float,
    output_redshifts: list[float],
    params: dict[str, Any],
    seed: int,
    source_run_id: str,
    source_hash: str,
) -> Path:
    """Reserve one bounded, DM-only run and save exact generated inputs."""
    if params.get("enable_ede") or abs(float(params.get("Omega_k", 0))) > 1e-12:
        raise ValueError("GADGET execution currently supports flat LCDM only")
    if not source_run_id or not source_hash:
        raise ValueError("An integrity-checked saved CLASS run is required")
    n = int(round(len(positions) ** (1.0 / 3.0)))
    if (
        n not in (16, 32)
        or positions.shape != (n**3, 3)
        or velocities.shape != positions.shape
        or ids.shape != (n**3,)
    ):
        raise ValueError("The current GADGET build supports bounded 16³ or 32³ runs")
    if not np.isfinite(box_size_mpc_h) or not 10 <= box_size_mpc_h <= 200:
        raise ValueError("The local run box must be 10–200 Mpc/h")
    if not np.isfinite(start_redshift) or not 10 <= start_redshift <= 99:
        raise ValueError("The local run start redshift must be 10–99")
    times = generate_output_times_file(output_redshifts, start_redshift=start_redshift)
    if len(output_redshifts) > 8 or 0.0 not in output_redshifts:
        raise ValueError("Use at most eight output redshifts and include z=0")
    root = RUN_ROOT / uuid.uuid4().hex
    root.mkdir(parents=True, exist_ok=False)
    for name in (
        "inputs",
        "config",
        "logs",
        "snapshots",
        "catalogues",
        "figures",
        "exports",
    ):
        (root / name).mkdir()
    ic = root / "inputs" / "ics.hdf5"
    export_gadget_hdf5_ic(
        str(ic), positions, velocities, ids, box_size_mpc_h, start_redshift, params
    )
    output_times = root / "config" / "output_times.txt"
    output_times.write_text(times)
    param = root / "config" / "param.txt"
    body = generate_gadget4_parameter_file(
        box_size_mpc_h,
        n,
        "snapshots",
        output_redshifts,
        params,
        start_redshift=start_redshift,
        ic_filename="inputs/ics.hdf5",
    ).replace(
        "OutputListFilename        output_times.txt",
        "OutputListFilename        config/output_times.txt",
    )
    param.write_text(body)
    record = {
        "schema_version": SCHEMA,
        "run_id": root.name,
        "state": "validated",
        "created_at": _now(),
        "source_run_id": source_run_id,
        "source_hash": source_hash,
        "gadget_commit": GADGET4_PINNED_COMMIT,
        "box_size_mpc_h": box_size_mpc_h,
        "particles_per_dim": n,
        "start_redshift": start_redshift,
        "requested_redshifts": output_redshifts,
        "omega_m": float(params["Omega_m"]),
        "h": float(params["H0"]) / 100.0,
        "seed": int(seed),
        "input_hashes": {
            "inputs/ics.hdf5": _sha256(ic),
            "config/param.txt": _sha256(param),
            "config/output_times.txt": _sha256(output_times),
        },
    }
    _save(root, record)
    return root


def start_run(root: str | Path) -> dict[str, Any]:
    root = Path(root).resolve()
    record = read_run(root)
    if record["state"] != "validated":
        raise ValueError("Only a validated draft can be started")
    if not Path("/usr/local/bin/Gadget4").is_file():
        raise RuntimeError("GADGET-4 executable is missing from this installation")
    for name, expected in record["input_hashes"].items():
        if _sha256(root / name) != expected:
            raise ValueError(f"Run input {name} changed after validation")
    record.update(state="queued", queued_at=_now())
    _save(root, record)
    with (root / "logs" / "controller.log").open("w") as log:
        process = subprocess.Popen(
            [sys.executable, "-m", "engine.gadget_job_worker", str(root)],
            cwd=root,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
            env={
                **os.environ,
                "PYTHONPATH": "/app:" + os.environ.get("PYTHONPATH", ""),
            },
        )
    record = read_run(root)
    record["worker_pid"] = process.pid
    _save(root, record)
    return record


def stop_run(root: str | Path) -> dict[str, Any]:
    root = Path(root).resolve()
    record = read_run(root)
    if record["state"] not in {"queued", "running"}:
        raise ValueError("This run is not active")
    os.killpg(int(record["worker_pid"]), signal.SIGTERM)
    record.update(state="interrupted", stopped_at=_now())
    _save(root, record)
    return record


def log_tail(root: str | Path, *, lines: int = 80) -> str:
    root = Path(root).resolve()
    read_run(root)
    path = root / "logs" / "gadget.log"
    if not path.is_file():
        path = root / "logs" / "controller.log"
    if not path.is_file():
        return "Waiting for the worker to start."
    return "\n".join(path.read_text(errors="replace").splitlines()[-lines:])
