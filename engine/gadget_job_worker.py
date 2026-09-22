"""Detached one-run GADGET worker with durable completion verification."""

from __future__ import annotations

import hashlib
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import h5py

from engine.gadget4_adapter import GADGET4_PINNED_COMMIT
from engine.gadget_run import _save, read_run
from engine.gadget_snapshot import load_gadget4_dm_snapshot


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main(root: Path) -> None:
    record = read_run(root)
    if record["state"] not in {"queued", "validated"}:
        raise ValueError("Worker found a run in the wrong state")
    record["state"] = "running"
    record["started_at"] = datetime.now(timezone.utc).isoformat()
    _save(root, record)
    log_path = root / "logs" / "gadget.log"
    try:
        with log_path.open("w") as log:
            result = subprocess.run(
                ["/usr/local/bin/Gadget4", "config/param.txt"],
                cwd=root,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if result.returncode:
            raise RuntimeError(f"GADGET-4 exited with code {result.returncode}")
        if "Simulation ends." not in log_path.read_text(errors="replace"):
            raise RuntimeError("GADGET-4 has no terminal completion marker")
        files = sorted((root / "snapshots").glob("snap_*.hdf5"))
        if len(files) != len(record["requested_redshifts"]):
            raise RuntimeError("GADGET-4 produced an unexpected number of snapshots")
        snapshots = []
        for index, path in enumerate(files):
            snapshot = load_gadget4_dm_snapshot(
                path, omega_m=record["omega_m"], h=record["h"]
            )
            if snapshot.particle_ids.size != int(record["particles_per_dim"]) ** 3:
                raise RuntimeError("GADGET snapshot has an incomplete particle count")
            with h5py.File(path) as handle:
                commit = handle["Header"].attrs.get("Git_commit", b"")
                if isinstance(commit, bytes):
                    commit = commit.decode()
                if commit != GADGET4_PINNED_COMMIT:
                    raise RuntimeError(
                        "GADGET snapshot came from a different executable revision"
                    )
            requested = sorted(record["requested_redshifts"], reverse=True)[index]
            if abs(snapshot.redshift - requested) > 0.1:
                raise RuntimeError(
                    "GADGET snapshot is too far from its planned redshift"
                )
            catalogue = root / "snapshots" / f"fof_subhalo_tab_{index:03d}.hdf5"
            if not catalogue.is_file():
                raise RuntimeError("GADGET FoF/Subfind catalogue is missing")
            snapshots.append(
                {
                    "path": str(path),
                    "sha256": _sha256(path),
                    "redshift": snapshot.redshift,
                    "scale_factor": snapshot.scale_factor,
                    "catalogue_path": str(catalogue),
                    "catalogue_sha256": _sha256(catalogue),
                }
            )
        record.update(
            state="completed",
            completed_at=datetime.now(timezone.utc).isoformat(),
            snapshots=snapshots,
        )
    except Exception as exc:
        record.update(
            state="failed",
            failed_at=datetime.now(timezone.utc).isoformat(),
            error=str(exc),
        )
    _save(root, record)


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve())
