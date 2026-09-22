"""Exercise the same detached GADGET controller used by the UI."""

from __future__ import annotations

import json
import hashlib
import time
from pathlib import Path

import numpy as np

from config.defaults import DEFAULT_PARAMS
from engine.gadget_catalogue import load_gadget4_group_catalogue
from engine.gadget_run import create_run, log_tail, read_run, start_run
from engine.gadget_snapshot import load_gadget4_dm_snapshot
from engine.ic_generator import generate_zeldovich_particles
from engine.rockstar_adapter import (
    load_rockstar_ascii_catalogue,
    prepare_rockstar_hdf5_snapshot,
    run_rockstar_single_snapshot,
)
from engine.snapshot_sequence import load_snapshot_sequence, render_snapshot_movie
from engine.gadget4_adapter import GADGET4_PINNED_COMMIT
from engine.rockstar_adapter import ROCKSTAR_PINNED_COMMIT


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main() -> None:
    params = {**DEFAULT_PARAMS, "enable_ede": False}
    z_start = 10.0
    k = np.geomspace(1e-3, 10.0, 512)
    p = 20.0 * (k / 0.1) ** -1.0
    omega_m = float(params["Omega_m"])
    e_rate = np.sqrt(omega_m * (1 + z_start) ** 3 + 1 - omega_m)
    positions, velocities, ids, report = generate_zeldovich_particles(
        20.0, 16, k, p, z_start, params, seed=7,
        spectrum_redshift=z_start, growth_rate=1.0, expansion_rate_E=e_rate,
    )
    if not report.checks_passed:
        raise RuntimeError(report.summary)
    root = create_run(
        positions=positions, velocities=velocities, ids=ids,
        box_size_mpc_h=20.0, start_redshift=z_start,
        output_redshifts=[5.0, 2.0, 0.0], params=params, seed=7,
        source_run_id="synthetic-controller-smoke",
        source_hash="synthetic-fixture-not-science",
    )
    start_run(root)
    deadline = time.monotonic() + 240
    while time.monotonic() < deadline:
        record = read_run(root)
        if record["state"] in {"completed", "failed", "interrupted"}:
            if record["state"] != "completed":
                raise RuntimeError(f"{record['state']}: {record.get('error')}\n{log_tail(root)}")
            final = record["snapshots"][-1]
            snapshot = load_gadget4_dm_snapshot(
                final["path"], omega_m=record["omega_m"], h=record["h"]
            )
            gadget_halos = load_gadget4_group_catalogue(final["catalogue_path"])
            converted = prepare_rockstar_hdf5_snapshot(
                snapshot, root / "catalogues" / "rockstar_input.hdf5"
            )
            ascii_path = run_rockstar_single_snapshot(
                converted, root / "catalogues" / "rockstar_final",
                force_res_mpc_h=20.0 / 960.0,
            )
            rockstar = load_rockstar_ascii_catalogue(ascii_path, expected=converted)
            sequence = load_snapshot_sequence(
                [item["path"] for item in record["snapshots"]],
                omega_m=record["omega_m"], h=record["h"],
            )
            movie = render_snapshot_movie(
                sequence, video_format="gif", cells=16, width=320, height=240
            )
            (root / "exports" / "structure.gif").write_bytes(movie)
            print(json.dumps({
                "run": str(root),
                "fixture_kind": "synthetic-power-software-smoke-not-science-validation",
                "gadget_commit": GADGET4_PINNED_COMMIT,
                "rockstar_commit": ROCKSTAR_PINNED_COMMIT,
                "gadget_binary_sha256": _sha256("/usr/local/bin/Gadget4"),
                "rockstar_binary_sha256": _sha256("/usr/local/bin/rockstar"),
                "final_snapshot_sha256": final["sha256"],
                "rockstar_catalogue_sha256": rockstar.sha256,
                "movie_sha256": hashlib.sha256(movie).hexdigest(),
                "redshifts": [item["redshift"] for item in record["snapshots"]],
                "gadget_fof_hosts": len(gadget_halos.group_mass_msun_h),
                "rockstar_halos": rockstar.count,
                "rockstar_periodic": rockstar.periodic,
                "movie_bytes": len(movie),
            }))
            return
        time.sleep(1)
    raise RuntimeError(f"GADGET controller timed out\n{log_tail(root)}")


if __name__ == "__main__":
    main()
