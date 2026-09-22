"""Small executable acceptance fixture; synthetic spectrum, not a science run."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np

from config.defaults import DEFAULT_PARAMS
from engine.gadget4_adapter import (
    generate_gadget4_parameter_file,
    generate_output_times_file,
)
from engine.gadget_snapshot import load_gadget4_dm_snapshot
from engine.ic_generator import export_gadget_hdf5_ic, generate_zeldovich_particles


def main() -> None:
    root = Path("/var/lib/haloforge/smoke")
    root.mkdir(parents=True, exist_ok=True)
    params = {**DEFAULT_PARAMS, "enable_ede": False}
    z_start = 9.0
    k = np.geomspace(1e-3, 10.0, 512)
    p = 20.0 * (k / 0.1) ** -1.0
    omega_m = float(params["Omega_m"])
    e_rate = np.sqrt(omega_m * (1 + z_start) ** 3 + 1 - omega_m)
    positions, velocities, ids, report = generate_zeldovich_particles(
        20.0,
        16,
        k,
        p,
        z_start,
        params,
        seed=7,
        spectrum_redshift=z_start,
        growth_rate=1.0,
        expansion_rate_E=e_rate,
    )
    if not report.checks_passed:
        raise RuntimeError(report.summary)
    export_gadget_hdf5_ic(
        str(root / "ics.hdf5"), positions, velocities, ids, 20.0, z_start, params
    )
    (root / "output").mkdir(exist_ok=True)
    (root / "output_times.txt").write_text(
        generate_output_times_file([5.0, 2.0, 0.0], start_redshift=z_start)
    )
    (root / "param.txt").write_text(
        generate_gadget4_parameter_file(
            20.0, 16, "output", [5.0, 2.0, 0.0], params, start_redshift=z_start
        )
    )
    with (root / "gadget.log").open("w") as log:
        result = subprocess.run(
            ["/usr/local/bin/Gadget4", "param.txt"],
            cwd=root,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
            timeout=600,
        )
    if result.returncode:
        raise RuntimeError(
            f"GADGET-4 exited {result.returncode}; see {root / 'gadget.log'}"
        )
    snapshots = sorted((root / "output").glob("snap_*.hdf5"))
    results = [
        load_gadget4_dm_snapshot(path, omega_m=omega_m, h=float(params["H0"]) / 100.0)
        for path in snapshots
    ]
    if len(results) != 3 or [round(item.redshift) for item in results] != [5, 2, 0]:
        raise RuntimeError("GADGET-4 did not create the three planned snapshots")
    print(
        json.dumps(
            {"snapshots": len(results), "redshifts": [x.redshift for x in results]}
        )
    )


if __name__ == "__main__":
    main()
