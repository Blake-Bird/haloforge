"""Strict reader for collisionless GADGET-4 HDF5 snapshot particles.

This boundary deliberately reads a snapshot; it does not infer that a file is
an evolved production simulation.  Callers must retain the execution manifest
and declare the source before using a catalogue for scientific comparison.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class GadgetSnapshotParticles:
    """DM particle data and immutable header facts from one HDF5 snapshot."""

    source_path: str
    positions_mpc_h: np.ndarray
    velocities_raw: np.ndarray
    particle_ids: np.ndarray
    box_size_mpc_h: float
    particle_mass_msun_h: float
    redshift: float
    scale_factor: float
    omega_m: float
    h: float


def load_gadget4_dm_snapshot(
    path: str | Path, *, omega_m: float | None = None, h: float | None = None
) -> GadgetSnapshotParticles:
    """Load and validate ``PartType1`` from a GADGET-4 HDF5 snapshot.

    HaloForge-generated GADGET plans set ``UnitLength_in_cm`` to one Mpc/h,
    so coordinates and ``BoxSize`` are already in comoving Mpc/h. Velocities
    are retained as stored because the exact
    conversion convention must be taken from the corresponding GADGET run
    configuration; FOF does not require that conversion.
    """
    source = Path(path)
    if not source.is_file():
        raise ValueError("GADGET-4 snapshot file does not exist")
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - package is a declared dependency
        raise RuntimeError("h5py is required to read GADGET-4 HDF5 snapshots") from exc

    try:
        with h5py.File(source, "r") as handle:
            if "Header" not in handle or "PartType1" not in handle:
                raise ValueError("Snapshot needs Header and PartType1 groups")
            header = handle["Header"].attrs
            group = handle["PartType1"]
            for name in ("Coordinates", "Velocities", "ParticleIDs"):
                if name not in group:
                    raise ValueError(f"PartType1 is missing {name}")
            positions = np.asarray(group["Coordinates"], dtype=float)
            velocities = np.asarray(group["Velocities"], dtype=float)
            ids = np.asarray(group["ParticleIDs"])
            box_size = float(header["BoxSize"])
            scale_factor = float(header["Time"])
            redshift = float(header["Redshift"])
            # Upstream GADGET-4 output may intentionally omit cosmological
            # metadata.  It must then be supplied by the immutable run binding,
            # never guessed from defaults or a current UI draft.
            header_omega_m = header.get("Omega0")
            header_h = header.get("HubbleParam")
            snapshot_omega_m = (
                float(header_omega_m) if header_omega_m is not None else omega_m
            )
            snapshot_h = float(header_h) if header_h is not None else h
            if snapshot_omega_m is None or snapshot_h is None:
                raise ValueError(
                    "Snapshot omits Omega0/HubbleParam; supply omega_m and h from its verified saved-run binding"
                )
            mass_table = np.asarray(header["MassTable"], dtype=float)
            if mass_table.ndim != 1 or mass_table.size < 2 or mass_table[1] <= 0:
                if "Masses" not in group:
                    raise ValueError(
                        "PartType1 has no positive MassTable entry or per-particle Masses"
                    )
                masses = np.asarray(group["Masses"], dtype=float)
                if masses.shape != (positions.shape[0],) or np.any(masses <= 0):
                    raise ValueError("PartType1 Masses are invalid")
                particle_mass = float(np.median(masses) * 1.0e10)
            else:
                particle_mass = float(mass_table[1] * 1.0e10)
    except OSError as exc:
        raise ValueError(f"Snapshot could not be read as HDF5: {exc}") from exc

    if (
        positions.ndim != 2
        or positions.shape[1] != 3
        or velocities.shape != positions.shape
        or ids.ndim != 1
        or ids.shape[0] != positions.shape[0]
        or positions.shape[0] == 0
    ):
        raise ValueError("PartType1 coordinate, velocity, and ID arrays are incompatible")
    if (
        not np.isfinite(box_size)
        or box_size <= 0
        or not np.isfinite(particle_mass)
        or particle_mass <= 0
        or not np.isfinite(scale_factor)
        or not 0 < scale_factor <= 1
        or not np.isfinite(redshift)
        or redshift < 0
        or not np.isfinite(snapshot_omega_m)
        or not 0 < snapshot_omega_m <= 1
        or not np.isfinite(snapshot_h)
        or snapshot_h <= 0
    ):
        raise ValueError("Snapshot Header contains invalid cosmological values")
    if np.any(~np.isfinite(positions)) or np.any(~np.isfinite(velocities)):
        raise ValueError("Snapshot contains non-finite particle values")
    if np.any(positions < 0) or np.any(positions >= box_size):
        raise ValueError("Snapshot coordinates fall outside its periodic box")
    if len(np.unique(ids)) != ids.size:
        raise ValueError("Snapshot PartType1 IDs are not unique")

    return GadgetSnapshotParticles(
        source_path=str(source.resolve()),
        positions_mpc_h=positions,
        velocities_raw=velocities,
        particle_ids=ids,
        box_size_mpc_h=box_size,
        particle_mass_msun_h=particle_mass,
        redshift=redshift,
        scale_factor=scale_factor,
        omega_m=float(snapshot_omega_m),
        h=float(snapshot_h),
    )
