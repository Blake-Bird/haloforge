"""Strict reader for collisionless GADGET-4 single and split HDF5 snapshots.

This boundary deliberately reads a snapshot; it does not infer that a file is
an evolved production simulation.  Callers must retain the execution manifest
and declare the source before using a catalogue for scientific comparison.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

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
    source_paths: tuple[str, ...] = ()


def _load_one_dm_snapshot(
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
                if (
                    masses.shape != (positions.shape[0],)
                    or np.any(~np.isfinite(masses))
                    or np.any(masses <= 0)
                ):
                    raise ValueError("PartType1 Masses are invalid")
                if not np.allclose(masses, masses[0], rtol=1e-10, atol=0):
                    raise ValueError(
                        "Variable PartType1 masses are unsupported by the equal-mass FOF and density workflow"
                    )
                particle_mass = float(masses[0] * 1.0e10)
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
        raise ValueError(
            "PartType1 coordinate, velocity, and ID arrays are incompatible"
        )
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
    if not np.isclose(redshift, 1.0 / scale_factor - 1.0, rtol=1e-5, atol=1e-5):
        raise ValueError("Snapshot Time and Redshift headers are inconsistent")
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
        source_paths=(str(source.resolve()),),
    )


def load_gadget4_dm_snapshot(
    path: str | Path, *, omega_m: float | None = None, h: float | None = None
) -> GadgetSnapshotParticles:
    """Read a complete single or split DM snapshot, rejecting partial shard sets.

    A path to any shard selects its whole snapshot. A missing unsuffixed path
    may also select a present ``.0.hdf5`` shard. Native split headers must
    declare both the shard count and total PartType1 count.
    """
    import h5py

    source = Path(path)
    match = re.fullmatch(r"(.+)\.(\d+)\.hdf5", source.name)
    base = source.with_name(match.group(1)) if match else source.with_suffix("")
    first = base.with_name(base.name + ".0.hdf5")
    if match or (not source.is_file() and first.is_file()):
        if not first.is_file():
            raise ValueError("Split snapshot is missing shard 0")
        with h5py.File(first, "r") as handle:
            if "Header" not in handle:
                raise ValueError("Split snapshot shard 0 has no Header")
            attrs = handle["Header"].attrs
            if "NumFilesPerSnapshot" not in attrs or "NumPart_Total" not in attrs:
                raise ValueError(
                    "Split snapshot is missing shard count or total particle count"
                )
            count = int(attrs["NumFilesPerSnapshot"])
            if count < 2 or count > 4096:
                raise ValueError("Split snapshot has an invalid shard count")
            if match and int(match.group(2)) >= count:
                raise ValueError(
                    "Selected snapshot shard is outside the declared shard count"
                )
            low = np.asarray(attrs["NumPart_Total"])
            high = np.asarray(attrs.get("NumPart_Total_HighWord", np.zeros_like(low)))
            if low.ndim != 1 or low.size < 2 or high.shape != low.shape:
                raise ValueError("Split snapshot has invalid total particle counts")
            total_dm = int(low[1]) + (int(high[1]) << 32)
        files = tuple(
            base.with_name(f"{base.name}.{index}.hdf5") for index in range(count)
        )
        if any(not item.is_file() for item in files):
            raise ValueError("Split snapshot is missing one or more shards")
    else:
        files = (source,)
        total_dm = None
        if source.is_file():
            with h5py.File(source, "r") as handle:
                if "Header" in handle:
                    declared = int(handle["Header"].attrs.get("NumFilesPerSnapshot", 1))
                    if declared != 1:
                        raise ValueError(
                            "Snapshot declares multiple files but only one was selected"
                        )

    parts = [_load_one_dm_snapshot(item, omega_m=omega_m, h=h) for item in files]
    reference = parts[0]
    for item, part in zip(files, parts):
        if not (
            np.isclose(part.box_size_mpc_h, reference.box_size_mpc_h, rtol=0, atol=1e-8)
            and np.isclose(part.scale_factor, reference.scale_factor, rtol=0, atol=1e-8)
            and np.isclose(part.redshift, reference.redshift, rtol=0, atol=1e-8)
            and np.isclose(part.omega_m, reference.omega_m, rtol=0, atol=1e-8)
            and np.isclose(part.h, reference.h, rtol=0, atol=1e-8)
            and np.isclose(
                part.particle_mass_msun_h,
                reference.particle_mass_msun_h,
                rtol=0,
                atol=1e-3,
            )
        ):
            raise ValueError(
                "Snapshot shards disagree on box, epoch, cosmology, or mass"
            )
        if total_dm is not None:
            with h5py.File(item, "r") as handle:
                attrs = handle["Header"].attrs
                if int(attrs.get("NumFilesPerSnapshot", -1)) != len(files):
                    raise ValueError("Snapshot shards disagree on shard count")
                low = np.asarray(attrs.get("NumPart_Total", []))
                high = np.asarray(
                    attrs.get("NumPart_Total_HighWord", np.zeros_like(low))
                )
                if (
                    low.size < 2
                    or high.shape != low.shape
                    or int(low[1]) + (int(high[1]) << 32) != total_dm
                ):
                    raise ValueError("Snapshot shards disagree on total particle count")
                this_file = np.asarray(attrs.get("NumPart_ThisFile", []))
                if this_file.size < 2 or int(this_file[1]) != part.particle_ids.size:
                    raise ValueError(
                        "Snapshot shard particle count disagrees with its header"
                    )

    positions = np.concatenate([part.positions_mpc_h for part in parts])
    velocities = np.concatenate([part.velocities_raw for part in parts])
    ids = np.concatenate([part.particle_ids for part in parts])
    if total_dm is not None and ids.size != total_dm:
        raise ValueError("Split snapshot particle count disagrees with total header")
    if np.unique(ids).size != ids.size:
        raise ValueError("Snapshot PartType1 IDs are not unique across shards")
    return GadgetSnapshotParticles(
        source_path=str(files[0].resolve()),
        positions_mpc_h=positions,
        velocities_raw=velocities,
        particle_ids=ids,
        box_size_mpc_h=reference.box_size_mpc_h,
        particle_mass_msun_h=reference.particle_mass_msun_h,
        redshift=reference.redshift,
        scale_factor=reference.scale_factor,
        omega_m=reference.omega_m,
        h=reference.h,
        source_paths=tuple(str(item.resolve()) for item in files),
    )
