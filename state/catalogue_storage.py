"""Durable, versioned local storage for a HaloForge FOF catalogue.

The stored record is deliberately a catalogue product, not a validation
certificate.  Its manifest links the result to exact snapshot bytes and the
saved linear run, while retaining the limits of the in-app FOF implementation.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import h5py
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from engine.gadget_snapshot import GadgetSnapshotParticles
from engine.halo_catalogue import HaloCatalogue, catalogue_to_dataframe
from state.storage_policy import default_data_root, require_safe_persistent_storage


CATALOGUE_SCHEMA_VERSION = "haloforge-fof-catalogue-v1"
CATALOGUE_DIR = default_data_root() / "simulation_catalogues"


@dataclass(frozen=True)
class StoredCatalogue:
    catalogue_id: str
    directory: Path
    hdf5_path: Path
    parquet_path: Path
    manifest_path: Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_hdf5(path: Path, frame: pd.DataFrame, metadata: dict) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with h5py.File(temporary, "w") as handle:
            header = handle.create_group("Header")
            for key, value in metadata.items():
                header.attrs[key] = json.dumps(value) if isinstance(value, (dict, list)) else value
            catalogue = handle.create_group("FOF")
            for column in frame.columns:
                catalogue.create_dataset(column, data=frame[column].to_numpy())
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_parquet(path: Path, frame: pd.DataFrame, metadata: dict[str, str]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        table = pa.Table.from_pandas(frame, preserve_index=False)
        schema_metadata = dict(table.schema.metadata or {})
        schema_metadata.update({key.encode(): value.encode() for key, value in metadata.items()})
        pq.write_table(table.replace_schema_metadata(schema_metadata), temporary, compression="zstd")
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def save_fof_catalogue(
    catalogue: HaloCatalogue,
    snapshot: GadgetSnapshotParticles,
    snapshot_manifest: dict,
) -> StoredCatalogue:
    """Atomically store full FOF data as HDF5 and Parquet with a manifest.

    ``snapshot_manifest`` must be the result of
    ``load_and_validate_snapshot_manifest``.  The caller is therefore
    responsible for fail-closed validation of its exact snapshot/run binding.
    """
    require_safe_persistent_storage()
    if snapshot_manifest.get("schema_version") != "haloforge-snapshot-link-v1":
        raise ValueError("A verified HaloForge snapshot-link manifest is required")
    recorded_snapshot = snapshot_manifest.get("snapshot", {})
    if recorded_snapshot.get("particle_count") != int(snapshot.particle_ids.size):
        raise ValueError("Snapshot-link manifest does not match the loaded snapshot")
    if recorded_snapshot.get("redshift") != float(snapshot.redshift):
        raise ValueError("Snapshot-link manifest redshift does not match the loaded snapshot")
    if catalogue.redshift != snapshot.redshift:
        raise ValueError("FOF catalogue redshift does not match its snapshot")
    if catalogue.box_size_mpc_h != snapshot.box_size_mpc_h:
        raise ValueError("FOF catalogue box size does not match its snapshot")

    CATALOGUE_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    catalogue_id = f"fof_{stamp}_{recorded_snapshot['sha256'][:12]}_{uuid4().hex[:8]}"
    destination = CATALOGUE_DIR / catalogue_id
    destination.mkdir(mode=0o700)
    frame = catalogue_to_dataframe(catalogue)
    hdf5_path = destination / "catalogue.hdf5"
    parquet_path = destination / "catalogue.parquet"
    manifest_path = destination / "manifest.json"
    metadata = {
        "schema_version": CATALOGUE_SCHEMA_VERSION,
        "catalogue_id": catalogue_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scientific_status": "demonstration",
        "scope_limit": (
            "FOF groups are derived from an exact linked snapshot but are not an "
            "externally validated halo catalogue or an EDE simulation validation."
        ),
        "linear_run": snapshot_manifest["linear_run"],
        "snapshot": recorded_snapshot,
        "finder": {
            "name": "HaloForge periodic cKDTree FOF",
            "linking_length_b": catalogue.linking_length_b,
            "minimum_particles": catalogue.min_particles,
            "mass_definition": "FOF b=0.2 group mass",
            "unbinding": "not performed",
            "subfind": "not performed",
            "velocity_convention": "raw snapshot values; no physical conversion claimed",
        },
        "units": {
            "position": "comoving h^-1 Mpc",
            "fof_mass": "h^-1 solar mass",
            "velocity": "raw GADGET snapshot convention",
        },
        "record_count": int(len(frame)),
    }
    _atomic_hdf5(hdf5_path, frame, metadata)
    _atomic_parquet(
        parquet_path,
        frame,
        {
            "haloforge.schema_version": CATALOGUE_SCHEMA_VERSION,
            "haloforge.catalogue_id": catalogue_id,
            "haloforge.scientific_status": "demonstration",
            "haloforge.mass_definition": "FOF b=0.2 group mass",
            "haloforge.box_size_mpc_h": str(snapshot.box_size_mpc_h),
            "haloforge.redshift": str(snapshot.redshift),
            "haloforge.particle_mass_msun_h": str(snapshot.particle_mass_msun_h),
            "haloforge.position_unit": "comoving h^-1 Mpc",
            "haloforge.mass_unit": "h^-1 solar mass",
            "haloforge.snapshot_sha256": str(recorded_snapshot["sha256"]),
            "haloforge.linear_run_hash": str(snapshot_manifest["linear_run"]["reproducibility_hash"]),
        },
    )
    metadata["files"] = {
        "catalogue.hdf5": _sha256(hdf5_path),
        "catalogue.parquet": _sha256(parquet_path),
    }
    _atomic_json(manifest_path, metadata)
    return StoredCatalogue(catalogue_id, destination, hdf5_path, parquet_path, manifest_path)
