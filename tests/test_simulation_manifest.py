import json

import h5py
import numpy as np
import pytest

from config.defaults import DEFAULT_PARAMS
from engine.gadget_snapshot import load_gadget4_dm_snapshot
from engine.simulation_manifest import (
    load_and_validate_snapshot_manifest,
    manifest_path_for_snapshot,
    write_snapshot_manifest,
)


def _snapshot(tmp_path):
    path = tmp_path / "snap_001.hdf5"
    with h5py.File(path, "w") as handle:
        header = handle.create_group("Header")
        header.attrs["BoxSize"] = 10_000.0
        header.attrs["Time"] = 1.0
        header.attrs["Redshift"] = 0.0
        header.attrs["Omega0"] = DEFAULT_PARAMS["Omega_m"]
        header.attrs["HubbleParam"] = DEFAULT_PARAMS["H0"] / 100.0
        masses = np.zeros(6)
        masses[1] = 1.0
        header.attrs["MassTable"] = masses
        dm = handle.create_group("PartType1")
        dm.create_dataset("Coordinates", data=[[1_000.0, 2_000.0, 3_000.0]])
        dm.create_dataset("Velocities", data=[[0.0, 0.0, 0.0]])
        dm.create_dataset("ParticleIDs", data=[1])
    return load_gadget4_dm_snapshot(path)


def _verified_run():
    return {
        "run_id": "linear-run-1",
        "reproducibility_hash": "abc123",
        "integrity_status": {"state": "verified"},
    }


def test_snapshot_manifest_binds_exact_snapshot_bytes_to_verified_run(tmp_path):
    snapshot = _snapshot(tmp_path)
    path = write_snapshot_manifest(snapshot, _verified_run())
    manifest = load_and_validate_snapshot_manifest(snapshot, _verified_run())

    assert path == manifest_path_for_snapshot(snapshot.source_path)
    assert manifest["linear_run"]["run_id"] == "linear-run-1"
    assert manifest["snapshot"]["particle_count"] == 1


def test_snapshot_manifest_rejects_changed_snapshot_or_different_run(tmp_path):
    snapshot = _snapshot(tmp_path)
    write_snapshot_manifest(snapshot, _verified_run())
    with pytest.raises(ValueError, match="different saved linear run"):
        load_and_validate_snapshot_manifest(
            snapshot, {**_verified_run(), "run_id": "another-run"}
        )

    sidecar = manifest_path_for_snapshot(snapshot.source_path)
    payload = json.loads(sidecar.read_text())
    payload["snapshot"]["sha256"] = "not-the-file"
    sidecar.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="SHA-256"):
        load_and_validate_snapshot_manifest(snapshot, _verified_run())


def test_split_snapshot_manifest_binds_every_shard(tmp_path):
    for index, particle_id in enumerate((1, 2)):
        path = tmp_path / f"snap_001.{index}.hdf5"
        with h5py.File(path, "w") as handle:
            header = handle.create_group("Header")
            header.attrs["BoxSize"] = 10.0
            header.attrs["Time"] = 1.0
            header.attrs["Redshift"] = 0.0
            header.attrs["Omega0"] = 0.3
            header.attrs["HubbleParam"] = 0.7
            header.attrs["MassTable"] = [0, 1, 0, 0, 0, 0]
            header.attrs["NumFilesPerSnapshot"] = 2
            header.attrs["NumPart_ThisFile"] = [0, 1, 0, 0, 0, 0]
            header.attrs["NumPart_Total"] = [0, 2, 0, 0, 0, 0]
            part = handle.create_group("PartType1")
            part.create_dataset("Coordinates", data=[[particle_id, 1, 1]])
            part.create_dataset("Velocities", data=[[0, 0, 0]])
            part.create_dataset("ParticleIDs", data=[particle_id])
    snapshot = load_gadget4_dm_snapshot(tmp_path / "snap_001.1.hdf5")
    write_snapshot_manifest(snapshot, _verified_run())
    manifest = load_and_validate_snapshot_manifest(snapshot, _verified_run())
    assert manifest["schema_version"] == "haloforge-snapshot-link-v2"
    assert len(manifest["snapshot"]["files"]) == 2
    with h5py.File(tmp_path / "snap_001.1.hdf5", "r+") as handle:
        handle["PartType1/Velocities"][0, 0] = 1
    with pytest.raises(ValueError, match="SHA-256"):
        load_and_validate_snapshot_manifest(snapshot, _verified_run())
