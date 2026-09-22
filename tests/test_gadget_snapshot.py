import numpy as np
import h5py
import pytest

from config.defaults import DEFAULT_PARAMS
from engine.gadget_snapshot import load_gadget4_dm_snapshot


def test_load_gadget4_dm_snapshot_round_trip(tmp_path):
    positions = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    velocities = np.array([[10.0, 0.0, 0.0], [0.0, 20.0, 0.0]])
    ids = np.array([10, 11], dtype=np.int64)
    snapshot_path = tmp_path / "ics.hdf5"
    with h5py.File(snapshot_path, "w") as handle:
        header = handle.create_group("Header")
        header.attrs["BoxSize"] = 20.0
        header.attrs["Time"] = 1.0 / 50.0
        header.attrs["Redshift"] = 49.0
        header.attrs["Omega0"] = DEFAULT_PARAMS["Omega_m"]
        header.attrs["HubbleParam"] = DEFAULT_PARAMS["H0"] / 100.0
        masses = np.zeros(6)
        masses[1] = 1.5
        header.attrs["MassTable"] = masses
        part = handle.create_group("PartType1")
        part.create_dataset("Coordinates", data=positions)
        part.create_dataset("Velocities", data=velocities)
        part.create_dataset("ParticleIDs", data=ids)

    snapshot = load_gadget4_dm_snapshot(snapshot_path)
    assert np.allclose(snapshot.positions_mpc_h, positions)
    assert np.allclose(snapshot.velocities_raw, velocities)
    assert snapshot.particle_ids.tolist() == [10, 11]
    assert snapshot.box_size_mpc_h == 20.0
    assert snapshot.redshift == 49.0
    assert snapshot.omega_m == DEFAULT_PARAMS["Omega_m"]


def test_snapshot_rejects_variable_particle_masses_and_inconsistent_epoch(tmp_path):
    path = tmp_path / "variable.hdf5"
    with h5py.File(path, "w") as handle:
        header = handle.create_group("Header")
        header.attrs["BoxSize"] = 20.0
        header.attrs["Time"] = 1.0
        header.attrs["Redshift"] = 0.0
        header.attrs["Omega0"] = 0.3
        header.attrs["HubbleParam"] = 0.7
        header.attrs["MassTable"] = np.zeros(6)
        part = handle.create_group("PartType1")
        part.create_dataset("Coordinates", data=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        part.create_dataset("Velocities", data=np.zeros((2, 3)))
        part.create_dataset("ParticleIDs", data=[1, 2])
        part.create_dataset("Masses", data=[1.0, 2.0])
    with pytest.raises(ValueError, match="Variable PartType1 masses"):
        load_gadget4_dm_snapshot(path)
    with h5py.File(path, "r+") as handle:
        handle["PartType1/Masses"][:] = [1.0, 1.0]
        handle["Header"].attrs["Redshift"] = 1.0
    with pytest.raises(ValueError, match="Time and Redshift"):
        load_gadget4_dm_snapshot(path)


def _write_split_shard(path, ids, *, total=4, redshift=0.0):
    with h5py.File(path, "w") as handle:
        header = handle.create_group("Header")
        header.attrs["BoxSize"] = 20.0
        header.attrs["Time"] = 1.0
        header.attrs["Redshift"] = redshift
        header.attrs["Omega0"] = 0.3
        header.attrs["HubbleParam"] = 0.7
        header.attrs["MassTable"] = [0.0, 1.5, 0, 0, 0, 0]
        header.attrs["NumFilesPerSnapshot"] = 2
        header.attrs["NumPart_Total"] = [0, total, 0, 0, 0, 0]
        header.attrs["NumPart_ThisFile"] = [0, len(ids), 0, 0, 0, 0]
        part = handle.create_group("PartType1")
        part.create_dataset(
            "Coordinates", data=np.array([[float(i), 1, 1] for i in ids])
        )
        part.create_dataset("Velocities", data=np.zeros((len(ids), 3)))
        part.create_dataset("ParticleIDs", data=ids)


def test_split_snapshot_loads_all_shards_from_any_shard(tmp_path):
    first = tmp_path / "snap_000.0.hdf5"
    second = tmp_path / "snap_000.1.hdf5"
    _write_split_shard(first, [1, 2])
    _write_split_shard(second, [3, 4])
    snapshot = load_gadget4_dm_snapshot(second)
    assert snapshot.particle_ids.tolist() == [1, 2, 3, 4]
    assert snapshot.source_paths == (str(first), str(second))
    assert load_gadget4_dm_snapshot(tmp_path / "snap_000.hdf5").particle_ids.size == 4


def test_split_snapshot_rejects_partial_duplicate_and_mismatched_shards(tmp_path):
    first = tmp_path / "snap_000.0.hdf5"
    second = tmp_path / "snap_000.1.hdf5"
    _write_split_shard(first, [1, 2])
    with pytest.raises(ValueError, match="missing one or more shards"):
        load_gadget4_dm_snapshot(first)
    _write_split_shard(second, [2, 3])
    with pytest.raises(ValueError, match="not unique across shards"):
        load_gadget4_dm_snapshot(first)
    _write_split_shard(second, [3, 4], redshift=1.0)
    with pytest.raises(ValueError, match="Time and Redshift"):
        load_gadget4_dm_snapshot(first)
    _write_split_shard(second, [3, 4], total=5)
    with pytest.raises(ValueError, match="disagree on total particle count"):
        load_gadget4_dm_snapshot(first)
