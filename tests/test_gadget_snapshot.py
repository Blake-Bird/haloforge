import numpy as np
import h5py

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
