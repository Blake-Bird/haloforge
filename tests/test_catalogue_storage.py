import json

import h5py
import numpy as np
import pyarrow.parquet as pq

from config.defaults import DEFAULT_PARAMS
from engine.gadget_snapshot import load_gadget4_dm_snapshot
from engine.halo_catalogue import find_fof_halos
from engine.simulation_manifest import write_snapshot_manifest
from state import catalogue_storage


def _snapshot(tmp_path):
    path = tmp_path / "snapshot.hdf5"
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
        positions = np.column_stack((np.linspace(10, 90, 20), np.full(20, 100.0), np.full(20, 100.0)))
        dm.create_dataset("Coordinates", data=positions)
        dm.create_dataset("Velocities", data=np.zeros((20, 3)))
        dm.create_dataset("ParticleIDs", data=np.arange(20))
    return load_gadget4_dm_snapshot(path)


def _run():
    return {
        "run_id": "linear-1",
        "reproducibility_hash": "exact-linear-run",
        "integrity_status": {"state": "verified"},
    }


def test_saves_full_catalogue_with_schema_hashes_and_provenance(tmp_path, monkeypatch):
    monkeypatch.setattr(catalogue_storage, "CATALOGUE_DIR", tmp_path / "catalogues")
    snapshot = _snapshot(tmp_path)
    manifest_path = write_snapshot_manifest(snapshot, _run())
    manifest = json.loads(manifest_path.read_text())
    catalogue = find_fof_halos(
        snapshot.positions_mpc_h,
        snapshot.velocities_raw,
        snapshot.box_size_mpc_h,
        snapshot.particle_mass_msun_h,
        snapshot.redshift,
        omega_m=snapshot.omega_m,
    )

    stored = catalogue_storage.save_fof_catalogue(catalogue, snapshot, manifest)

    assert stored.hdf5_path.is_file()
    assert stored.parquet_path.is_file()
    record = json.loads(stored.manifest_path.read_text())
    assert record["schema_version"] == catalogue_storage.CATALOGUE_SCHEMA_VERSION
    assert record["finder"]["mass_definition"] == "FOF b=0.2 group mass"
    assert record["files"]["catalogue.hdf5"]
    table = pq.read_table(stored.parquet_path)
    assert table.num_rows == len(catalogue.halos)
    parquet_metadata = table.schema.metadata
    assert parquet_metadata[b"haloforge.box_size_mpc_h"] == b"10000.0"
    assert parquet_metadata[b"haloforge.redshift"] == b"0.0"
    assert parquet_metadata[b"haloforge.position_unit"] == b"comoving h^-1 Mpc"
    with h5py.File(stored.hdf5_path) as handle:
        assert handle["Header"].attrs["schema_version"] == catalogue_storage.CATALOGUE_SCHEMA_VERSION
