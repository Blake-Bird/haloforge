import h5py
import numpy as np
import pytest

from engine.snapshot_sequence import (
    load_snapshot_sequence,
    render_snapshot_movie,
    snapshot_sequence_manifest,
)


def _write_snapshot(path, *, redshift, box=20.0, ids=(1, 2), omega_m=0.3):
    with h5py.File(path, "w") as handle:
        header = handle.create_group("Header")
        header.attrs["BoxSize"] = box
        header.attrs["Time"] = 1 / (1 + redshift)
        header.attrs["Redshift"] = redshift
        header.attrs["Omega0"] = omega_m
        header.attrs["HubbleParam"] = 0.7
        header.attrs["MassTable"] = np.array([0, 1.5, 0, 0, 0, 0])
        part = handle.create_group("PartType1")
        part.create_dataset("Coordinates", data=[[1, 2, 3], [4, 5, 6]])
        part.create_dataset("Velocities", data=np.zeros((2, 3)))
        part.create_dataset("ParticleIDs", data=ids)


def test_sequence_keeps_exact_files_box_and_epoch_in_manifest(tmp_path):
    early, late = tmp_path / "snap_001.hdf5", tmp_path / "snap_002.hdf5"
    _write_snapshot(early, redshift=2.0)
    _write_snapshot(late, redshift=0.0)
    sequence = load_snapshot_sequence([early, late])
    manifest = snapshot_sequence_manifest(sequence)
    assert manifest["box_size_mpc_h"] == 20.0
    assert [frame["redshift"] for frame in manifest["frames"]] == [2.0, 0.0]
    assert all(len(frame["sha256"]) == 64 for frame in manifest["frames"])


def test_recorded_sequence_renders_downloadable_gif(tmp_path):
    early, late = tmp_path / "snap_001.hdf5", tmp_path / "snap_002.hdf5"
    _write_snapshot(early, redshift=2.0)
    _write_snapshot(late, redshift=0.0)
    movie = render_snapshot_movie(
        load_snapshot_sequence([early, late]),
        video_format="gif",
        cells=8,
        width=320,
        height=240,
    )
    assert movie.startswith(b"GIF89a")


def test_recorded_sequence_renders_downloadable_mp4(tmp_path):
    early, late = tmp_path / "snap_001.hdf5", tmp_path / "snap_002.hdf5"
    _write_snapshot(early, redshift=2.0)
    _write_snapshot(late, redshift=0.0)
    movie = render_snapshot_movie(
        load_snapshot_sequence([early, late]),
        video_format="mp4",
        cells=8,
        width=320,
        height=240,
    )
    assert movie[4:8] == b"ftyp"


@pytest.mark.parametrize(
    "changed,match",
    [
        ({"box": 25.0}, "box_size"),
        ({"ids": (2, 3)}, "particle IDs"),
        ({"omega_m": 0.35}, "omega_m"),
        ({"redshift": 3.0}, "high to low"),
    ],
)
def test_sequence_rejects_incompatible_or_misordered_frames(tmp_path, changed, match):
    early, late = tmp_path / "snap_001.hdf5", tmp_path / "snap_002.hdf5"
    _write_snapshot(early, redshift=2.0)
    _write_snapshot(late, **({"redshift": 0.0} | changed))
    with pytest.raises(ValueError, match=match):
        load_snapshot_sequence([early, late])


def test_sequence_accepts_gadget_group_order_reordering(tmp_path):
    early, late = tmp_path / "snap_001.hdf5", tmp_path / "snap_002.hdf5"
    _write_snapshot(early, redshift=2.0, ids=(1, 2))
    _write_snapshot(late, redshift=0.0, ids=(2, 1))
    assert len(load_snapshot_sequence([early, late])) == 2
