"""Measured snapshot imagery must preserve the file's geometry and particles."""

import numpy as np
import pytest

from engine.gadget_snapshot import GadgetSnapshotParticles
from engine.snapshot_visuals import (
    density_slice_figure,
    particle_cube_figure,
    particle_density_grid,
    projected_overdensity,
)


def _snapshot():
    return GadgetSnapshotParticles(
        source_path="/fixture/snap_001.hdf5",
        positions_mpc_h=np.array([[0.1, 0.1, 0.1], [0.2, 0.2, 0.2], [9.9, 9.9, 9.9]]),
        velocities_raw=np.zeros((3, 3)),
        particle_ids=np.array([1, 2, 3]),
        box_size_mpc_h=10.0,
        particle_mass_msun_h=1e10,
        redshift=0.0,
        scale_factor=1.0,
        omega_m=0.3,
        h=0.7,
    )


def test_density_projection_conserves_particles_and_uses_recorded_box():
    snap = _snapshot()
    grid = particle_density_grid(snap, 8)
    assert grid.sum() == 3
    assert grid[0, 0, 0] == 2
    assert grid[-1, -1, -1] == 1
    projection = projected_overdensity(snap, 8)
    assert projection.shape == (8, 8)
    assert projection.mean() == pytest.approx(0.0)
    figure = density_slice_figure(snap, cells=8)
    assert figure.layout.xaxis.range == (0, 10)
    assert np.isfinite(np.asarray(figure.data[0].z)).all()


def test_particle_cube_uses_recorded_coordinates_with_bounded_sampling():
    snap = _snapshot()
    figure = particle_cube_figure(snap, max_points=2)
    assert len(figure.data[0].x) == 2
    assert figure.layout.scene.xaxis.range == (0, 10)
    assert set(figure.data[0].x).issubset(set(snap.positions_mpc_h[:, 0]))


def test_density_view_rejects_invalid_resolution():
    with pytest.raises(ValueError, match="cells"):
        particle_density_grid(_snapshot(), 7)
