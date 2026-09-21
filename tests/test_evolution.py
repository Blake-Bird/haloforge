import numpy as np
import pytest

from engine.evolution import evolution_manifest, evolution_redshifts


@pytest.mark.parametrize("sampling", ["uniform_z", "uniform_a", "uniform_log_a"])
def test_evolution_grid_has_exact_endpoints_and_descends(sampling):
    grid = evolution_redshifts(20, 0, 5, sampling)
    assert grid[0] == pytest.approx(20)
    assert grid[-1] == pytest.approx(0)
    assert np.all(np.diff(grid) < 0)


def test_scale_factor_sampling_is_uniform_in_scale_factor():
    grid = evolution_redshifts(20, 0, 5, "uniform_a")
    np.testing.assert_allclose(np.diff(1 / (1 + grid)), np.diff(1 / (1 + grid))[0])


def test_custom_grid_is_preserved_without_nearest_frame_substitution():
    grid = evolution_redshifts(20, 0, 2, "custom", custom_redshifts=[20, 3.7, 0])
    assert grid.tolist() == [20, 3.7, 0]
    assert evolution_manifest(grid, "custom")["redshifts"] == [20.0, 3.7, 0.0]
