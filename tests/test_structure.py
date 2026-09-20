import numpy as np
import pytest

from engine.structure import gaussian_field_slice, shared_fourier_seed


def test_shared_fourier_seed_is_deterministic_and_changes_with_seed():
    modes_a, kk_a = shared_fourier_seed(200, 16, 42)
    modes_b, kk_b = shared_fourier_seed(200, 16, 42)
    modes_c, _ = shared_fourier_seed(200, 16, 43)
    np.testing.assert_array_equal(modes_a, modes_b)
    np.testing.assert_array_equal(kk_a, kk_b)
    assert not np.array_equal(modes_a, modes_c)


def test_linear_field_slice_is_reproducible_and_bounded_to_input_power_grid():
    modes, kk = shared_fourier_seed(200, 16, 42)
    k = np.geomspace(0.02, 2, 32)
    power = k**-1.2
    first = gaussian_field_slice(k, power, 200, 16, 2, modes, kk)
    second = gaussian_field_slice(k, power, 200, 16, 2, modes, kk)
    np.testing.assert_array_equal(first, second)
    assert first.shape == (16, 16)
    assert np.all(np.isfinite(first))
    assert np.std(first) > 0
    with pytest.raises(ValueError, match="strictly increasing"):
        gaussian_field_slice(k[::-1], power, 200, 16, 2, modes, kk)


@pytest.mark.parametrize("grid", [1, 3.5, np.nan, np.inf, True])
def test_structure_rejects_invalid_grid_sizes(grid):
    with pytest.raises(ValueError, match="Grid size"):
        shared_fourier_seed(200, grid, 42)


@pytest.mark.parametrize("seed", [-1, 1.5, True])
def test_structure_rejects_invalid_seeds(seed):
    with pytest.raises(ValueError, match="Seed"):
        shared_fourier_seed(200, 8, seed)


@pytest.mark.parametrize("target", ["k", "power", "modes", "kk"])
@pytest.mark.parametrize("value", [np.nan, np.inf])
def test_structure_rejects_nonfinite_inputs(target, value):
    modes, kk = shared_fourier_seed(200, 8, 42)
    arrays = {
        "k": np.geomspace(0.01, 2, 32),
        "power": np.ones(32),
        "modes": modes,
        "kk": kk,
    }
    arrays[target].flat[1] = value
    with pytest.raises(ValueError):
        gaussian_field_slice(
            arrays["k"], arrays["power"], 200, 8, 2, arrays["modes"], arrays["kk"]
        )


def test_structure_rejects_wavenumbers_from_another_box():
    modes, kk = shared_fourier_seed(200, 8, 42)
    with pytest.raises(ValueError, match="wavenumbers do not match"):
        gaussian_field_slice([0.001, 10], [1, 1], 100, 8, 0, modes, kk)


@pytest.mark.parametrize("n", [7, 8])
def test_white_spectrum_has_physical_cell_volume_normalization(n):
    box, seed, power = 200.0, 42, 1000.0
    modes, kk = shared_fourier_seed(box, n, seed)
    field = gaussian_field_slice([1e-6, 10], [power, power], box, n, 0, modes, kk)
    white = np.random.default_rng(seed).normal(size=(n, n, n))
    white -= white.mean()
    # For a constant physical P, variance per cell is P / V_cell.
    expected = white[:, :, n // 2] * np.sqrt(power / (box / n) ** 3)
    np.testing.assert_allclose(field, expected, rtol=1e-12, atol=1e-15)


def test_structure_power_amplitude_scales_density_as_square_root():
    modes, kk = shared_fourier_seed(200, 16, 42)
    k = np.geomspace(0.01, 10, 64)
    power = k**-1.2
    field = gaussian_field_slice(k, power, 200, 16, 2, modes, kk)
    doubled = gaussian_field_slice(k, 4 * power, 200, 16, 2, modes, kk)
    np.testing.assert_allclose(doubled, 2 * field, rtol=1e-12, atol=1e-15)
