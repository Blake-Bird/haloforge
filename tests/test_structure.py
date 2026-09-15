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
