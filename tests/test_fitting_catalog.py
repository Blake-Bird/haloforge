import numpy as np
import pytest

from engine.fitting_functions import FITTING_NAMES, fitting_values, f_press_schechter, f_sheth_tormen


def test_all_published_fits_are_positive_and_finite():
    sigma = np.logspace(-0.7, 0.5, 64)
    for name in FITTING_NAMES:
        values = fitting_values(sigma, 1.686, name, z=0.5, omega_m_z=0.59, delta_halo=200.0)
        assert values.shape == sigma.shape
        assert np.all(np.isfinite(values)), name
        assert np.all(values >= 0.0), name
        assert np.any(values > 0.0), name


def test_press_schechter_matches_closed_form():
    sigma = np.array([0.5, 1.0, 2.0])
    nu = 1.686 / sigma
    expected = np.sqrt(2 / np.pi) * nu * np.exp(-nu**2 / 2)
    np.testing.assert_allclose(f_press_schechter(sigma), expected, rtol=1e-13)


def test_sheth_tormen_normalization_parameters():
    sigma = np.array([0.8, 1.2])
    got = f_sheth_tormen(sigma)
    assert got[0] == pytest.approx(0.1619997779967776)
    assert got[1] == pytest.approx(0.2879055779409089)


def test_unknown_fit_is_not_silently_mapped():
    with pytest.raises(ValueError):
        fitting_values(np.array([1.0]), 1.686, "invented fit")
