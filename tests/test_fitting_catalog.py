import numpy as np
import pytest

from engine.fitting_functions import (
    FITTING_NAMES,
    fitting_values,
    f_press_schechter,
    f_sheth_tormen,
    f_reed03,
    f_reed07,
    f_watson_so13,
)


def test_all_published_fits_are_positive_and_finite():
    sigma = np.logspace(-0.7, 0.5, 64)
    for name in FITTING_NAMES:
        values = fitting_values(
            sigma, 1.686, name, z=0.5, omega_m_z=0.59, delta_halo=200.0, neff=-2.0
        )
        assert values.shape == sigma.shape
        assert np.all(np.isfinite(values)), name
        assert np.all(values >= 0.0), name
        assert np.any(values > 0.0), name


def test_press_schechter_matches_closed_form():
    sigma = np.array([0.5, 1.0, 2.0])
    nu = 1.686 / sigma
    expected = np.sqrt(2 / np.pi) * nu * np.exp(-(nu**2) / 2)
    np.testing.assert_allclose(f_press_schechter(sigma), expected, rtol=1e-13)


def test_sheth_tormen_normalization_parameters():
    sigma = np.array([0.8, 1.2])
    got = f_sheth_tormen(sigma)
    assert got[0] == pytest.approx(0.1619997779967776)
    assert got[1] == pytest.approx(0.2879055779409089)


def test_unknown_fit_is_not_silently_mapped():
    with pytest.raises(ValueError):
        fitting_values(np.array([1.0]), 1.686, "invented fit")


def test_reed03_correction_has_one_power_of_sigma():
    sigma = np.array([0.35, 0.5, 1, 2, 5])
    expected_ratio = np.exp(-0.7 / sigma / np.cosh(2 * sigma) ** 5)
    np.testing.assert_allclose(
        f_reed03(sigma) / f_sheth_tormen(sigma), expected_ratio, rtol=2e-14
    )


def test_reed07_slope_dependence_matches_published_exponential():
    sigma = np.array([0.5, 1, 2])
    slope = np.array([-2.8, -2.3, -1.5])
    ratio = f_reed07(sigma, neff=slope) / f_reed07(sigma, neff=-2.0)
    expected = np.exp(-0.03 * (1.686 / sigma) ** 0.6 * ((slope + 3) ** -2 - 1))
    np.testing.assert_allclose(ratio, expected, rtol=2e-14)
    with pytest.raises(ValueError, match="requires the effective"):
        f_reed07(sigma)
    with pytest.raises(ValueError, match="n_eff > -3"):
        f_reed07(sigma, neff=-3)


@pytest.mark.parametrize(
    "z, expected",
    [
        (0.0, [0.018385830031551, 0.288231441278056, 0.316945283685185]),
        (6.0, [0.015824907419128, 0.210487592906024, 0.408227607148221]),
    ],
)
def test_watson_so_endpoint_coefficients_match_published_mapping(z, expected):
    # Watson et al. (2013), eq. 17 and Table 2, Δ=178 and Ωm=0.3.
    np.testing.assert_allclose(
        f_watson_so13(np.array([0.5, 1.0, 2.0]), z=z, omega_m_z=0.3, delta_halo=178),
        expected,
        rtol=2e-12,
    )


@pytest.mark.parametrize("fit", FITTING_NAMES)
def test_multiplicity_extreme_variance_preserves_finite_limits(fit):
    sigma = np.array([1e-300, 1e-200, 1e-10, 0.1, 1.0, 10.0, 1e100, 1e300])
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        values = fitting_values(sigma, 1.686, fit, neff=-2.0, delta_halo=200.0)
    assert np.isfinite(values).all()
    assert (values >= 0).all()
    assert values[0] == 0


def test_sheth_tormen_large_sigma_matches_analytic_asymptote():
    # f ~ A sqrt(2a/pi) a^-p (delta_c/sigma)^(1-2p) for small peak height.
    sigma = np.array([1e100, 1e200, 1e300])
    expected = (
        0.3222 * np.sqrt(2 * 0.707 / np.pi) * 0.707**-0.3 * (1.686 / sigma) ** 0.4
    )
    np.testing.assert_allclose(f_sheth_tormen(sigma), expected, rtol=2e-13)


def test_press_schechter_and_sheth_tormen_mass_fraction_normalization():
    from scipy.integrate import quad

    for function in (f_press_schechter, f_sheth_tormen):
        fraction, _ = quad(
            lambda lognu: float(function(1.686 / np.exp(lognu))), -50, 5, epsabs=1e-9
        )
        assert fraction == pytest.approx(1.0, abs=6e-5)
