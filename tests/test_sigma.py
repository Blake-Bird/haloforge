import numpy as np
import pytest
from scipy.special import gamma

from engine.sigma import (
    R_from_mass,
    build_power_interpolator,
    dlog_sigma_dlog_M,
    mass_from_R,
    sigma_integrand_per_logk,
    sigma_squared,
    sigma_grid,
)


def test_mass_radius_roundtrip():
    rho0 = 1.0e11
    mass = np.array([1.0e10, 1.0e12])
    radius = R_from_mass(mass, rho0)
    assert np.allclose(mass_from_R(radius, rho0), mass)


def test_power_interpolator_positive():
    k = np.logspace(-3, 1, 20)
    p = k**-1
    interp = build_power_interpolator(k, p)
    assert float(interp(0.1)) > 0


def test_sigma_integrand_positive_shape():
    k = np.logspace(-3, 1, 20)
    p = np.ones_like(k)
    values = sigma_integrand_per_logk(k, p, 1.0, "Gaussian")
    assert values.shape == k.shape
    assert np.all(values >= 0)


def test_dlog_sigma_shape():
    m = np.logspace(8, 12, 10)
    sigma = m**-0.1
    slope = dlog_sigma_dlog_M(m, sigma)
    assert slope.shape == m.shape
    assert np.all(slope < 0)


@pytest.mark.parametrize("index", [-3.0, -2.0, 0.0, 1.0])
def test_sharp_k_matches_finite_power_law_integral_between_samples(index):
    k = np.geomspace(1e-4, 100, 35)
    amplitude = 13.0
    power = amplitude * k**index
    for radius in (0.001, 0.37, 1.91, 900.0, 2e4):
        upper = np.clip(1 / radius, k[0], k[-1])
        exponent = index + 3
        expected = (
            amplitude
            / (2 * np.pi**2)
            * (
                np.log(upper / k[0])
                if exponent == 0
                else (upper**exponent - k[0] ** exponent) / exponent
            )
        )
        assert sigma_squared(radius, k, power, "Sharp-k") == pytest.approx(
            expected, rel=2e-13, abs=1e-15
        )


@pytest.mark.parametrize("index", [-2.0, -1.0, 0.0, 1.0])
def test_gaussian_variance_and_derivative_match_analytic_power_law(index):
    k = np.geomspace(1e-7, 1e3, 4001)
    mass = np.geomspace(1e10, 1e15, 51)
    grid = sigma_grid(mass, k, k**index, {"h": 0.7, "Omega_m": 0.3}, "Gaussian")
    expected = gamma((index + 3) / 2) / (4 * np.pi**2 * grid["R"] ** (index + 3))
    np.testing.assert_allclose(grid["sigma"] ** 2, expected, rtol=3e-6)
    np.testing.assert_allclose(grid["dlnsigma_dlnM"], -(index + 3) / 6, rtol=1e-5)


def test_sharp_k_mass_grid_has_no_sampled_cutoff_staircase():
    k = np.geomspace(1e-6, 100, 40)
    mass = np.geomspace(1e10, 1e15, 300)
    grid = sigma_grid(mass, k, k**-2, {"h": 0.7, "Omega_m": 0.3}, "Sharp-k")
    expected = (1 / grid["R"] - k[0]) / (2 * np.pi**2)
    np.testing.assert_allclose(grid["sigma"] ** 2, expected, rtol=1e-13)
    np.testing.assert_allclose(grid["dlnsigma_dlnM"], -1 / 6, rtol=3e-5)
    assert "cutoff" in grid["integration_method"]


@pytest.mark.parametrize("bad", [0, -1, np.nan, np.inf])
def test_interpolation_rejects_invalid_power_and_queries(bad):
    with pytest.raises(ValueError):
        build_power_interpolator([1, 2], [1, bad])
    interpolate = build_power_interpolator([1, 2], [2, 1])
    with pytest.raises(ValueError):
        interpolate(bad)
    with pytest.raises(ValueError):
        sigma_squared(bad, np.geomspace(0.01, 10, 10), np.ones(10), "Top-hat")


@pytest.mark.parametrize(
    "mass,sigma",
    [
        ([1, 1], [1, 2]),
        ([1, 2], [1, 0]),
        ([1, 2], [1, np.nan]),
        ([1], [1]),
        ([1, 2], [1]),
    ],
)
def test_derivative_rejects_invalid_arrays(mass, sigma):
    with pytest.raises(ValueError):
        dlog_sigma_dlog_M(mass, sigma)
