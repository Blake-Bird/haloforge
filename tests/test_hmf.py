import numpy as np
import pytest

from engine.fitting_functions import (
    f_press_schechter,
    f_sheth_tormen,
    f_reed07,
    nu_from_sigma,
)
from engine.hmf import cumulative_hmf, hmf_from_sigma


def test_fitting_functions_positive_and_finite():
    sigma = np.array([0.5, 1.0, 2.0])
    ps = f_press_schechter(sigma, 1.686)
    st = f_sheth_tormen(sigma, 1.686)
    assert np.all(np.isfinite(ps))
    assert np.all(np.isfinite(st))
    assert np.all(ps > 0)
    assert np.all(st > 0)


def test_nu_from_sigma():
    assert np.allclose(nu_from_sigma(np.array([1.0, 2.0]), 1.686), [1.686, 0.843])


def test_hmf_positive_shape():
    M = np.logspace(10, 14, 20)
    sigma = (M / 1e12) ** -0.15
    hmf = hmf_from_sigma(
        M, sigma, rho0=4e10, h=0.7, fitting="Sheth-Tormen", delta_c=1.686
    )
    assert hmf.shape == M.shape
    assert np.all(hmf > 0)


@pytest.mark.parametrize("spectral_index", [-2.8, -2.0, -1.0])
def test_reed07_hmf_uses_the_measured_mass_variance_slope(spectral_index):
    mass = np.geomspace(1e10, 1e14, 80)
    exponent = -(spectral_index + 3) / 6
    sigma = (mass / 1e12) ** exponent
    actual = hmf_from_sigma(
        mass, sigma, 4e10, 0.7, "Reed 2007", 1.686, mass_definition="fof_b0.2"
    )
    expected = (
        4e10 / mass / 0.7**3 * abs(exponent) * f_reed07(sigma, neff=spectral_index)
    )
    np.testing.assert_allclose(actual, expected, rtol=2e-11)


def test_hmf_underflow_is_zero_not_fabricated_positive_density():
    mass = np.geomspace(1e13, 1e15, 20)
    result = hmf_from_sigma(
        mass, (mass / 1e13) ** -0.15 * 0.001, 4e10, 0.7, "Press-Schechter", 1.686
    )
    np.testing.assert_array_equal(result, np.zeros_like(mass))


@pytest.mark.parametrize("bad", [0, -1, np.nan, np.inf])
def test_hmf_invalid_sigma_rejected(bad):
    with pytest.raises(ValueError):
        hmf_from_sigma(
            [1e10, 1e11, 1e12], [2, 1, bad], 4e10, 0.7, "Press-Schechter", 1.686
        )


def test_cumulative_hmf_decreases_with_mass():
    M = np.logspace(10, 14, 20)
    values = M**-0.5
    cumulative = cumulative_hmf(M, values)
    assert np.all(np.diff(cumulative[:-1]) <= 0)


def test_cumulative_constant_per_log_mass_has_exact_finite_bound():
    mass = np.geomspace(1e8, 1e15, 79)
    actual = cumulative_hmf(mass, np.full(mass.shape, 0.02))
    np.testing.assert_allclose(actual, 0.02 * np.log(mass[-1] / mass), atol=1e-15)
    assert actual[-1] == 0


@pytest.mark.parametrize("slope", [-5.0, -1.5, -0.01, 0.01, 1.0])
def test_cumulative_power_law_is_exact_even_on_sparse_mass_grids(slope):
    mass = np.geomspace(1e10, 1e15, 20)
    abundance = (mass / mass[0]) ** slope
    expected = abundance * np.expm1(slope * np.log(mass[-1] / mass)) / slope
    np.testing.assert_allclose(
        cumulative_hmf(mass, abundance), expected, rtol=2e-13, atol=1e-15
    )


def test_cumulative_zero_tail_uses_linear_interval_without_a_positive_floor():
    mass = np.exp(np.array([0.0, 1.0, 2.0, 3.0]))
    abundance = np.array([4.0, 2.0, 0.0, 0.0])
    expected = [2 / np.log(2) + 1, 1, 0, 0]
    np.testing.assert_allclose(cumulative_hmf(mass, abundance), expected, rtol=1e-14)


def test_cumulative_logarithmic_mean_remains_finite_across_extreme_dynamic_range():
    result = cumulative_hmf([1.0, np.e], [1e300, 1e-300])
    assert result[0] == pytest.approx(1e300 / (600 * np.log(10)), rel=1e-14)
    assert result[1] == 0


@pytest.mark.parametrize(
    "mass, abundance",
    [
        ([1, 1], [2, 1]),
        ([0, 1], [2, 1]),
        ([1, 2], [2, -1]),
        ([1, 2], [2, np.nan]),
        ([1, 2], [1]),
    ],
)
def test_cumulative_rejects_invalid_data(mass, abundance):
    with pytest.raises(ValueError):
        cumulative_hmf(mass, abundance)
