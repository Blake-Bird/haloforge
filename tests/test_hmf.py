import numpy as np

from engine.fitting_functions import f_press_schechter, f_sheth_tormen, nu_from_sigma
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
    hmf = hmf_from_sigma(M, sigma, rho0=4e10, h=0.7, fitting="Sheth-Tormen", delta_c=1.686)
    assert hmf.shape == M.shape
    assert np.all(hmf > 0)


def test_cumulative_hmf_decreases_with_mass():
    M = np.logspace(10, 14, 20)
    values = M ** -0.5
    cumulative = cumulative_hmf(M, values)
    assert np.all(np.diff(cumulative[:-1]) <= 0)
