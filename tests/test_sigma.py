import numpy as np

from engine.sigma import R_from_mass, build_power_interpolator, dlog_sigma_dlog_M, mass_from_R, sigma_integrand_per_logk


def test_mass_radius_roundtrip():
    rho0 = 1.0e11
    mass = np.array([1.0e10, 1.0e12])
    radius = R_from_mass(mass, rho0)
    assert np.allclose(mass_from_R(radius, rho0), mass)


def test_power_interpolator_positive():
    k = np.logspace(-3, 1, 20)
    p = k ** -1
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
    sigma = m ** -0.1
    slope = dlog_sigma_dlog_M(m, sigma)
    assert slope.shape == m.shape
    assert np.all(slope < 0)

