import numpy as np

from engine.sigma import rho_crit_from_h, rho0_from_cosmology


def test_rho_crit_uses_h_squared():
    assert np.isclose(rho_crit_from_h(0.7), 2.775e11 * 0.7**2)


def test_rho0_from_cosmology():
    assert np.isclose(rho0_from_cosmology({"h": 0.7, "Omega_m": 0.3}), 0.3 * 2.775e11 * 0.7**2)
