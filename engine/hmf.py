"""Halo mass function calculations."""

from __future__ import annotations

import numpy as np
from scipy.integrate import trapezoid

from engine.fitting_functions import fitting_values
from engine.cosmology import omega_radiation
from engine.sigma import dlog_sigma_dlog_M


def hmf_from_sigma(
    M_values: np.ndarray,
    sigma_values: np.ndarray,
    rho0: float,
    h: float,
    fitting: str,
    delta_c: float,
    z: float = 0.0,
    omega_m_z: float = 0.3,
    delta_halo: float = 200.0,
) -> np.ndarray:
    """Return dn/dlnM in h^3 Mpc^-3."""
    M = np.asarray(M_values, dtype=float)
    sigma = np.asarray(sigma_values, dtype=float)
    dlnsigma = dlog_sigma_dlog_M(M, sigma)
    f_values = fitting_values(
        sigma, delta_c, fitting, z=float(z), omega_m_z=float(omega_m_z), delta_halo=float(delta_halo)
    )
    hmf = (float(rho0) / M) * f_values * np.abs(dlnsigma)
    return np.maximum(hmf / float(h) ** 3, 1e-300)


def hmf_z(run: dict, z: float, fitting: str) -> dict:
    """Compute HMF using sigma(M,z) integrated from the matching CLASS P(k,z)."""
    sigma_result = run["sigma_result"]
    params = run["params"]
    h = float(run.get("power_result", {}).get("derived", {}).get("h", float(params["H0"]) / 100.0))
    redshifts = np.asarray(sigma_result.get("redshifts", [0.0]), dtype=float)
    index = int(np.argmin(np.abs(redshifts - float(z))))
    if not np.isclose(redshifts[index], float(z), rtol=0.0, atol=1e-9):
        raise ValueError(f"z={z:g} was not included in this CLASS run; rerun with that redshift selected")
    sigma_at_z = np.asarray(sigma_result.get("sigma_by_z", [sigma_result["sigma"]]))[index]
    omega_m0 = float(params["Omega_m"])
    omega_r0 = float(params.get("Omega_r", omega_radiation(params)))
    omega_k0 = float(params.get("Omega_k", 0.0))
    omega_l0 = max(1.0 - omega_m0 - omega_r0 - omega_k0, 0.0)
    ez2 = omega_r0 * (1 + z) ** 4 + omega_m0 * (1 + z) ** 3 + omega_k0 * (1 + z) ** 2 + omega_l0
    omega_m_z = omega_m0 * (1 + z) ** 3 / ez2
    hmf = hmf_from_sigma(
        sigma_result["M"],
        sigma_at_z,
        sigma_result["rho0"],
        h,
        fitting,
        float(params["delta_c"]),
        z=float(z),
        omega_m_z=omega_m_z,
        delta_halo=float(params.get("delta_halo", 200.0)),
    )
    return {
        "M_h": sigma_result["M_h"],
        "M": sigma_result["M"],
        "sigma": sigma_at_z,
        "hmf": hmf,
        "z": float(z),
        "fitting": fitting,
    }


def cumulative_hmf(M_h_values: np.ndarray, hmf_values: np.ndarray) -> np.ndarray:
    """Return cumulative n(>M) by integrating dn/dlnM from high mass downward."""
    mass = np.asarray(M_h_values, dtype=float)
    hmf = np.asarray(hmf_values, dtype=float)
    logm = np.log(mass)
    out = np.zeros_like(hmf)
    for i in range(len(hmf)):
        out[i] = trapezoid(hmf[i:], x=logm[i:])
    return np.maximum(out, 1e-300)
