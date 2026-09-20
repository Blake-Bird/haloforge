"""Halo mass function calculations."""

from __future__ import annotations

import numpy as np

from engine.fitting_functions import fitting_values
from engine.contracts import validity_report
from engine.cosmology import omega_radiation
from engine.sigma import dlog_sigma_dlog_M
from engine.redshift import redshift_index


def require_hmf_window(window: str) -> None:
    """Require the smoothing convention used by the supported HMF pipeline."""
    if window != "Top-hat":
        raise ValueError(
            f"{window} is available for variance exploration only. Halo abundance "
            "requires real-space Top-hat smoothing; alternate windows need a "
            "separately calibrated mass assignment and multiplicity model."
        )


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
    window_type: str = "Top-hat",
    mass_definition: str = "analytic_top_hat",
) -> np.ndarray:
    """Return dn/dlnM in h^3 Mpc^-3."""
    require_hmf_window(window_type)
    validity_report(fitting, sigma_values, z, mass_definition, delta_halo)
    M = np.asarray(M_values, dtype=float)
    sigma = np.asarray(sigma_values, dtype=float)
    if M.ndim != 1 or M.size < 3 or sigma.shape != M.shape:
        raise ValueError("HMF requires at least three matching mass and sigma samples")
    if np.any(~np.isfinite(M)) or np.any(M <= 0) or np.any(np.diff(M) <= 0):
        raise ValueError("Mass must be finite, positive, and strictly increasing")
    if np.any(~np.isfinite(sigma)) or np.any(sigma <= 0):
        raise ValueError("sigma must be finite and strictly positive")
    if any(not np.isfinite(v) or v <= 0 for v in (rho0, h, delta_c)):
        raise ValueError(
            "Mean density, h, and collapse threshold must be finite and positive"
        )
    dlnsigma = dlog_sigma_dlog_M(M, sigma)
    f_values = fitting_values(
        sigma,
        delta_c,
        fitting,
        z=float(z),
        omega_m_z=float(omega_m_z),
        delta_halo=float(delta_halo),
        neff=-6.0 * dlnsigma - 3.0,
    )
    hmf = (float(rho0) / M) * f_values * np.abs(dlnsigma)
    result = hmf / float(h) ** 3
    if np.any(~np.isfinite(result)) or np.any(result < 0):
        raise FloatingPointError("The halo model produced invalid abundance values")
    return result


def hmf_z(run: dict, z: float, fitting: str) -> dict:
    """Compute HMF using sigma(M,z) integrated from the matching CLASS P(k,z)."""
    sigma_result = run["sigma_result"]
    params = run["params"]
    window = sigma_result.get("window_type", params.get("window_type"))
    require_hmf_window(window)
    h = float(
        run.get("power_result", {})
        .get("derived", {})
        .get("h", float(params["H0"]) / 100.0)
    )
    redshifts = np.asarray(sigma_result.get("redshifts", [0.0]), dtype=float)
    index = redshift_index(redshifts, z)
    sigma_by_z = sigma_result.get("sigma_by_z")
    if sigma_by_z is None:
        sigma_by_z = np.asarray([sigma_result["sigma"]])
    sigma_at_z = np.asarray(sigma_by_z, dtype=float)[index]
    background_omega_m = np.asarray(
        run.get("power_result", {}).get("background_omega_m_by_z", []), dtype=float
    )
    if background_omega_m.shape == redshifts.shape and np.all(
        np.isfinite(background_omega_m)
    ):
        omega_m_z = float(background_omega_m[index])
        omega_m_source = "CLASS/AxiCLASS background"
    elif params.get("enable_ede") and fitting == "Watson SO 2013":
        raise ValueError(
            "Watson SO is unavailable for EDE without Ωm(z) from the AxiCLASS background."
        )
    else:
        omega_m0 = float(params["Omega_m"])
        omega_r0 = float(params.get("Omega_r", omega_radiation(params)))
        omega_k0 = float(params.get("Omega_k", 0.0))
        omega_l0 = 1.0 - omega_m0 - omega_r0 - omega_k0
        ez2 = (
            omega_r0 * (1 + z) ** 4
            + omega_m0 * (1 + z) ** 3
            + omega_k0 * (1 + z) ** 2
            + omega_l0
        )
        omega_m_z = omega_m0 * (1 + z) ** 3 / ez2
        omega_m_source = "analytic LCDM background"
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
        window_type=window,
        mass_definition=params.get("mass_definition", "analytic_top_hat"),
    )
    validity = validity_report(
        fitting,
        sigma_at_z,
        z,
        params.get("mass_definition", "analytic_top_hat"),
        float(params.get("delta_halo", 200.0)),
    )
    return {
        "M_h": sigma_result["M_h"],
        "M": sigma_result["M"],
        "sigma": sigma_at_z,
        "hmf": hmf,
        "z": float(z),
        "fitting": fitting,
        "omega_m_z": omega_m_z,
        "omega_m_source": omega_m_source,
        "validity": validity,
    }


def cumulative_hmf(M_h_values: np.ndarray, hmf_values: np.ndarray) -> np.ndarray:
    """Integrate dn/dlnM to the maximum sampled mass in linear time.

    This is n(M < halo mass < M_max), not an integral to infinity. The last
    sample is exactly zero. Positive intervals use exact integration of their
    log-log interpolant; intervals touching zero use a linear interpolant in
    ln(M). No positive floor is added for logarithmic plots.
    """
    mass = np.asarray(M_h_values, dtype=float)
    hmf = np.asarray(hmf_values, dtype=float)
    if mass.ndim != 1 or mass.size < 2 or hmf.shape != mass.shape:
        raise ValueError(
            "Cumulative HMF requires at least two matching mass and abundance samples"
        )
    if np.any(~np.isfinite(mass)) or np.any(mass <= 0) or np.any(np.diff(mass) <= 0):
        raise ValueError("Mass must be finite, positive, and strictly increasing")
    if np.any(~np.isfinite(hmf)) or np.any(hmf < 0):
        raise ValueError("Halo abundance must be finite and nonnegative")
    logm = np.log(mass)
    out = np.zeros_like(hmf)
    left, right = hmf[:-1], hmf[1:]
    means = left / 2 + right / 2
    positive = (left > 0) & (right > 0)
    high = np.maximum(left[positive], right[positive])
    low = np.minimum(left[positive], right[positive])
    separation = np.log(high) - np.log(low)
    factor = np.ones_like(separation)
    unequal = separation > 0
    factor[unequal] = -np.expm1(-separation[unequal]) / separation[unequal]
    means[positive] = high * factor
    intervals = means * np.diff(logm)
    out[:-1] = np.cumsum(intervals[::-1])[::-1]
    if not np.all(np.isfinite(out)):
        raise FloatingPointError("Cumulative halo abundance overflowed")
    return out
