"""Stable sigma(M) integration on the sampled logarithmic CLASS k grid."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.integrate import simpson

from config.constants import RHO_CRIT_COEFF
from engine.windows import window_squared


def rho_crit_from_h(h: float) -> float:
    return RHO_CRIT_COEFF * float(h) ** 2


def rho0_from_cosmology(cosmology: dict) -> float:
    h = float(cosmology.get("h", float(cosmology.get("H0", 67.81)) / 100.0))
    return float(cosmology["Omega_m"]) * rho_crit_from_h(h)


def build_power_interpolator(
    k: np.ndarray, P: np.ndarray
) -> Callable[[float | np.ndarray], np.ndarray]:
    k_arr = np.asarray(k, dtype=float)
    p_arr = np.maximum(np.asarray(P, dtype=float), 1e-300)
    if k_arr.ndim != 1 or p_arr.ndim != 1 or len(k_arr) != len(p_arr):
        raise ValueError("k and P must be one-dimensional arrays of equal length")
    if np.any(k_arr <= 0.0) or np.any(np.diff(k_arr) <= 0.0):
        raise ValueError("k must be strictly positive and increasing")
    logk = np.log(k_arr)
    logp = np.log(p_arr)

    def interpolator(k_value: float | np.ndarray) -> np.ndarray:
        kv = np.asarray(k_value, dtype=float)
        if np.any(kv < k_arr[0]) or np.any(kv > k_arr[-1]):
            raise ValueError(
                "P(k) interpolation requested outside the computed CLASS range"
            )
        return np.exp(np.interp(np.log(kv), logk, logp))

    return interpolator


def P_of_k(
    k_value: float | np.ndarray,
    interpolator: Callable[[float | np.ndarray], np.ndarray],
) -> np.ndarray:
    return interpolator(k_value)


def mass_from_R(R: np.ndarray | float, rho0: float) -> np.ndarray:
    r = np.asarray(R, dtype=float)
    return 4.0 * np.pi * float(rho0) * r**3 / 3.0


def R_from_mass(M: np.ndarray | float, rho0: float) -> np.ndarray:
    m = np.asarray(M, dtype=float)
    return (3.0 * m / (4.0 * np.pi * float(rho0))) ** (1.0 / 3.0)


def _validate_power_grid(k: np.ndarray, P: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    k_arr = np.asarray(k, dtype=float)
    p_arr = np.asarray(P, dtype=float)
    if k_arr.ndim != 1 or p_arr.ndim != 1 or k_arr.size != p_arr.size:
        raise ValueError("k and P must be one-dimensional arrays of equal length")
    if k_arr.size < 8:
        raise ValueError("At least eight k samples are required for sigma integration")
    if (
        np.any(~np.isfinite(k_arr))
        or np.any(~np.isfinite(p_arr))
        or np.any(k_arr <= 0)
        or np.any(p_arr <= 0)
    ):
        raise ValueError("k and P must be finite and strictly positive")
    if np.any(np.diff(k_arr) <= 0):
        raise ValueError("k must be strictly increasing")
    return k_arr, p_arr


def sigma_squared(
    R: float, k: np.ndarray, P: np.ndarray, window_type: str, quad_limit: int = 200
) -> float:
    """Compute sigma^2 with Simpson integration in ln(k).

    ``quad_limit`` is accepted for backward-compatible saved parameter files;
    integration uses the complete sampled CLASS grid rather than an adaptive
    integrator, eliminating repeated interpolation and subdivision failures.
    """
    del quad_limit
    k_arr, p_arr = _validate_power_grid(k, P)
    logk = np.log(k_arr)
    integrand = sigma_integrand_per_logk(k_arr, p_arr, float(R), window_type)
    value = simpson(integrand, x=logk)
    if not np.isfinite(value):
        raise FloatingPointError("sigma integration returned a non-finite value")
    return max(float(value), 0.0)


def sigma_grid(
    M_values: np.ndarray,
    k: np.ndarray,
    P: np.ndarray,
    cosmology: dict,
    window_type: str,
    quad_limit: int = 200,
) -> dict:
    """Compute R(M), sigma(M), and dlnsigma/dlnM in stable vectorized batches."""
    k_arr, p_arr = _validate_power_grid(k, P)
    rho0 = rho0_from_cosmology(cosmology)
    M = np.asarray(M_values, dtype=float)
    if (
        M.ndim != 1
        or M.size < 4
        or np.any(~np.isfinite(M))
        or np.any(M <= 0)
        or np.any(np.diff(M) <= 0)
    ):
        raise ValueError(
            "Mass grid must be finite, positive, one-dimensional, and increasing"
        )
    R = R_from_mass(M, rho0)
    logk = np.log(k_arr)
    base = k_arr**3 * p_arr / (2.0 * np.pi**2)
    sigma2 = np.empty(M.size, dtype=float)
    batch = max(8, min(int(quad_limit), M.size))
    for start in range(0, M.size, batch):
        stop = min(start + batch, M.size)
        response = window_squared(R[start:stop, None] * k_arr[None, :], window_type)
        sigma2[start:stop] = simpson(base[None, :] * response, x=logk, axis=1)
    sigma_values = np.sqrt(np.maximum(sigma2, 0.0))
    if np.any(~np.isfinite(sigma_values)) or np.any(sigma_values <= 0):
        raise FloatingPointError(
            "sigma(M) contains non-finite or non-positive values; increase the k range or k sampling"
        )
    return {
        "M": M,
        "R": R,
        "sigma": sigma_values,
        "dlnsigma_dlnM": dlog_sigma_dlog_M(M, sigma_values),
        "rho0": rho0,
        "integration_method": "log-k Simpson",
    }


def dlog_sigma_dlog_M(M_values: np.ndarray, sigma_values: np.ndarray) -> np.ndarray:
    M = np.asarray(M_values, dtype=float)
    sigma = np.asarray(sigma_values, dtype=float)
    edge_order = 2 if M.size >= 3 else 1
    return np.gradient(
        np.log(np.maximum(sigma, 1e-300)), np.log(M), edge_order=edge_order
    )


def sigma_integrand_per_logk(
    k_values: np.ndarray, P_values: np.ndarray, R: float, window_type: str
) -> np.ndarray:
    k = np.asarray(k_values, dtype=float)
    P = np.asarray(P_values, dtype=float)
    return k**3 * P * window_squared(k * R, window_type) / (2.0 * np.pi**2)


def sigma_numerical_diagnostics(
    M: np.ndarray,
    k: np.ndarray,
    P: np.ndarray,
    cosmology: dict,
    window_type: str,
    quad_limit: int,
) -> dict:
    """Report sampled-range coverage and a repeatable truncation sensitivity check.

    This is an internal check on the already sampled P(k), not a claim that
    CLASS settings or physical modelling are converged. The check deliberately
    removes part of the low- or high-k range and reports the resulting σ(M)
    shift; it cannot reveal missing power outside the original CLASS grid.
    """
    k, P = _validate_power_grid(k, P)
    full = sigma_grid(M, k, P, cosmology, window_type, quad_limit)
    R = full["R"]
    samples_per_decade = (k.size - 1) / np.log10(k[-1] / k[0])
    coverage = {
        "kR_min_at_largest_mass": float(k[0] * R[-1]),
        "kR_max_at_smallest_mass": float(k[-1] * R[0]),
        "k_samples_per_decade": float(samples_per_decade),
        "mass_samples_per_decade": float((len(M) - 1) / np.log10(M[-1] / M[0])),
    }
    coverage_notes = []
    if coverage["kR_min_at_largest_mass"] > 0.1:
        coverage_notes.append(
            "k_min may omit large-scale contributions at the largest mass."
        )
    if coverage["kR_max_at_smallest_mass"] < 20:
        coverage_notes.append(
            "k_max may omit small-scale contributions at the smallest mass."
        )
    if samples_per_decade < 30:
        coverage_notes.append(
            "The sampled P(k) grid has fewer than 30 points per decade."
        )
    coverage["status"] = "review_required" if coverage_notes else "range_looks_adequate"
    coverage["notes"] = coverage_notes or [
        "Coverage heuristics passed; run a wider CLASS grid to establish convergence."
    ]

    def sensitivity(mask: np.ndarray, label: str) -> dict:
        if int(mask.sum()) < 8:
            return {
                "status": "not_evaluated",
                "reason": f"{label} truncation leaves fewer than eight k samples.",
            }
        partial = sigma_grid(M, k[mask], P[mask], cosmology, window_type, quad_limit)[
            "sigma"
        ]
        fractional = np.abs(partial / full["sigma"] - 1.0)
        index = int(np.argmax(fractional))
        max_fractional = float(fractional[index])
        status = "low_sensitivity" if max_fractional < 0.01 else "material_sensitivity"
        return {
            "status": status,
            "maximum_fractional_sigma_change": max_fractional,
            "mass_at_maximum_Msun": float(M[index]),
            "removed_fraction_of_log_k_range": float(
                1
                - (np.log(k[mask][-1]) - np.log(k[mask][0]))
                / (np.log(k[-1]) - np.log(k[0]))
            ),
        }

    high_mask = k <= k[-1] / 2.0
    low_mask = k >= k[0] * 2.0
    return {
        "method": "Reintegrate the sampled P(k) after removing half of the high- or low-k endpoint range.",
        "scope_limit": "This does not test power outside the computed CLASS range or empirical-HMF calibration.",
        "coverage": coverage,
        "high_k_truncation": sensitivity(high_mask, "High-k"),
        "low_k_truncation": sensitivity(low_mask, "Low-k"),
    }


def compute_sigma_result(power_result: dict, params: dict) -> dict:
    h = float(power_result.get("derived", {}).get("h", float(params["H0"]) / 100.0))
    mass_min = float(params["mass_min_exp"])
    mass_max = float(params["mass_max_exp"])
    M_h = np.logspace(mass_min, mass_max, int(params["mass_points"]))
    M = M_h / h
    cosmology = {"h": h, "Omega_m": float(params["Omega_m"])}
    redshifts = np.asarray(power_result.get("redshifts", [0.0]), dtype=float)
    p_by_z = np.asarray(power_result.get("P_by_z", [power_result["P"]]), dtype=float)
    grids = [
        sigma_grid(
            M,
            power_result["k"],
            pz,
            cosmology,
            params["window_type"],
            int(params.get("quad_limit", 200)),
        )
        for pz in p_by_z
    ]
    grid = grids[0]
    sigma_by_z = np.asarray([g["sigma"] for g in grids])
    derivative_by_z = np.asarray([g["dlnsigma_dlnM"] for g in grids])
    sigma8 = np.asarray(
        [
            np.sqrt(
                sigma_squared(
                    8.0 / h,
                    power_result["k"],
                    pz,
                    "Top-hat",
                    int(params.get("quad_limit", 200)),
                )
            )
            for pz in p_by_z
        ]
    )
    k = np.asarray(power_result["k"], dtype=float)
    focus_index = int(np.argmin(np.abs(redshifts - float(params.get("single_z", 0.0)))))
    numerical_diagnostics = sigma_numerical_diagnostics(
        M,
        k,
        p_by_z[focus_index],
        cosmology,
        params["window_type"],
        int(params.get("quad_limit", 200)),
    )
    return {
        "M_h": M_h,
        "M": grid["M"],
        "R": grid["R"],
        "sigma": grid["sigma"],
        "sigma_by_z": sigma_by_z,
        "redshifts": redshifts,
        "dlnsigma_dlnM": grid["dlnsigma_dlnM"],
        "dlnsigma_dlnM_by_z": derivative_by_z,
        "rho0": grid["rho0"],
        "window_type": params["window_type"],
        "delta_c": float(params["delta_c"]),
        "sigma8_pipeline_by_z": sigma8,
        "integration_method": "log-k Simpson",
        "numerical_diagnostics": numerical_diagnostics,
    }
