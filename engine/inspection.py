"""Cross-pipeline scientific readouts for a selected halo mass."""

from __future__ import annotations

import numpy as np

from engine.sigma import sigma_integrand_per_logk


def _central_contribution_band(
    k: np.ndarray, integrand: np.ndarray
) -> tuple[float, float]:
    """Return the 10–90% cumulative σ² band in logarithmic k."""
    k = np.asarray(k, dtype=float)
    values = np.maximum(np.asarray(integrand, dtype=float), 0.0)
    if k.size < 2 or not np.any(values > 0):
        return float("nan"), float("nan")
    logk = np.log(k)
    intervals = (values[:-1] + values[1:]) * np.diff(logk) / 2.0
    cumulative = np.concatenate([[0.0], np.cumsum(intervals)])
    total = cumulative[-1]
    if not np.isfinite(total) or total <= 0:
        return float("nan"), float("nan")
    return tuple(
        float(np.exp(np.interp(fraction * total, cumulative, logk)))
        for fraction in (0.1, 0.9)
    )


def inspect_mass_point(
    run: dict,
    selected_mass_hinv_msun: float,
    redshift: float | None = None,
    validity: dict | None = None,
) -> dict:
    """Connect a selected mass to radius, σ, rarity and contributing modes.

    The result only describes sampled modes in the supplied run and does not
    convert a numerical contribution band into a convergence claim.
    """
    params = run["params"]
    power = run["power_result"]
    sigma = run["sigma_result"]
    masses = np.asarray(sigma["M_h"], dtype=float)
    if masses.ndim != 1 or masses.size == 0 or np.any(masses <= 0):
        raise ValueError("A positive sampled halo-mass grid is required for inspection")
    index = int(
        np.argmin(np.abs(np.log(masses) - np.log(float(selected_mass_hinv_msun))))
    )
    redshifts = np.asarray(
        sigma.get("redshifts", power.get("redshifts", [0.0])), dtype=float
    )
    requested_z = float(params.get("single_z", 0.0) if redshift is None else redshift)
    z_index = int(np.argmin(np.abs(redshifts - requested_z)))
    if not np.isclose(redshifts[z_index], requested_z, atol=1e-9, rtol=0.0):
        raise ValueError("The requested redshift was not sampled in this run")
    p_by_z = np.asarray(power.get("P_by_z", [power["P"]]), dtype=float)
    sigma_by_z = np.asarray(sigma.get("sigma_by_z", [sigma["sigma"]]), dtype=float)
    k = np.asarray(power["k"], dtype=float)
    R = float(np.asarray(sigma["R"], dtype=float)[index])
    sigma_value = float(sigma_by_z[z_index, index])
    integrand = sigma_integrand_per_logk(
        k, p_by_z[z_index], R, params.get("window_type", "Top-hat")
    )
    k_low, k_high = _central_contribution_band(k, integrand)
    calibrated = None
    if validity and validity.get("calibrated_mask") is not None:
        mask = np.asarray(validity["calibrated_mask"], dtype=bool)
        if mask.shape == masses.shape:
            calibrated = bool(mask[index])
    return {
        "requested_mass_hinv_msun": float(selected_mass_hinv_msun),
        "mass_hinv_msun": float(masses[index]),
        "mass_msun": float(np.asarray(sigma["M"], dtype=float)[index]),
        "radius_mpc": R,
        "redshift": float(redshifts[z_index]),
        "sigma": sigma_value,
        "nu": float(params.get("delta_c", 1.686)) / sigma_value,
        "k_10_mpc_inv": k_low,
        "k_90_mpc_inv": k_high,
        "calibrated_at_mass": calibrated,
        "scope_limit": "The k band contains the central sampled contribution to σ²; it does not test uncomputed modes or establish HMF calibration.",
    }


def inspect_k_range(
    run: dict,
    k_start_mpc_inv: float,
    k_end_mpc_inv: float,
    redshift: float | None = None,
    *,
    top_n: int = 8,
) -> dict:
    """Rank sampled halo masses by the fraction of σ² from a selected k band.

    This maps a Fourier interval to the stored smoothing-grid calculation. It
    is contribution accounting on sampled modes, not physical causation.
    """
    if not 1 <= int(top_n) <= 100:
        raise ValueError("top_n must be between 1 and 100")
    k_start, k_end = float(k_start_mpc_inv), float(k_end_mpc_inv)
    if not (np.isfinite(k_start) and np.isfinite(k_end) and 0 < k_start < k_end):
        raise ValueError(
            "The selected k interval must be finite, positive, and increasing"
        )
    params, power, sigma = run["params"], run["power_result"], run["sigma_result"]
    k = np.asarray(power["k"], dtype=float)
    masses, radii = (
        np.asarray(sigma["M_h"], dtype=float),
        np.asarray(sigma["R"], dtype=float),
    )
    if (
        k.ndim != 1
        or k.size < 2
        or np.any(np.diff(k) <= 0)
        or masses.shape != radii.shape
    ):
        raise ValueError("A valid sampled k and mass-radius grid is required")
    redshifts = np.asarray(
        sigma.get("redshifts", power.get("redshifts", [0.0])), dtype=float
    )
    requested_z = float(params.get("single_z", 0.0) if redshift is None else redshift)
    z_index = int(np.argmin(np.abs(redshifts - requested_z)))
    if not np.isclose(redshifts[z_index], requested_z, atol=1e-9, rtol=0.0):
        raise ValueError("The requested redshift was not sampled in this run")
    p_by_z = np.asarray(power.get("P_by_z", [power["P"]]), dtype=float)
    if p_by_z.ndim != 2 or p_by_z.shape[1] != k.size:
        raise ValueError(
            "The stored P(k,z) grid is incompatible with the selected range"
        )
    in_band = (k >= k_start) & (k <= k_end)
    if np.count_nonzero(in_band) < 2:
        raise ValueError("Select a k interval containing at least two sampled k values")
    logk = np.log(k)
    rows = []
    for mass, radius in zip(masses, radii):
        integrand = np.maximum(
            sigma_integrand_per_logk(
                k, p_by_z[z_index], float(radius), params.get("window_type", "Top-hat")
            ),
            0.0,
        )
        total = float(np.trapezoid(integrand, logk))
        band = float(np.trapezoid(integrand[in_band], logk[in_band]))
        fraction = band / total if total > 0 and np.isfinite(total) else float("nan")
        rows.append(
            {
                "mass_hinv_msun": float(mass),
                "radius_mpc": float(radius),
                "sigma2_fraction_from_selected_k": fraction,
            }
        )
    rows.sort(key=lambda row: row["sigma2_fraction_from_selected_k"], reverse=True)
    return {
        "k_start_mpc_inv": k_start,
        "k_end_mpc_inv": k_end,
        "actual_sampled_k_start_mpc_inv": float(k[in_band][0]),
        "actual_sampled_k_end_mpc_inv": float(k[in_band][-1]),
        "redshift": float(redshifts[z_index]),
        "rows": rows[: int(top_n)],
        "scope_limit": "Fractions use trapezoidal integration over the already sampled P(k) grid and omit all modes outside that grid. They rank variance contributions, not physical causation, convergence, or HMF-fit validity.",
    }
