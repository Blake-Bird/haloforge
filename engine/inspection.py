"""Cross-pipeline scientific readouts for a selected halo mass."""

from __future__ import annotations

import numpy as np
from scipy.integrate import trapezoid

from engine.sigma import sigma_integrand_per_logk
from engine.redshift import redshift_index


def _positive_grid(values, name: str, minimum: int = 1) -> np.ndarray:
    grid = np.asarray(values, dtype=float)
    if (
        grid.ndim != 1
        or grid.size < minimum
        or not np.all(np.isfinite(grid))
        or np.any(grid <= 0)
        or np.any(np.diff(grid) <= 0)
    ):
        raise ValueError(f"{name} must be finite, positive, and strictly increasing")
    return grid


def _power_at_redshift(power: dict, z: float, k: np.ndarray) -> np.ndarray:
    redshifts = np.asarray(power.get("redshifts", [0.0]), dtype=float)
    index = redshift_index(redshifts, z)
    values = np.asarray(power.get("P_by_z", [power["P"]]), dtype=float)
    if (
        values.shape != (redshifts.size, k.size)
        or not np.all(np.isfinite(values))
        or np.any(values <= 0)
    ):
        raise ValueError(
            "The stored P(k,z) grid must be finite, positive, and match its coordinates"
        )
    return values[index]


def _central_contribution_band(
    k: np.ndarray, integrand: np.ndarray
) -> tuple[float, float]:
    """Invert the integral of the piecewise-linear sampled contribution in ln(k)."""
    k = _positive_grid(k, "Wavenumbers", 2)
    values = _normalized_contribution(integrand, k.shape)
    logk = np.log(k)
    intervals = (values[:-1] + values[1:]) * np.diff(logk) / 2.0
    cumulative = np.concatenate([[0.0], np.cumsum(intervals)])
    bounds = []
    for fraction in (0.1, 0.9):
        target = fraction * cumulative[-1]
        index = min(
            int(np.searchsorted(cumulative, target, side="right")) - 1, k.size - 2
        )
        width = logk[index + 1] - logk[index]
        area = (target - cumulative[index]) / width
        start, end = values[index : index + 2]
        # Stable positive root of start*t + (end-start)*t²/2 = area.
        root = np.sqrt(max(0.0, start * start + 2.0 * (end - start) * area))
        position = 2.0 * area / (start + root)
        bounds.append(float(np.exp(logk[index] + np.clip(position, 0.0, 1.0) * width)))
    return tuple(bounds)


def _normalized_contribution(integrand, shape: tuple) -> np.ndarray:
    values = np.asarray(integrand, dtype=float)
    if values.shape != shape or not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError(
            "Variance contributions must be finite, nonnegative, and match the k grid"
        )
    scale = float(np.max(values))
    if scale <= 0:
        raise ValueError("No nonzero variance contribution is resolved on this grid")
    # These diagnostics use only fractions; scaling prevents cumulative overflow.
    return values / scale


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
    masses = _positive_grid(sigma["M_h"], "Halo masses")
    requested_mass = float(selected_mass_hinv_msun)
    if not np.isfinite(requested_mass) or not masses[0] <= requested_mass <= masses[-1]:
        raise ValueError("The inspection mass must lie within the saved mass grid")
    index = int(
        np.argmin(np.abs(np.log(masses) - np.log(float(selected_mass_hinv_msun))))
    )
    redshifts = np.asarray(
        sigma.get("redshifts", power.get("redshifts", [0.0])), dtype=float
    )
    requested_z = float(params.get("single_z", 0.0) if redshift is None else redshift)
    z_index = redshift_index(redshifts, requested_z)
    sigma_by_z = np.asarray(sigma.get("sigma_by_z", [sigma["sigma"]]), dtype=float)
    if (
        sigma_by_z.shape != (redshifts.size, masses.size)
        or not np.all(np.isfinite(sigma_by_z))
        or np.any(sigma_by_z <= 0)
    ):
        raise ValueError(
            "The stored variance grid must be finite, positive, and match its coordinates"
        )
    k = _positive_grid(power["k"], "Wavenumbers", 2)
    physical_masses = _positive_grid(sigma["M"], "Physical masses")
    radii = _positive_grid(sigma["R"], "Smoothing radii")
    if radii.shape != masses.shape or physical_masses.shape != masses.shape:
        raise ValueError("Mass and radius grids must have matching lengths")
    R = float(radii[index])
    sigma_value = float(sigma_by_z[z_index, index])
    delta_c = float(params.get("delta_c", 1.686))
    if not np.isfinite(delta_c) or delta_c <= 0:
        raise ValueError("The collapse threshold must be finite and positive")
    integrand = sigma_integrand_per_logk(
        k,
        _power_at_redshift(power, requested_z, k),
        R,
        sigma.get("window_type", params.get("window_type", "Top-hat")),
    )
    k_low, k_high = _central_contribution_band(k, integrand)
    calibrated = None
    if validity and validity.get("calibrated_mask") is not None:
        mask = np.asarray(validity["calibrated_mask"])
        if mask.shape == masses.shape and mask.dtype.kind == "b":
            calibrated = bool(mask[index])
    return {
        "requested_mass_hinv_msun": float(selected_mass_hinv_msun),
        "mass_hinv_msun": float(masses[index]),
        "mass_msun": float(np.asarray(sigma["M"], dtype=float)[index]),
        "radius_mpc": R,
        "redshift": float(redshifts[z_index]),
        "sigma": sigma_value,
        "nu": delta_c / sigma_value,
        "k_10_mpc_inv": k_low,
        "k_90_mpc_inv": k_high,
        "fit_checks_pass_at_mass": calibrated,
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
    if (
        isinstance(top_n, (bool, np.bool_))
        or not isinstance(top_n, (int, np.integer))
        or not 1 <= top_n <= 100
    ):
        raise ValueError("top_n must be an integer between 1 and 100")
    k_start, k_end = float(k_start_mpc_inv), float(k_end_mpc_inv)
    if not (np.isfinite(k_start) and np.isfinite(k_end) and 0 < k_start < k_end):
        raise ValueError(
            "The selected k interval must be finite, positive, and increasing"
        )
    params, power, sigma = run["params"], run["power_result"], run["sigma_result"]
    k = _positive_grid(power["k"], "Wavenumbers", 2)
    masses, radii = (
        _positive_grid(sigma["M_h"], "Halo masses"),
        _positive_grid(sigma["R"], "Smoothing radii"),
    )
    if masses.shape != radii.shape:
        raise ValueError("Mass and radius grids must have matching lengths")
    if k_start < k[0] or k_end > k[-1]:
        raise ValueError("The selected k interval must lie within the saved spectrum")
    redshifts = np.asarray(
        sigma.get("redshifts", power.get("redshifts", [0.0])), dtype=float
    )
    requested_z = float(params.get("single_z", 0.0) if redshift is None else redshift)
    z_index = redshift_index(redshifts, requested_z)
    power_at_z = _power_at_redshift(power, requested_z, k)
    in_band = (k >= k_start) & (k <= k_end)
    if np.count_nonzero(in_band) < 2:
        raise ValueError("Select a k interval containing at least two sampled k values")
    logk = np.log(k)
    rows = []
    for mass, radius in zip(masses, radii):
        integrand = _normalized_contribution(
            sigma_integrand_per_logk(
                k,
                power_at_z,
                float(radius),
                sigma.get("window_type", params.get("window_type", "Top-hat")),
            ),
            k.shape,
        )
        total = float(trapezoid(integrand, logk))
        band = float(trapezoid(integrand[in_band], logk[in_band]))
        fraction = band / total
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
