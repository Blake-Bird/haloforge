"""Scientific array invariants at solver and cache boundaries."""

from __future__ import annotations

import numpy as np


def validate_power_arrays(result: dict, params: dict | None = None) -> None:
    """Require complete sampled-redshift output; never invent missing slices."""
    try:
        for key in (
            "k",
            "P",
            "P_by_z",
            "redshifts",
            "growth_class",
            "background_omega_m_by_z",
            "background_cosmic_time_gyr_by_z",
        ):
            if key in result and np.asarray(result[key]).dtype.kind not in "fiu":
                raise ValueError(f"{key} must contain real numeric arrays")
        k, power, spectra, redshifts, growth = (
            np.asarray(result[key], dtype=float)
            for key in ("k", "P", "P_by_z", "redshifts", "growth_class")
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Incomplete matter-power arrays") from exc
    if (
        k.ndim != 1
        or k.size < 2
        or not np.isfinite(k).all()
        or np.any(k <= 0)
        or np.any(np.diff(k) <= 0)
    ):
        raise ValueError(
            "Matter-power k grid must be positive, finite, and strictly increasing"
        )
    if (
        redshifts.ndim != 1
        or redshifts.size == 0
        or not np.isfinite(redshifts).all()
        or redshifts[0] != 0
        or np.any(np.diff(redshifts) <= 0)
    ):
        raise ValueError("Matter-power redshifts must be ordered and start at zero")
    if (
        spectra.shape != (redshifts.size, k.size)
        or not np.isfinite(spectra).all()
        or np.any(spectra <= 0)
    ):
        raise ValueError(
            "P_by_z must contain a positive finite spectrum for every redshift"
        )
    if power.shape != k.shape or not np.array_equal(power, spectra[0]):
        raise ValueError("P must equal the stored zero-redshift spectrum")
    if (
        growth.shape != redshifts.shape
        or not np.isfinite(growth).all()
        or np.any(growth <= 0)
        or not np.isclose(growth[0], 1, rtol=1e-12, atol=0)
    ):
        raise ValueError(
            "Growth must be positive, finite, and normalized at zero redshift"
        )
    background = np.asarray(result.get("background_omega_m_by_z", []), dtype=float)
    if background.shape != (0,) and (
        background.shape != redshifts.shape
        or not np.isfinite(background).all()
        or np.any(background <= 0)
    ):
        raise ValueError(
            "Background matter fractions must be positive finite values at every redshift, or unavailable"
        )
    cosmic_time = np.asarray(
        result.get("background_cosmic_time_gyr_by_z", []), dtype=float
    )
    if cosmic_time.shape != (0,) and (
        cosmic_time.shape != redshifts.shape
        or not np.isfinite(cosmic_time).all()
        or np.any(cosmic_time <= 0)
        or np.any(np.diff(cosmic_time) >= 0)
    ):
        raise ValueError(
            "Background cosmic time must be positive, finite, and decrease with redshift, or unavailable"
        )
    if params is not None:
        requested_z = sorted(
            {
                0.0,
                float(params.get("single_z", 0.0)),
                *map(float, params.get("z_values", [])),
            }
        )
        if not np.array_equal(redshifts, requested_z):
            raise ValueError("Stored redshifts do not match the requested calculation")
        requested_k = np.logspace(
            np.log10(float(params["k_min"])),
            np.log10(float(params["k_max"])),
            int(params["k_points"]),
        )
        if k.shape != requested_k.shape or not np.allclose(
            k, requested_k, rtol=1e-12, atol=0
        ):
            raise ValueError("Stored k grid does not match the requested calculation")
