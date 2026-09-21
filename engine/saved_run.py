"""Normalize durable saved-run records for scientific post-processing.

Storage keeps numerical arrays in a compact ``arrays`` payload, while the
scientific pipeline uses explicit power and variance sections. This adapter is
the single, tested boundary between those representations.
"""

from __future__ import annotations

from copy import deepcopy

import numpy as np


def pipeline_from_saved_run(saved: dict) -> dict:
    """Return the HMF/validation-compatible pipeline for a durable saved run."""
    arrays = saved.get("arrays")
    params = saved.get("params")
    if not isinstance(arrays, dict) or not isinstance(params, dict):
        raise ValueError(
            "Saved run is missing its parameter or numerical-array payload"
        )
    required = ("k", "P", "M_h", "M", "R", "sigma", "dlnsigma_dlnM")
    missing = [key for key in required if key not in arrays]
    if missing:
        raise ValueError(
            "Saved run cannot be rehydrated for scientific post-processing; "
            f"missing arrays: {', '.join(missing)}"
        )
    redshifts = np.asarray(arrays.get("redshifts", [0.0]), dtype=float)
    power_by_z = np.asarray(arrays.get("P_by_z", [arrays["P"]]), dtype=float)
    sigma_by_z = np.asarray(arrays.get("sigma_by_z", [arrays["sigma"]]), dtype=float)
    derivative_by_z = np.asarray(
        arrays.get("dlnsigma_dlnM_by_z", [arrays["dlnsigma_dlnM"]]), dtype=float
    )
    if power_by_z.shape[0] != redshifts.size or sigma_by_z.shape[0] != redshifts.size:
        raise ValueError("Saved run has incompatible redshift-indexed numerical arrays")
    return {
        "params": deepcopy(params),
        "class_status": saved.get("class_status", "UNKNOWN"),
        "integrity_status": deepcopy(saved.get("integrity_status", {})),
        "arrays": arrays,
        "numerical_diagnostics": deepcopy(saved.get("numerical_diagnostics", {})),
        "power_result": {
            "k": arrays["k"],
            "P": arrays["P"],
            "P_by_z": power_by_z,
            "redshifts": redshifts,
            "derived": {
                "h": float(
                    saved.get("derived", {}).get(
                        "h", float(params.get("H0", 67.36)) / 100.0
                    )
                )
            },
            "background_omega_m_by_z": arrays.get("background_omega_m_by_z", []),
        },
        "sigma_result": {
            "M_h": arrays["M_h"],
            "M": arrays["M"],
            "R": arrays["R"],
            "sigma": sigma_by_z[0],
            "sigma_by_z": sigma_by_z,
            "dlnsigma_dlnM": derivative_by_z[0],
            "dlnsigma_dlnM_by_z": derivative_by_z,
            "sigma8_pipeline_by_z": arrays.get("sigma8_pipeline_by_z", []),
            "redshifts": redshifts,
            "rho0": saved.get("rho0"),
            "window_type": saved.get("window_type", params.get("window_type")),
            "numerical_diagnostics": deepcopy(saved.get("numerical_diagnostics", {})),
        },
    }
