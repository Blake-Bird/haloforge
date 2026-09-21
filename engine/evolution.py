"""Reproducible redshift-frame specifications for evolution calculations."""

from __future__ import annotations

import numpy as np


SAMPLING_RULES = ("uniform_z", "uniform_a", "uniform_log_a", "custom")


def evolution_redshifts(
    start_z: float,
    end_z: float,
    frame_count: int,
    sampling: str = "uniform_a",
    *,
    custom_redshifts: list[float] | None = None,
) -> np.ndarray:
    """Return an exact descending redshift grid with no implicit substitution."""
    if sampling not in SAMPLING_RULES:
        raise ValueError(f"Unknown evolution sampling rule: {sampling}")
    if sampling == "custom":
        values = np.asarray(custom_redshifts, dtype=float)
        if values.ndim != 1 or values.size < 2:
            raise ValueError("Custom evolution requires at least two redshifts")
    else:
        if not all(np.isfinite(value) and value >= 0 for value in (start_z, end_z)):
            raise ValueError("Evolution redshifts must be finite and nonnegative")
        if start_z <= end_z:
            raise ValueError(
                "Evolution must run from a higher start redshift to a lower end redshift"
            )
        if not isinstance(frame_count, int) or frame_count < 2:
            raise ValueError("Evolution requires at least two frames")
        if sampling == "uniform_z":
            values = np.linspace(start_z, end_z, frame_count)
        else:
            a_start, a_end = 1 / (1 + start_z), 1 / (1 + end_z)
            a = (
                np.linspace(a_start, a_end, frame_count)
                if sampling == "uniform_a"
                else np.exp(np.linspace(np.log(a_start), np.log(a_end), frame_count))
            )
            values = 1 / a - 1
    if (
        np.any(~np.isfinite(values))
        or np.any(values < 0)
        or np.any(np.diff(values) >= 0)
    ):
        raise ValueError(
            "Evolution redshifts must be finite, nonnegative, and strictly descending"
        )
    return values


def evolution_manifest(redshifts: np.ndarray, sampling: str) -> dict:
    """Portable frame metadata; calculations must evaluate these exact values."""
    values = np.asarray(redshifts, dtype=float)
    if sampling not in SAMPLING_RULES or values.ndim != 1 or values.size < 2:
        raise ValueError("Invalid evolution frame manifest")
    return {
        "schema_version": "haloforge-evolution-frames-v1",
        "sampling": sampling,
        "redshifts": values.tolist(),
        "scale_factors": (1 / (1 + values)).tolist(),
    }
