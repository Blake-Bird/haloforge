"""Comparison of sampled curves without extrapolation or fabricated ratios."""

from __future__ import annotations

import numpy as np


def _validated_curve(x, y) -> tuple[np.ndarray, np.ndarray]:
    x, y = (np.asarray(value, dtype=float) for value in (x, y))
    if x.ndim != 1 or y.shape != x.shape or x.size < 2:
        raise ValueError(
            "Each curve needs at least two matching one-dimensional samples"
        )
    if not np.all(np.isfinite(x)) or np.any(np.diff(x) <= 0):
        raise ValueError("Curve coordinates must be finite and strictly increasing")
    if not np.all(np.isfinite(y)):
        raise ValueError("Curve values must be finite")
    return x, y


def sample_curve_at(x, y, point: float) -> dict:
    """Sample inside one stored curve without extrapolating beyond its domain.

    Positive coordinate/value pairs use log-log interpolation, matching
    :func:`transform_curve`.  The result states whether it was an observed
    sample or an interpolated value so UI callers cannot overstate precision.
    """
    x, y = _validated_curve(x, y)
    point = float(point)
    if not np.isfinite(point) or point < x[0] or point > x[-1]:
        raise ValueError(
            f"Requested point must lie inside the stored domain {x[0]:.6g} to {x[-1]:.6g}"
        )
    match = np.flatnonzero(np.isclose(x, point, rtol=1e-12, atol=0.0))
    if match.size:
        return {
            "point": point,
            "value": float(y[match[0]]),
            "method": "observed stored sample",
        }
    positive = point > 0 and np.all(x > 0) and np.all(y > 0)
    value = (
        np.exp(np.interp(np.log(point), np.log(x), np.log(y)))
        if positive
        else np.interp(point, x, y)
    )
    return {
        "point": point,
        "value": float(value),
        "method": "log-log interpolation inside stored domain"
        if positive
        else "linear interpolation inside stored domain",
    }


def compare_at_point(x, y, bx, by, point: float) -> dict:
    """Return a bounded candidate/baseline ratio at one physical coordinate."""
    candidate = sample_curve_at(x, y, point)
    baseline = sample_curve_at(bx, by, point)
    denominator = baseline["value"]
    if denominator == 0:
        raise ValueError(
            "The baseline value is zero at this point, so a ratio is undefined"
        )
    ratio = candidate["value"] / denominator
    return {
        "point": float(point),
        "candidate_value": candidate["value"],
        "baseline_value": denominator,
        "ratio": float(ratio),
        "percent_difference": float((ratio - 1) * 100),
        "candidate_method": candidate["method"],
        "baseline_method": baseline["method"],
        "scope_limit": "This is a bounded comparison of stored curves. It does not quantify model, calibration, or theory uncertainty.",
    }


def transform_curve(x, y, bx, by, mode):
    """Interpolate only inside the baseline domain; undefined ratios are NaN.

    Positive baseline data uses log-log interpolation. Other data uses linear
    interpolation. Overlay preserves the candidate, including its full domain.
    """
    x, y = _validated_curve(x, y)
    bx, by = _validated_curve(bx, by)
    if mode == "Overlay":
        return y.copy()
    if mode not in {"Ratio", "Fractional difference", "Percent difference"}:
        raise ValueError(f"Unknown comparison mode: {mode}")
    base = np.full(x.shape, np.nan)
    inside = (x >= bx[0]) & (x <= bx[-1])
    if np.all(bx > 0) and np.all(np.isfinite(by)) and np.all(by > 0):
        base[inside] = np.exp(np.interp(np.log(x[inside]), np.log(bx), np.log(by)))
    else:
        base[inside] = np.interp(x[inside], bx, by)
    ratio = np.full(x.shape, np.nan)
    valid = np.isfinite(y) & np.isfinite(base) & (base != 0)
    with np.errstate(over="ignore", invalid="ignore"):
        np.divide(y, base, out=ratio, where=valid)
        result = ratio if mode == "Ratio" else ratio - 1
        if mode == "Percent difference":
            result *= 100
    result[~np.isfinite(result)] = np.nan
    return result
