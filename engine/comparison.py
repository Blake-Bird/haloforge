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


def _interpolate_curve(x, y, points):
    """Use log-log interpolation on each positive bracket, linear otherwise.

    A zero at the rare-tail endpoint must not change interpolation throughout
    the positive part of a mass function. Exact stored samples retain their
    original values, including underflow zeros.
    """
    points = np.asarray(points, dtype=float)
    indices = np.clip(np.searchsorted(x, points, side="right") - 1, 0, x.size - 2)
    left, right = x[indices], x[indices + 1]
    low, high = y[indices], y[indices + 1]
    values = np.interp(points, x, y)
    positive = (left > 0) & (low > 0) & (high > 0)
    if np.any(positive):
        fraction = (np.log(points[positive]) - np.log(left[positive])) / (
            np.log(right[positive]) - np.log(left[positive])
        )
        values[positive] = np.exp(
            (1 - fraction) * np.log(low[positive]) + fraction * np.log(high[positive])
        )
    values = np.where(points == left, low, values)
    values = np.where(points == right, high, values)
    return values, positive


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
    values, positive = _interpolate_curve(x, y, np.asarray([point]))
    return {
        "point": point,
        "value": float(values[0]),
        "method": "log-log interpolation inside stored domain"
        if positive[0]
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
    if not np.isfinite(ratio) or not np.isfinite((ratio - 1) * 100):
        raise ValueError("The comparison exceeds floating-point range at this point")
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
    if mode not in {
        "Ratio",
        "Fractional difference",
        "Percent difference",
        "Residual",
        "Standardized residual",
    }:
        raise ValueError(f"Unknown comparison mode: {mode}")
    base = np.full(x.shape, np.nan)
    inside = (x >= bx[0]) & (x <= bx[-1])
    base[inside], _ = _interpolate_curve(bx, by, x[inside])

    if mode == "Residual":
        diff = np.full(x.shape, np.nan)
        valid = np.isfinite(y) & np.isfinite(base)
        diff[valid] = y[valid] - base[valid]
        diff[~np.isfinite(diff)] = np.nan
        return diff

    if mode == "Standardized residual":
        res = np.full(x.shape, np.nan)
        valid = np.isfinite(y) & np.isfinite(base) & (np.abs(base) > 0)
        res[valid] = (y[valid] - base[valid]) / np.sqrt(np.abs(base[valid]))
        res[~np.isfinite(res)] = np.nan
        return res

    ratio = np.full(x.shape, np.nan)
    valid = np.isfinite(y) & np.isfinite(base) & (base != 0)
    with np.errstate(over="ignore", invalid="ignore"):
        np.divide(y, base, out=ratio, where=valid)
        result = ratio if mode == "Ratio" else ratio - 1
        if mode == "Percent difference":
            result *= 100
    result[~np.isfinite(result)] = np.nan
    return result
