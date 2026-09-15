"""Conservative, data-derived annotations for scientific plots."""

from __future__ import annotations

import numpy as np


def peak_point(x, y) -> dict | None:
    """Return the finite positive peak without interpolating or smoothing it."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    if x.ndim != 1 or y.shape != x.shape or not np.any(valid):
        return None
    index = int(np.nanargmax(np.where(valid, y, np.nan)))
    return {"index": index, "x": float(x[index]), "y": float(y[index])}


def turning_points(x, y) -> list[dict]:
    """Return sampled local extrema, never a fitted or smoothed estimate.

    Flat samples and non-finite triples are skipped: calling either an
    extremum would create a visual claim unsupported by the stored samples.
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.ndim != 1 or y.shape != x.shape or x.size < 3:
        return []
    points = []
    for index in range(1, x.size - 1):
        left, center, right = y[index - 1 : index + 2]
        if not (
            np.isfinite(x[index])
            and np.isfinite(left)
            and np.isfinite(center)
            and np.isfinite(right)
        ):
            continue
        left_slope, right_slope = center - left, right - center
        if left_slope > 0 and right_slope < 0:
            kind = "sampled local maximum"
        elif left_slope < 0 and right_slope > 0:
            kind = "sampled local minimum"
        else:
            continue
        points.append(
            {"index": index, "x": float(x[index]), "y": float(center), "kind": kind}
        )
    return points


def reference_crossings(x, y, *, reference: float) -> list[dict]:
    """Locate observed reference hits and sampled brackets around crossings.

    Brackets deliberately report their two stored endpoints instead of an
    interpolated crossing coordinate. This keeps a visual crossing cue from
    implying more numerical precision than the curves contain.
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    reference = float(reference)
    if x.ndim != 1 or y.shape != x.shape or x.size < 1 or not np.isfinite(reference):
        return []
    crossings = []
    seen_points: set[int] = set()
    for index, value in enumerate(y):
        if np.isfinite(x[index]) and np.isfinite(value) and value == reference:
            crossings.append(
                {
                    "kind": "observed reference crossing",
                    "index": index,
                    "x": float(x[index]),
                    "y": float(value),
                }
            )
            seen_points.add(index)
    for index in range(x.size - 1):
        left_x, right_x, left_y, right_y = (
            x[index],
            x[index + 1],
            y[index],
            y[index + 1],
        )
        if not (
            np.isfinite(left_x)
            and np.isfinite(right_x)
            and np.isfinite(left_y)
            and np.isfinite(right_y)
            and right_x > left_x
        ):
            continue
        left_delta, right_delta = left_y - reference, right_y - reference
        if left_delta * right_delta < 0:
            crossings.append(
                {
                    "kind": "sampled crossing bracket",
                    "left_index": index,
                    "right_index": index + 1,
                    "x_lower": float(left_x),
                    "x_upper": float(right_x),
                    "reference": reference,
                }
            )
    return crossings


def validity_boundaries(x, calibrated_mask) -> list[float]:
    """Return sampled x locations at which a fit-calibration state changes."""
    x = np.asarray(x, dtype=float)
    mask = np.asarray(calibrated_mask, dtype=bool)
    if x.ndim != 1 or mask.shape != x.shape or x.size < 2:
        return []
    return [
        float(x[index])
        for index in np.flatnonzero(mask[1:] != mask[:-1]) + 1
        if np.isfinite(x[index])
    ]


def largest_deviation_point(x, values, *, reference: float) -> dict | None:
    """Locate the largest finite sampled departure from a stated reference."""
    x, values = np.asarray(x, dtype=float), np.asarray(values, dtype=float)
    valid = np.isfinite(x) & np.isfinite(values)
    if x.ndim != 1 or values.shape != x.shape or not np.any(valid):
        return None
    index = int(np.nanargmax(np.where(valid, np.abs(values - reference), np.nan)))
    return {
        "index": index,
        "x": float(x[index]),
        "value": float(values[index]),
        "absolute_deviation": float(abs(values[index] - reference)),
    }
