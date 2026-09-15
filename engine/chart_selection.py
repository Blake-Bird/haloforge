"""Parse bounded x-ranges from Plotly selection events without UI coupling."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np


def selected_x_bounds(event) -> tuple[float, float] | None:
    """Return finite increasing bounds from a multi-point brush, otherwise None."""
    if event is None:
        return None
    selection = getattr(
        event,
        "selection",
        event.get("selection") if isinstance(event, Mapping) else None,
    )
    if not isinstance(selection, Mapping):
        return None
    points = selection.get("points")
    if not isinstance(points, (list, tuple)):
        return None
    values = []
    for point in points:
        if not isinstance(point, Mapping):
            continue
        try:
            value = float(point["x"])
        except (KeyError, TypeError, ValueError):
            continue
        if np.isfinite(value):
            values.append(value)
    if len(values) < 2:
        return None
    lo, hi = min(values), max(values)
    return (lo, hi) if hi > lo else None
