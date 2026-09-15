"""Accessible long-form data alternatives for interactive figures."""

from __future__ import annotations

import pandas as pd


def chart_transcript(fig) -> pd.DataFrame:
    """Return every pointwise x/y trace in a screen-reader-friendly table."""
    rows: list[dict] = []
    for index, trace in enumerate(fig.data):
        x = getattr(trace, "x", None)
        y = getattr(trace, "y", None)
        if x is None or y is None:
            continue
        name = str(getattr(trace, "name", None) or f"series {index + 1}")
        for x_value, y_value in zip(x, y):
            rows.append({"series": name, "x": x_value, "y": y_value})
    return pd.DataFrame(rows, columns=["series", "x", "y"])
