"""Fourier-space smoothing window functions."""

from __future__ import annotations

import numpy as np


def top_hat_W_series(y: np.ndarray | float) -> np.ndarray:
    """Taylor expansion for the real-space top-hat window near y=0."""
    y_arr = np.asarray(y, dtype=float)
    return (
        1.0
        - y_arr**2 / 10.0
        + y_arr**4 / 280.0
        - y_arr**6 / 15120.0
        + y_arr**8 / 1330560.0
        - y_arr**10 / 172972800.0
    )


def top_hat_W_exact(y: np.ndarray | float) -> np.ndarray:
    """Exact W(y)=3*(sin(y)-y*cos(y))/y^3, with y=0 filled as 1."""
    y_arr = np.asarray(y, dtype=float)
    out = np.empty_like(y_arr, dtype=float)
    mask = y_arr != 0.0
    out[~mask] = 1.0
    ym = y_arr[mask]
    out[mask] = 3.0 * (np.sin(ym) - ym * np.cos(ym)) / ym**3
    return out


def top_hat_W(y: np.ndarray | float) -> np.ndarray:
    """Taylor-safe real-space top-hat window."""
    y_arr = np.asarray(y, dtype=float)
    return np.where(np.abs(y_arr) < 0.1, top_hat_W_series(y_arr), top_hat_W_exact(y_arr))


def gaussian_W(y: np.ndarray | float) -> np.ndarray:
    """Gaussian W(y)=exp(-y^2/2)."""
    return np.exp(-0.5 * np.asarray(y, dtype=float) ** 2)


def sharp_k_W(y: np.ndarray | float) -> np.ndarray:
    """Sharp-k W(y)=1 for y<=1 and 0 otherwise."""
    return (np.asarray(y, dtype=float) <= 1.0).astype(float)


def window_W(y: np.ndarray | float, window_type: str) -> np.ndarray:
    """Dispatch a named window."""
    if window_type == "Gaussian":
        return gaussian_W(y)
    if window_type == "Sharp-k":
        return sharp_k_W(y)
    return top_hat_W(y)


def window_squared(y: np.ndarray | float, window_type: str) -> np.ndarray:
    """Return W(y)^2 for the requested smoothing window."""
    w = window_W(y, window_type)
    return w * w


W_top_hat = top_hat_W

