"""Deterministic linear Gaussian structure-field utilities, independent of UI."""

from __future__ import annotations

import numpy as np


def shared_fourier_seed(
    box_mpc: float, grid_size: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Generate reproducible Fourier phases and their physical wavenumbers."""
    box, n = float(box_mpc), int(grid_size)
    if not np.isfinite(box) or box <= 0 or n < 2:
        raise ValueError(
            "Box size must be finite and positive; grid size must be at least 2"
        )
    rng = np.random.default_rng(int(seed))
    modes = np.fft.rfftn(rng.normal(size=(n, n, n)))
    kx = 2 * np.pi * np.fft.fftfreq(n, d=box / n)
    ky = 2 * np.pi * np.fft.fftfreq(n, d=box / n)
    kz = 2 * np.pi * np.fft.rfftfreq(n, d=box / n)
    kk = np.sqrt(
        kx[:, None, None] ** 2 + ky[None, :, None] ** 2 + kz[None, None, :] ** 2
    )
    return modes, kk


def gaussian_field_slice(
    k_mpc_inv,
    power_mpc3,
    box_mpc: float,
    grid_size: int,
    smoothing_mpc: float,
    modes,
    kk,
) -> np.ndarray:
    """Filter shared phases with a stored linear spectrum and return one slice.

    This is a linear Gaussian illustration, not an N-body realization or halo
    catalogue. Power interpolation is bounded to the supplied positive grid.
    """
    k, power = np.asarray(k_mpc_inv, float), np.asarray(power_mpc3, float)
    box, n, smoothing = float(box_mpc), int(grid_size), float(smoothing_mpc)
    modes, kk = np.asarray(modes), np.asarray(kk, float)
    if (
        k.ndim != 1
        or power.shape != k.shape
        or k.size < 2
        or np.any(k <= 0)
        or np.any(power <= 0)
        or np.any(np.diff(k) <= 0)
    ):
        raise ValueError(
            "A positive, strictly increasing stored power grid is required"
        )
    expected = (n, n, n // 2 + 1)
    if (
        modes.shape != expected
        or kk.shape != expected
        or not np.isfinite(box)
        or box <= 0
        or not np.isfinite(smoothing)
        or smoothing < 0
    ):
        raise ValueError(
            "Shared Fourier modes, box size, and smoothing must be compatible and finite"
        )
    amplitude = np.zeros_like(kk, dtype=float)
    mask = (kk >= k[0]) & (kk <= k[-1]) & (kk > 0)
    interpolated = np.exp(np.interp(np.log(kk[mask]), np.log(k), np.log(power)))
    amplitude[mask] = (
        (n / box) ** 1.5
        * np.sqrt(interpolated)
        * np.exp(-0.5 * (kk[mask] * smoothing) ** 2)
    )
    filtered = modes * amplitude
    filtered[0, 0, 0] = 0
    field = np.fft.irfftn(filtered, s=(n, n, n), axes=(0, 1, 2)).real
    field -= field.mean()
    return field[:, :, n // 2]
