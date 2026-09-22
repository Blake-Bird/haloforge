"""Deterministic MP4, WebM, and GIF export for precomputed evolution frames."""

from __future__ import annotations

from typing import Any

import imageio.v3 as iio
import numpy as np
import plotly.io as pio

from engine.evolution_studio import build_evolution_figure


VIDEO_FORMATS = {"mp4": "video/mp4", "webm": "video/webm", "gif": "image/gif"}


def render_evolution_video(
    frames: list[dict[str, Any]],
    observable: str,
    *,
    video_format: str,
    fps: float = 2.0,
    width: int = 1280,
    height: int = 720,
    theme: str = "Dark",
) -> bytes:
    """Render a reproducible video directly from the supplied frame payload.

    The caller owns the scientific frame generation. This exporter deliberately
    performs no interpolation, data mutation, or autoscaling beyond the fixed
    axes requested from the evolution figure builder.
    """
    fmt = video_format.lower().lstrip(".")
    if fmt not in VIDEO_FORMATS:
        raise ValueError(f"Unsupported video format: {video_format}")
    if not frames:
        raise ValueError("Cannot render an empty evolution sequence")
    if not np.isfinite(fps) or not 0.1 <= float(fps) <= 60.0:
        raise ValueError("Video fps must be between 0.1 and 60")
    if not (320 <= int(width) <= 3840 and 240 <= int(height) <= 2160):
        raise ValueError("Video dimensions must be within 320×240 and 3840×2160")

    pixels = []
    for index in range(len(frames)):
        figure = build_evolution_figure(
            frames,
            index,
            observable,
            fixed_axes=True,
            theme=theme,
        )
        png = pio.to_image(figure, format="png", width=int(width), height=int(height), scale=1)
        pixels.append(iio.imread(png, extension=".png"))
    sequence = np.stack(pixels)

    options: dict[str, Any] = {"extension": f".{fmt}", "fps": float(fps)}
    if fmt == "mp4":
        options.update(codec="libx264", pixelformat="yuv420p")
    elif fmt == "webm":
        options.update(codec="libvpx-vp9")
    else:
        options.update(loop=0)
    return bytes(iio.imwrite("<bytes>", sequence, **options))
