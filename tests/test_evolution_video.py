import numpy as np

from config.defaults import DEFAULT_PARAMS
from engine.evolution_studio import calculate_evolution_frames
from engine.evolution_video import render_evolution_video


def test_render_evolution_video_mp4():
    k = np.geomspace(1e-3, 1.0, 32)
    p0 = 1e4 * k / (1.0 + k)
    frames = calculate_evolution_frames(
        k, p0, np.array([1.0, 0.0]), None, DEFAULT_PARAMS
    )
    output = render_evolution_video(
        frames,
        "Matter Power P(k)",
        video_format="mp4",
        fps=1.0,
        width=320,
        height=240,
    )
    assert output[4:8] == b"ftyp"
    assert len(output) > 1000
