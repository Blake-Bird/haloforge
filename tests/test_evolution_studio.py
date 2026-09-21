"""Tests for Redshift Evolution Movie Studio engine."""

import numpy as np
import pytest

from config.defaults import DEFAULT_PARAMS
from engine.evolution import evolution_redshifts
from engine.evolution_studio import (
    calculate_evolution_frames,
    build_evolution_figure,
    evolution_contact_sheet,
    evolution_frame_summary_table,
    cosmic_time_gyr,
)


@pytest.fixture
def sample_power():
    k = np.geomspace(1e-4, 10.0, 100)
    p0 = 1e4 * k / (1.0 + (k / 0.2) ** 3)
    return k, p0


def test_cosmic_time_monotonic():
    t_z20 = cosmic_time_gyr(20.0, DEFAULT_PARAMS)
    t_z0 = cosmic_time_gyr(0.0, DEFAULT_PARAMS)
    assert t_z20 < t_z0
    assert 13.0 < t_z0 < 14.5  # approx 13.8 Gyr at z=0


def test_calculate_evolution_frames(sample_power):
    k, p0 = sample_power
    zs = evolution_redshifts(20.0, 0.0, 6, "uniform_a")
    frames = calculate_evolution_frames(k, p0, zs, None, DEFAULT_PARAMS)
    assert len(frames) == 6
    # Redshifts should descend
    assert frames[0]["redshift"] > frames[-1]["redshift"]
    # Growth should increase towards z=0
    assert frames[0]["growth_factor"] < frames[-1]["growth_factor"]
    # Mass variance at fixed mass should grow with time
    assert frames[0]["sigma"][50] < frames[-1]["sigma"][50]


def test_build_evolution_figure_and_contact_sheet(sample_power):
    k, p0 = sample_power
    zs = evolution_redshifts(10.0, 0.0, 4, "uniform_a")
    frames = calculate_evolution_frames(k, p0, zs, None, DEFAULT_PARAMS)

    for obs in [
        "Matter Power P(k)",
        "Dimensionless Power Δ²(k)",
        "Mass Variance σ(M)",
        "Differential HMF dn/dlnM",
        "Cumulative HMF n(>M)",
    ]:
        fig = build_evolution_figure(frames, 0, obs, fixed_axes=True)
        assert fig.data is not None
        assert len(fig.data) >= 1

    sheet = evolution_contact_sheet(frames, "Matter Power P(k)", num_panels=3)
    assert sheet.data is not None
    assert len(sheet.data) == 3


def test_evolution_summary_table(sample_power):
    k, p0 = sample_power
    zs = evolution_redshifts(5.0, 0.0, 3, "uniform_a")
    frames = calculate_evolution_frames(k, p0, zs, None, DEFAULT_PARAMS)
    table = evolution_frame_summary_table(frames)
    assert len(table) == 3
    assert "Redshift z" in table[0]
    assert "Cosmic Age [Gyr]" in table[0]
