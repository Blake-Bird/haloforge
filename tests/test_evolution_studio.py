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


def test_evolution_uses_stored_multi_redshift_power(sample_power):
    k, p0 = sample_power
    zs = np.array([0.0, 1.0, 2.0])
    # Deliberately non-growth-like source spectra: the middle frame must be
    # inherited exactly from the supplied CLASS/AxiCLASS grid, not recreated
    # from the fallback growth approximation.
    p_by_z = np.vstack([p0, p0 * 0.37, p0 * 0.11])
    frames = calculate_evolution_frames(
        k,
        p0,
        zs,
        None,
        DEFAULT_PARAMS,
        power_by_z=p_by_z,
        source_redshifts=zs,
    )
    assert np.allclose(frames[1]["P_k"], p_by_z[1])
    assert frames[1]["power_source"].startswith("exact stored CLASS")
    assert frames[1]["scientific_status"] == "calculated_linear_theory"

    interpolated = calculate_evolution_frames(
        k,
        p0,
        np.array([1.5]),
        None,
        DEFAULT_PARAMS,
        power_by_z=p_by_z,
        source_redshifts=zs,
    )[0]
    assert interpolated["power_source"].startswith("log P")
    assert interpolated["scientific_status"] == "approximation"
    assert interpolated["interpolation_validation_median_fractional_error"] is not None


def test_ede_evolution_uses_supplied_solver_cosmic_time_or_marks_it_unavailable(
    sample_power,
):
    k, p0 = sample_power
    zs = np.array([2.0, 1.0, 0.0])
    params = dict(DEFAULT_PARAMS, enable_ede=True)
    frames = calculate_evolution_frames(
        k,
        p0,
        zs,
        None,
        params,
        power_by_z=np.vstack([p0 * 0.11, p0 * 0.37, p0]),
        source_redshifts=zs,
        cosmic_time_gyr_by_z=np.array([3.3, 5.9, 13.8]),
    )
    assert frames[0]["cosmic_time_gyr"] == pytest.approx(3.3)
    assert frames[0]["cosmic_time_source"].startswith("stored CLASS/AxiCLASS")

    without_time = calculate_evolution_frames(
        k,
        p0,
        zs,
        None,
        params,
        power_by_z=np.vstack([p0 * 0.11, p0 * 0.37, p0]),
        source_redshifts=zs,
    )
    assert without_time[0]["cosmic_time_gyr"] is None
    assert "EDE requires" in without_time[0]["cosmic_time_source"]


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
