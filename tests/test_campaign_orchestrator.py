"""Tests for Multi-Cosmology Campaign Orchestrator."""

import pytest

from engine.campaign import (
    latin_hypercube_campaign,
    sobol_campaign,
    estimate_campaign_resources,
)
from engine.campaign_orchestrator import (
    JobStatus,
    create_campaign,
    execute_campaign_step,
    campaign_to_dataframe,
    campaign_response_figure,
    campaign_parallel_coordinates,
    compute_campaign_sensitivities,
)


def test_campaign_resource_estimates():
    est = estimate_campaign_resources(16, worker_count=4)
    assert est.total_runs == 16
    assert est.estimated_wall_seconds < est.estimated_cpu_seconds
    assert est.recommended_workers >= 1


def test_latin_hypercube_sampling():
    bounds = {"H0": (65.0, 75.0), "Omega_m": (0.28, 0.35)}
    samples = latin_hypercube_campaign(bounds, 8, seed=123)
    assert len(samples) == 8
    for s in samples:
        assert 65.0 <= s["H0"] <= 75.0
        assert 0.28 <= s["Omega_m"] <= 0.35


def test_sobol_sampling():
    bounds = {"n_s": (0.94, 0.98), "H0": (66.0, 72.0)}
    samples = sobol_campaign(bounds, 8, seed=42)
    assert len(samples) == 8
    for s in samples:
        assert 0.94 <= s["n_s"] <= 0.98


def test_campaign_orchestration_lifecycle():
    combos = [
        {"H0": 67.0, "n_s": 0.96},
        {"H0": 70.0, "n_s": 0.96},
        {"H0": 73.0, "n_s": 0.96},
    ]
    camp = create_campaign("Test Sweep", "1D Grid", combos)
    assert len(camp.members) == 3
    assert camp.members[0].status == JobStatus.PENDING

    # Mock evaluator
    def mock_eval(params):
        return {
            "sigma8": float(params["H0"]) * 0.01 + 0.1,
            "Omega_m": 0.3,
            "h": params["H0"] / 100.0,
        }

    # Step execution
    progressed = execute_campaign_step(camp, mock_eval, max_steps=2)
    assert progressed is True
    assert camp.members[0].status == JobStatus.COMPLETED
    assert camp.members[1].status == JobStatus.COMPLETED
    assert camp.members[2].status == JobStatus.PENDING

    # Complete remaining
    execute_campaign_step(camp, mock_eval, max_steps=2)
    assert camp.members[2].status == JobStatus.COMPLETED

    df = campaign_to_dataframe(camp)
    assert len(df) == 3
    assert "sigma8" in df.columns

    sens = compute_campaign_sensitivities(df, "sigma8")
    assert len(sens) >= 1
    assert sens[0]["parameter"] == "H0"
    assert sens[0]["correlation"] == pytest.approx(1.0, rel=1e-3)

    fig = campaign_response_figure(df, "H0", "sigma8")
    assert fig.data is not None

    par_fig = campaign_parallel_coordinates(df, ["H0", "n_s"], "sigma8")
    assert par_fig.data is not None
