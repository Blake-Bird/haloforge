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
    campaign_counts,
    cancel_pending_members,
    campaign_to_dataframe,
    campaign_response_figure,
    campaign_parallel_coordinates,
    compute_campaign_sensitivities,
    pause_campaign,
    resume_campaign,
    retry_failed_members,
)


def test_campaign_resource_estimates():
    est = estimate_campaign_resources(16, worker_count=4, seconds_per_run=2.0)
    assert est.total_runs == 16
    assert est.estimated_wall_seconds < est.estimated_cpu_seconds
    assert est.recommended_workers >= 1


def test_campaign_resource_estimate_refuses_to_invent_solver_duration():
    est = estimate_campaign_resources(16, worker_count=4)
    assert est.estimated_cpu_seconds is None
    assert est.estimated_wall_seconds is None
    assert "No completed" in est.timing_basis


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


def test_campaign_worker_count_executes_a_bounded_batch():
    camp = create_campaign(
        "Parallel test", "grid", [{"H0": 66.0}, {"H0": 67.0}, {"H0": 68.0}]
    )

    def evaluator(params):
        return {
            "sigma8": 0.8,
            "Omega_m": 0.3,
            "h": params["H0"] / 100.0,
            "class_status": "CLASS",
        }

    assert execute_campaign_step(camp, evaluator, max_steps=3, worker_count=2)
    assert all(member.status == JobStatus.COMPLETED for member in camp.members)
    assert all(member.evidence == {"class_status": "CLASS"} for member in camp.members)


def test_campaign_executor_passes_only_the_declared_member_delta():
    camp = create_campaign("Baseline contract", "grid", [{"H0": 67.0}])
    received = []

    def evaluator(delta):
        received.append(delta)
        return {"sigma8": 0.8, "Omega_m": 0.3, "h": 0.67}

    execute_campaign_step(camp, evaluator, max_steps=1)
    assert received == [{"H0": 67.0}]


def test_campaign_pause_cancel_retry_lifecycle():
    camp = create_campaign("Lifecycle", "grid", [{"H0": 67.0}, {"H0": 68.0}])
    pause_campaign(camp)
    assert camp.metadata["lifecycle_state"] == "paused"
    assert not execute_campaign_step(
        camp,
        lambda _: {"sigma8": 0.8, "Omega_m": 0.3, "h": 0.67},
    )
    resume_campaign(camp)
    camp.members[0].status = JobStatus.FAILED
    camp.members[0].error_message = "solver failed"
    assert retry_failed_members(camp) == 1
    assert campaign_counts(camp)["pending"] == 2
    assert cancel_pending_members(camp) == 2
    assert campaign_counts(camp)["cancelled"] == 2
