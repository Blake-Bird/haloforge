"""Tests for experiment planner starting sources and controlled parameter diffs."""

from config.defaults import DEFAULT_PARAMS
from engine.experiment_design import STARTING_SOURCES, design_experiment


def test_starting_sources_defined():
    assert "active" in STARTING_SOURCES
    assert "named_baseline" in STARTING_SOURCES
    assert "canonical_preset" in STARTING_SOURCES


def test_tilt_experiment_starts_cleanly_from_canonical_preset():
    plan = design_experiment(
        "Tilt and low-mass structure",
        DEFAULT_PARAMS,
        starting_source="canonical_preset",
    )
    assert plan["parameter_diff"] == {
        "n_s": {"baseline": DEFAULT_PARAMS["n_s"], "candidate": 0.99}
    }
    assert "Planck 2018" in plan["starting_point"]


def test_experiment_with_named_baseline_includes_name():
    plan = design_experiment(
        "Numerical coverage at low mass",
        DEFAULT_PARAMS,
        starting_source="named_baseline",
        baseline_name="My Reference Baseline",
    )
    assert "My Reference Baseline" in plan["starting_point"]
    assert "k_max" in plan["parameter_diff"]
