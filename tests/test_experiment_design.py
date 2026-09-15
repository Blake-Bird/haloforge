from config.defaults import DEFAULT_PARAMS
from engine.experiment_design import PLANS, design_experiment


def test_experiment_design_has_a_visible_parameter_diff_and_caveat():
    plan = design_experiment("Tilt and low-mass structure", DEFAULT_PARAMS)
    assert plan["parameter_diff"]["n_s"]["candidate"] == 0.99
    assert plan["caveat"]
    assert "not active learning" in plan["scope_limit"]


def test_all_named_plans_are_constructible():
    assert all(design_experiment(goal, DEFAULT_PARAMS)["inspect"] for goal in PLANS)
