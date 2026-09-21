from config.defaults import DEFAULT_PARAMS
from engine.experiment_design import PLANS, design_experiment


def test_experiment_design_has_a_visible_parameter_diff_and_caveat():
    plan = design_experiment("Tilt and low-mass structure", DEFAULT_PARAMS)
    assert plan["parameter_diff"]["n_s"]["candidate"] == 0.99
    assert plan["caveat"]
    assert "not active learning" in plan["scope_limit"]


def test_controlled_plans_change_only_their_declared_parameter_from_any_start():
    ede_start = {**DEFAULT_PARAMS, "enable_ede": True}
    for goal in ("Tilt and low-mass structure", "Numerical coverage at low mass"):
        plan = design_experiment(goal, ede_start)
        assert set(plan["parameter_diff"]) == set(plan["candidate_parameters"])
        assert len(plan["parameter_diff"]) <= 1
        assert plan["starting_point"].startswith("The current staged")


def test_all_named_plans_are_constructible():
    assert all(design_experiment(goal, DEFAULT_PARAMS)["inspect"] for goal in PLANS)
