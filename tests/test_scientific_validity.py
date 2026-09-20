from copy import deepcopy

import numpy as np

from config.defaults import DEFAULT_PARAMS
from engine.sigma import compute_sigma_result
from engine.validity import VALIDITY_SCHEMA_VERSION, scientific_validity_record
from state.run_model import create_run_from_current_state


def _current(window="Top-hat"):
    params = {**deepcopy(DEFAULT_PARAMS), "enable_ede": False, "window_type": window}
    k = np.geomspace(1e-4, 100, 800)
    power = {
        "k": k,
        "P": 1e4 * k / (1 + (k / 0.1) ** 3),
        "derived": {"h": 0.6781, "sigma8": 0.8},
        "class_status": "TEST_DATA",
    }
    return {
        "params": params,
        "power_result": power,
        "sigma_result": compute_sigma_result(power, params),
    }


def test_validity_record_keeps_separate_claims_and_never_auto_approves_publication():
    record = scientific_validity_record(
        {"arrays": {"k": [0.1]}, "class_status": "AXICLASS"}
    )
    assert record["schema_version"] == VALIDITY_SCHEMA_VERSION
    assert record["overall_state"] != "publication_ready"
    assert {row["claim"] for row in record["claims"]} >= {
        "Solver completed",
        "Numerically converged",
        "Suitable for publication",
    }
    assert len(record["uncertainties"]) >= 10


def test_saved_run_has_machine_readable_validity_even_when_hmf_is_ineligible():
    run = create_run_from_current_state(_current("Gaussian"), "Variance-only")
    validity = run["scientific_validity"]
    assert validity["schema_version"] == VALIDITY_SCHEMA_VERSION
    assert validity["hmf_contract"]["all_evaluated_points_pass_fit_checks"] is False
    assert validity["overall_state"] == "not_computed"


def test_solver_completion_covers_lcdm_and_excludes_failed_integrity():
    for solver in ("CLASS", "AXICLASS"):
        run = {"arrays": {"k": [0.1, 1.0]}, "class_status": solver}
        record = scientific_validity_record(run)
        assert record["overall_state"].startswith("computed_")
        assert record["claims"][0]["claim"] == "Solver completed"
        run["integrity_status"] = {"state": "invalid"}
        assert scientific_validity_record(run)["overall_state"] == "not_computed"


def test_invalid_or_empty_fit_evidence_never_passes_any_inventory():
    from engine.assurance import assurance_report
    from engine.uncertainty import uncertainty_inventory

    run = {"params": {"fitting": "Tinker 2008"}}
    masks = (
        [],
        True,
        [1, 1],
        [float("nan")],
        ["False"],
        [[True, True]],
        [[True], []],
        [True, False],
    )
    for mask in masks:
        evidence = {"calibrated_mask": mask}
        claims = {row["claim"]: row for row in assurance_report(run, evidence)}
        assert claims["Calibrated by simulations"]["state"] != "pass"
        assert not scientific_validity_record(run, evidence)["hmf_contract"][
            "all_evaluated_points_pass_fit_checks"
        ]
        rows = {row["source"]: row for row in uncertainty_inventory(run, evidence)}
        assert rows["HMF fit calibration"]["state"] == "review"


def test_boolean_fit_evidence_is_consistent_across_reports():
    from engine.contracts import all_fit_points_checked

    assert all_fit_points_checked({"calibrated_mask": [True, True]})
    assert all_fit_points_checked({"calibrated_mask": np.ones(3, dtype=bool)})
    assert not all_fit_points_checked(None)


def test_fit_range_checks_do_not_certify_simulation_calibration():
    from engine.assurance import assurance_report

    for fitting in ("Sheth-Tormen 2001", "Tinker 2008"):
        claims = {
            row["claim"]: row
            for row in assurance_report(
                {"params": {"fitting": fitting, "enable_ede": True}},
                {"calibrated_mask": [True, True]},
            )
        }
        assert claims["Calibrated by simulations"]["state"] == "review"
        if fitting.startswith("Sheth"):
            assert "simulation-fitted" in claims["Calibrated by simulations"]["detail"]
