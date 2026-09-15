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
        "Computed precisely",
        "Numerically converged",
        "Suitable for publication",
    }
    assert len(record["uncertainties"]) >= 10


def test_saved_run_has_machine_readable_validity_even_when_hmf_is_ineligible():
    run = create_run_from_current_state(_current("Gaussian"), "Variance-only")
    validity = run["scientific_validity"]
    assert validity["schema_version"] == VALIDITY_SCHEMA_VERSION
    assert validity["hmf_contract"]["all_evaluated_points_calibrated"] is False
    assert validity["overall_state"] == "not_computed"
