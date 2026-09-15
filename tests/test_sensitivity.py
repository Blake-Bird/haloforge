import numpy as np
import pytest

from config.defaults import DEFAULT_PARAMS
from engine.sensitivity import controlled_sensitivity, smallest_saved_counterfactual


def _run(name, n_s, sigma):
    return {
        "name": name,
        "params": {**DEFAULT_PARAMS, "n_s": n_s},
        "arrays": {
            "M_h": np.array([1e10, 1e12]),
            "redshifts": np.array([0.0]),
            "sigma_by_z": np.array([[2.0, sigma]]),
            "sigma8_pipeline_by_z": np.array([0.8]),
        },
    }


def test_controlled_sensitivity_rejects_confounders_and_reports_finite_response():
    baseline = _run("baseline", 0.965, 1.0)
    valid = _run("tilt", 0.99, 1.1)
    confounded = _run("tilt plus matter", 0.98, 1.2)
    confounded["params"]["Omega_m"] = 0.32
    result = controlled_sensitivity(
        baseline, [valid, confounded], "n_s", "σ(M)", 1e12, 0
    )
    assert len(result["rows"]) == 1
    assert result["rows"][0]["observable_fractional_change"] == pytest.approx(0.1)
    assert "one-parameter" in result["excluded"][0]["reason"]


def test_counterfactual_selects_smallest_saved_change_that_reaches_target():
    report = {
        "rows": [
            {
                "run": "small",
                "parameter_fractional_change": 0.02,
                "observable_fractional_change": 0.08,
            },
            {
                "run": "first target",
                "parameter_fractional_change": 0.04,
                "observable_fractional_change": 0.11,
            },
            {
                "run": "larger",
                "parameter_fractional_change": 0.10,
                "observable_fractional_change": 0.3,
            },
        ]
    }
    result = smallest_saved_counterfactual(report, 0.1, "increase")
    assert result["status"] == "saved_match"
    assert result["selected"]["run"] == "first target"
    assert "not an inferred continuous threshold" in result["scope_limit"]


def test_counterfactual_refuses_to_invent_an_unrun_threshold():
    result = smallest_saved_counterfactual(
        {
            "rows": [
                {
                    "parameter_fractional_change": 0.02,
                    "observable_fractional_change": 0.01,
                }
            ]
        },
        0.1,
    )
    assert result["status"] == "not_reached"
    assert "does not interpolate" in result["scope_limit"]
    with pytest.raises(ValueError):
        smallest_saved_counterfactual({"rows": []}, 0)
