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


@pytest.mark.parametrize("mass", [1e9, 1e13, 0, np.nan, np.inf])
def test_sensitivity_never_clamps_a_requested_mass_to_the_saved_domain(mass):
    from engine.sensitivity import sigma_at_mass

    with pytest.raises(ValueError):
        sigma_at_mass(_run("baseline", 0.965, 1.0), mass, 0.0)


def test_candidate_with_insufficient_mass_coverage_is_excluded():
    baseline = _run("baseline", 0.965, 1.0)
    candidate = _run("short grid", 0.99, 1.1)
    candidate["arrays"]["M_h"] = np.array([1e10, 1e11])
    result = controlled_sensitivity(baseline, [candidate], "n_s", "σ(M)", 1e12)
    assert result["rows"] == []
    assert "stored domain" in result["excluded"][0]["reason"]


@pytest.mark.parametrize("redshifts", [[], [np.nan], [0.0, 0.0], [1.0, 0.0]])
def test_malformed_redshift_grids_raise_readable_errors(redshifts):
    from engine.sensitivity import sigma_at_mass, sigma8_at_redshift

    run = _run("invalid", 0.965, 1.0)
    run["arrays"]["redshifts"] = np.asarray(redshifts)
    for evaluate in (
        lambda: sigma_at_mass(run, 1e12, 0),
        lambda: sigma8_at_redshift(run, 0),
    ):
        with pytest.raises(ValueError, match="redshift"):
            evaluate()


@pytest.mark.parametrize("value", [np.nan, np.inf, 0, -1])
def test_invalid_sigma8_cannot_become_a_sensitivity_result(value):
    baseline = _run("baseline", 0.965, 1.0)
    candidate = _run("invalid", 0.99, 1.1)
    candidate["arrays"]["sigma8_pipeline_by_z"] = np.array([value])
    report = controlled_sensitivity(baseline, [candidate], "n_s", "σ₈")
    assert not report["rows"]
    assert "positive finite sigma8" in report["excluded"][0]["reason"]


def test_nonfinite_parameter_is_rejected_for_baseline_and_candidate():
    with pytest.raises(ValueError):
        controlled_sensitivity(_run("invalid", np.nan, 1.0), [], "n_s", "σ₈")
    report = controlled_sensitivity(
        _run("baseline", 0.965, 1.0), [_run("invalid", np.inf, 1.0)], "n_s", "σ₈"
    )
    assert not report["rows"]
    assert "finite" in report["excluded"][0]["reason"]


def test_log_interpolation_recovers_an_analytic_mass_power_law():
    from engine.sensitivity import sigma_at_mass

    run = _run("power law", 0.965, 1.0)
    run["arrays"]["sigma_by_z"] = (run["arrays"]["M_h"] / 1e10)[None, :] ** -0.25
    assert sigma_at_mass(run, 1e11, 0) == pytest.approx(10**-0.25, rel=1e-12)


def test_counterfactual_tie_uses_nearest_target_magnitude_for_either_direction():
    report = {
        "rows": [
            {
                "run": "increase",
                "parameter_fractional_change": 0.02,
                "observable_fractional_change": 0.15,
            },
            {
                "run": "decrease",
                "parameter_fractional_change": -0.02,
                "observable_fractional_change": -0.11,
            },
        ]
    }
    assert (
        smallest_saved_counterfactual(report, 0.1, "either")["selected"]["run"]
        == "decrease"
    )


def test_zero_curvature_baseline_reports_absolute_response_without_percentage():
    baseline = _run("flat", 0.965, 1.0)
    candidate = _run("curved", 0.965, 1.1)
    candidate["params"]["Omega_k"] = 0.01
    report = controlled_sensitivity(baseline, [candidate], "Omega_k", "σ(M)", 1e12)
    row = report["rows"][0]
    assert row["parameter_absolute_change"] == 0.01
    assert row["parameter_fractional_change"] is None
    assert row["finite_response_ratio"] is None
    assert row["fractional_response_per_parameter_unit"] == pytest.approx(10.0)
    assert smallest_saved_counterfactual(report, 0.05)["selected"]["run"] == "curved"
