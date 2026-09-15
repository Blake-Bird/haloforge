from copy import deepcopy

from config.defaults import DEFAULT_PARAMS
from config.ranges import CONTROL_RANGES
from state.session import validate_params


def test_parameter_validation_reports_malformed_external_values_without_crashing():
    params = deepcopy(DEFAULT_PARAMS)
    params.update(
        {
            "Omega_b": "not-a-number",
            "k_min": None,
            "k_points": 50.5,
            "mass_points": float("nan"),
            "z_values": [0, "not-a-redshift"],
            "enable_ede": True,
            "f_EDE": float("inf"),
        }
    )
    errors = validate_params(params)
    assert any("Omega_b must be a finite numeric value" in error for error in errors)
    assert any("k_min must be a finite numeric value" in error for error in errors)
    assert any("At least 50 k samples" in error for error in errors)
    assert any(
        "mass_points must be a finite numeric value" in error for error in errors
    )
    assert "Redshifts must be non-negative." in errors
    assert "f_EDE must lie between 0 and 0.3." in errors


def test_parameter_validation_preserves_valid_default_configuration():
    assert validate_params(deepcopy(DEFAULT_PARAMS)) == []


def test_density_controls_keep_the_physical_default_but_use_human_scale_steps():
    omega_m = CONTROL_RANGES["Omega_m"]
    omega_b = CONTROL_RANGES["Omega_b"]

    assert omega_m["min"] <= DEFAULT_PARAMS["Omega_m"] <= omega_m["max"]
    assert omega_m["step"] == 0.001
    # The widget ranges themselves prevent a non-positive cold-dark-matter
    # density; programmatic inputs remain protected by validate_params.
    assert omega_b["max"] < omega_m["min"]
