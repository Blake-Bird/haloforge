from copy import deepcopy

from config.defaults import DEFAULT_PARAMS
from config.ranges import CONTROL_RANGES
from engine.cosmology import parse_custom_redshifts
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


def test_custom_redshift_parser_reports_nonfinite_values_before_staging():
    values, issues = parse_custom_redshifts("0, nan, inf, -inf, -1, 2.5")
    assert values == [0, 2.5]
    assert len(issues) == 4


def test_density_controls_keep_the_physical_default_but_use_human_scale_steps():
    omega_m = CONTROL_RANGES["Omega_m"]
    omega_b = CONTROL_RANGES["Omega_b"]

    assert omega_m["min"] <= DEFAULT_PARAMS["Omega_m"] <= omega_m["max"]
    assert omega_m["step"] == 0.001
    # The widget ranges themselves prevent a non-positive cold-dark-matter
    # density; programmatic inputs remain protected by validate_params.
    assert omega_b["max"] < omega_m["min"]


def test_validation_reports_invalid_physical_values_before_solver_launch(monkeypatch):
    import pytest
    from engine.class_runner import compute_matter_power

    def unexpected_launch(*args, **kwargs):
        pytest.fail("Invalid physical inputs must not launch CLASS")

    monkeypatch.setattr("engine.class_runner.subprocess.run", unexpected_launch)
    invalid = {
        "H0": 0,
        "A_s": -1,
        "n_s": float("nan"),
        "k_pivot": 0,
        "Omega_m": -0.1,
        "Omega_b": -0.1,
        "Omega_k": float("inf"),
        "N_eff": -1,
        "Tcmb": 0,
        "tau_reio": -1,
        "single_z": -1,
        "k_points": 5.5,
        "n_EDE": 2.5,
        "log10_a_c": -1000,
        "scf_parameters": "nan,0",
        "enable_ede": "false",
    }
    for key, value in invalid.items():
        params = dict(DEFAULT_PARAMS, **{key: value})
        assert validate_params(params), key
        with pytest.raises(ValueError, match=key):
            compute_matter_power(params)


def test_analysis_settings_are_validated_before_integration():
    for key, value in {
        "delta_c": 0,
        "delta_halo": -200,
        "quad_limit": 1.5,
        "window_type": "unknown",
    }.items():
        assert validate_params(dict(DEFAULT_PARAMS, **{key: value})), key


def test_solver_validation_does_not_confuse_control_ranges_with_physics():
    from engine.class_runner import build_class_settings

    params = dict(DEFAULT_PARAMS, H0=85.0, n_s=1.1, enable_ede=False)
    assert validate_params(params) == []
    settings = build_class_settings(params)
    assert settings["H0"] == 85.0
    assert settings["n_s"] == 1.1


def test_zero_ede_ignores_unused_scalar_field_parameters():
    from engine.class_runner import build_class_settings

    zero = dict(
        DEFAULT_PARAMS,
        f_EDE=0,
        n_EDE="unused",
        log10_a_c="unused",
        scf_parameters="unused",
    )
    assert build_class_settings(zero) == build_class_settings(
        dict(DEFAULT_PARAMS, enable_ede=False)
    )
