import numpy as np
import pytest

from config.defaults import DEFAULT_PARAMS
from engine.benchmark import (
    CANONICAL_VALIDATION_CASES,
    CANONICAL_CASES_VERSION,
    adaptive_sigma8,
    assess_canonical_case,
    canonical_case,
    canonical_case_rows,
    internal_sigma8_benchmark,
)
from engine.sigma import sigma_squared


def test_internal_sigma8_benchmark_agrees_on_same_sampled_power():
    k = np.geomspace(1e-5, 1e2, 1200)
    power = 1e4 * k / (1 + (k / 0.2) ** 3)
    h = 0.7
    stored = np.sqrt(sigma_squared(8 / h, k, power, "Top-hat"))
    run = {
        "params": {**DEFAULT_PARAMS, "H0": 70.0},
        "power_result": {
            "k": k,
            "P": power,
            "P_by_z": np.array([power]),
            "redshifts": np.array([0.0]),
            "derived": {"h": h},
        },
        "sigma_result": {
            "sigma8_pipeline_by_z": np.array([stored]),
            "redshifts": np.array([0.0]),
        },
    }
    report = internal_sigma8_benchmark(run)
    assert report["rows"][0]["status"] in {"pass", "review"}
    assert report["rows"][0]["adaptive_reference"] == adaptive_sigma8(k, power, h)


def test_internal_sigma8_benchmark_never_hides_a_failed_discrepancy():
    k = np.geomspace(1e-5, 1e2, 1200)
    power = 1e4 * k / (1 + (k / 0.2) ** 3)
    run = {
        "params": {**DEFAULT_PARAMS, "H0": 70.0},
        "power_result": {
            "k": k,
            "P": power,
            "P_by_z": np.array([power]),
            "redshifts": np.array([0.0]),
            "derived": {"h": 0.7},
        },
        "sigma_result": {
            "sigma8_pipeline_by_z": np.array([999.0]),
            "redshifts": np.array([0.0]),
        },
    }
    assert internal_sigma8_benchmark(run)["rows"][0]["status"] == "fail"


def test_internal_sigma8_benchmark_accepts_the_durable_saved_run_shape():
    k = np.geomspace(1e-5, 1e2, 1200)
    power = 1e4 * k / (1 + (k / 0.2) ** 3)
    h = 0.7
    stored = np.sqrt(sigma_squared(8 / h, k, power, "Top-hat"))
    saved = {
        "params": {**DEFAULT_PARAMS, "H0": 70.0},
        "derived": {"h": h},
        "class_status": "AXICLASS",
        "rho0": 1.0,
        "window_type": "Top-hat",
        "arrays": {
            "k": k,
            "P": power,
            "P_by_z": np.array([power]),
            "redshifts": np.array([0.0]),
            "M_h": np.array([1e10, 1e11, 1e12]),
            "M": np.array([1e10, 1e11, 1e12]),
            "R": np.array([1.0, 2.0, 3.0]),
            "sigma": np.array([2.0, 1.0, 0.5]),
            "sigma_by_z": np.array([[2.0, 1.0, 0.5]]),
            "dlnsigma_dlnM": np.array([-0.2, -0.2, -0.2]),
            "dlnsigma_dlnM_by_z": np.array([[-0.2, -0.2, -0.2]]),
            "sigma8_pipeline_by_z": np.array([stored]),
        },
    }
    assert internal_sigma8_benchmark(saved)["rows"][0]["status"] in {"pass", "review"}


def test_canonical_case_registry_covers_declared_regimes_without_claiming_external_outputs():
    identifiers = {case.identifier for case in CANONICAL_VALIDATION_CASES}
    assert {
        "lcdm-planck-like",
        "ede-linear",
        "curved-lcdm",
        "high-redshift",
        "low-mass-coverage",
        "high-mass-tail",
        "tinker-redshift-boundary",
    } <= identifiers
    case = canonical_case("lcdm-planck-like")
    report = assess_canonical_case({**DEFAULT_PARAMS, **case.parameter_overrides}, case)
    assert report["configured"]
    assert report["suite_version"] == CANONICAL_CASES_VERSION
    assert "not evaluated for this run" in report["reference_status"]


def test_canonical_case_assessment_exposes_configuration_mismatches():
    report = assess_canonical_case(
        DEFAULT_PARAMS, canonical_case("tinker-redshift-boundary")
    )
    assert not report["configured"]
    assert {item["parameter"] for item in report["mismatches"]} >= {
        "fitting",
        "single_z",
    }
    assert len(canonical_case_rows(DEFAULT_PARAMS)) == len(CANONICAL_VALIDATION_CASES)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf, 0.0, -1.0])
def test_benchmark_rejects_invalid_saved_sigma8(value):
    k = np.geomspace(1e-4, 10, 32)
    power = np.ones_like(k)
    run = {
        "params": {"H0": 70},
        "power_result": {"k": k, "P": power, "redshifts": [0.0]},
        "sigma_result": {"sigma8_pipeline_by_z": [value]},
    }
    with pytest.raises(ValueError, match="positive finite sigma8"):
        internal_sigma8_benchmark(run)


@pytest.mark.parametrize("tolerance", [np.nan, np.inf, 0.0, -0.01])
def test_benchmark_rejects_invalid_tolerances_before_evaluation(tolerance):
    with pytest.raises(ValueError, match="tolerance"):
        internal_sigma8_benchmark({}, relative_tolerance=tolerance)
