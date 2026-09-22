"""Real AxiCLASS checks; enabled explicitly and required by container CI.

Run with HALOFORGE_TEST_CLASS=1 in an environment containing the pinned binding.
These test the solver-to-analysis interface and a bounded independent CAMB reference.
"""

import os
import json
from pathlib import Path

import numpy as np
import pytest

from config.defaults import DEFAULT_PARAMS
from engine.class_runner import compute_matter_power
from engine.sigma import compute_sigma_result, sigma_grid, build_power_interpolator


pytestmark = pytest.mark.skipif(
    os.environ.get("HALOFORGE_TEST_CLASS") != "1",
    reason="Requires the pinned AxiCLASS binding; enable HALOFORGE_TEST_CLASS=1",
)


@pytest.fixture(scope="module")
def real_lcdm():
    params = dict(
        DEFAULT_PARAMS,
        enable_ede=False,
        k_max=50.0,
        k_points=1200,
        z_values=[0.0, 2.0, 10.0],
        mass_points=40,
    )
    return params, compute_matter_power(params)


def test_real_solver_sigma8_growth_and_all_windows(real_lcdm):
    params, power = real_lcdm
    assert power["background_warning"] == ""
    assert power["solver_execution"]["isolation"] == "dedicated subprocess worker"
    assert power["solver_execution"]["worker_module"] == "engine.class_worker"
    assert power["solver_execution"]["requested_redshifts"] == [0.0, 2.0, 10.0]
    assert "stdout" in power["solver_execution"]
    assert "stderr" in power["solver_execution"]
    assert power["cosmic_time_warning"] == ""
    assert power["background_cosmic_time_gyr_by_z"].shape == power["redshifts"].shape
    assert np.all(np.diff(power["background_cosmic_time_gyr_by_z"]) < 0)
    for window in ("Top-hat", "Gaussian", "Sharp-k"):
        sigma = compute_sigma_result(power, dict(params, window_type=window))
        assert np.isfinite(sigma["sigma_by_z"]).all()
        assert np.all(np.diff(sigma["sigma_by_z"], axis=0) < 0)
        assert sigma["sigma8_pipeline_by_z"][0] == pytest.approx(
            power["derived"]["sigma8"], rel=1e-3
        )


def test_real_zero_ede_matches_lcdm(real_lcdm):
    params, reference = real_lcdm
    power = compute_matter_power(dict(params, enable_ede=True, f_EDE=0.0))
    assert power["class_status"] == "CLASS"
    np.testing.assert_allclose(power["P_by_z"], reference["P_by_z"], rtol=1e-10)


def test_real_ede_changes_spectrum_and_preserves_background(real_lcdm):
    params, reference = real_lcdm
    params = dict(params, enable_ede=True, f_EDE=0.12)
    power = compute_matter_power(params)
    assert power["class_status"] == "AXICLASS"
    assert power["background_warning"] == ""
    assert power["background_omega_m_by_z"].shape == power["redshifts"].shape
    assert power["cosmic_time_warning"] == ""
    assert power["background_cosmic_time_gyr_by_z"].shape == power["redshifts"].shape
    assert not np.allclose(power["P_by_z"], reference["P_by_z"], rtol=0.01)
    sigma = compute_sigma_result(power, params)
    assert sigma["sigma8_pipeline_by_z"][0] == pytest.approx(
        power["derived"]["sigma8"], rel=1e-3
    )


def test_real_closed_background_is_retained(real_lcdm):
    params, _ = real_lcdm
    power = compute_matter_power(dict(params, Omega_k=-0.05))
    assert power["background_warning"] == ""
    assert power["background_omega_m_by_z"][-1] > 1


def test_lcdm_matches_independent_frozen_camb_spectrum_and_variance(real_lcdm):
    reference = json.loads(
        (Path(__file__).parent / "reference/camb_lcdm.json").read_text()
    )
    params, power = real_lcdm
    for key, value in reference["parameters"].items():
        assert params[key] == value
    np.testing.assert_array_equal(power["redshifts"], reference["redshifts"])
    h = params["H0"] / 100
    masses = np.asarray(reference["mass_h"]) / h
    for index, spectrum in enumerate(power["P_by_z"]):
        sampled = build_power_interpolator(power["k"], spectrum)(reference["k"])
        np.testing.assert_allclose(
            sampled,
            reference["power_by_z"][index],
            rtol=reference["tolerances"]["power_relative"],
        )
        variance = sigma_grid(
            masses,
            power["k"],
            spectrum,
            {"h": h, "Omega_m": params["Omega_m"]},
            "Top-hat",
        )
        np.testing.assert_allclose(
            variance["sigma"],
            reference["sigma_by_z"][index],
            rtol=reference["tolerances"]["sigma_relative"],
        )
    result = compute_sigma_result(power, params)
    np.testing.assert_allclose(
        result["sigma8_pipeline_by_z"],
        reference["sigma8_by_z"],
        rtol=reference["tolerances"]["sigma_relative"],
    )


def test_generated_recreation_script_reproduces_the_focused_redshift(
    real_lcdm, tmp_path
):
    import subprocess
    import sys
    from state.run_storage import (
        _power_result_from_run,
        export_power_csv,
        python_recreation_script,
    )

    params, power = real_lcdm
    run = {
        "params": dict(params, single_z=2.0),
        "arrays": power,
        "class_settings": power["class_settings"],
    }
    (tmp_path / "params.json").write_text(json.dumps(run["params"]))
    (tmp_path / "power_spectrum.csv").write_text(
        export_power_csv(_power_result_from_run(run))
    )
    script = tmp_path / "recreate.py"
    script.write_text(python_recreation_script(run))
    completed = subprocess.run(
        [sys.executable, str(script)], capture_output=True, text=True, timeout=60
    )
    assert completed.returncode == 0, completed.stderr
    assert "z=2:" in completed.stdout and "{fractional" not in completed.stdout
    recreated = np.loadtxt(
        tmp_path / "recreated_power_spectrum.csv", delimiter=",", skiprows=1
    )
    np.testing.assert_allclose(recreated[:, 0], power["k"], rtol=1e-13)
    np.testing.assert_allclose(recreated[:, 1], power["P_by_z"][1], rtol=1e-10)
