from copy import deepcopy
from zipfile import ZipFile

import numpy as np
import pytest

from config.defaults import DEFAULT_PARAMS
from engine.hmf import hmf_z, hmf_from_sigma
from engine.sigma import compute_sigma_result
from state.run_model import create_run_from_current_state
from state import run_storage


def pipeline(window):
    params = {**deepcopy(DEFAULT_PARAMS), "window_type": window, "enable_ede": False}
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


@pytest.mark.parametrize("window", ["Gaussian", "Sharp-k"])
def test_alternate_variance_cannot_enter_hmf(window):
    run = pipeline(window)
    assert np.isfinite(run["sigma_result"]["sigma"]).all()
    with pytest.raises(ValueError, match="variance exploration only"):
        hmf_z(run, 0, "Press-Schechter 1974")
    with pytest.raises(ValueError, match="variance exploration only"):
        hmf_from_sigma(
            [1e10, 1e11, 1e12],
            [2, 1, 0.5],
            4e10,
            0.7,
            "Press-Schechter",
            1.686,
            window_type=window,
        )


def test_actual_sigma_window_overrides_stale_parameter_label():
    run = pipeline("Gaussian")
    run["params"]["window_type"] = "Top-hat"
    with pytest.raises(ValueError, match="Gaussian"):
        hmf_z(run, 0, "Press-Schechter")


@pytest.mark.parametrize("window", ["Top-hat", "Gaussian", "Sharp-k"])
def test_save_and_export_preserve_variance_without_inventing_halos(
    window, tmp_path, monkeypatch
):
    for key, path in {
        "DATA_ROOT": tmp_path,
        "RUN_DIR": tmp_path / "saved_runs",
        "EXPORT_DIR": tmp_path / "exports",
        "STATE_DIR": tmp_path / "state",
    }.items():
        monkeypatch.setattr(run_storage, key, path)
    run = create_run_from_current_state(pipeline(window), "Window test")
    files = run_storage.generate_run_exports(run)
    restored = run_storage.load_run(run["run_id"])
    np.testing.assert_array_equal(restored["arrays"]["sigma"], run["arrays"]["sigma"])
    assert restored["integration_method"] == run["integration_method"]
    assert ("hmf_press_schechter_z0" in restored["arrays"]) == (window == "Top-hat")
    with ZipFile(files["exports.zip"]) as archive:
        assert "sigma.csv" in archive.namelist()
        assert run["integration_method"] in archive.read("run_summary.md").decode()
        assert ("hmf.csv" in archive.namelist()) == (window == "Top-hat")
        if window != "Top-hat":
            assert "Unavailable" in archive.read("run_summary.md").decode()


def test_empirical_run_keeps_explicit_analytic_reference_curves():
    current = pipeline("Top-hat")
    current["params"].update(
        fitting="Tinker 2008", mass_definition="so_mean", delta_halo=200
    )
    run = create_run_from_current_state(current, "Tinker test")
    assert run["hmf_status"] == "top_hat_model"
    assert "hmf_press_schechter_z0" in run["arrays"]
    assert "hmf_sheth_tormen_z0" in run["arrays"]
