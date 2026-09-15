import sys
import types
import json
import subprocess

import numpy as np
import pytest

from config.defaults import DEFAULT_PARAMS
from engine.class_runner import build_class_settings, compute_matter_power


def test_ede_settings_match_axiclass_parameterization():
    p = dict(DEFAULT_PARAMS)
    settings = build_class_settings(p)
    assert settings["scf_potential"] == "axion"
    assert settings["n_axion"] == 3
    assert settings["log10_axion_ac"] == pytest.approx(-3.5)
    assert settings["fraction_axion_ac"] == pytest.approx(0.10)
    assert settings["scf_parameters"] == "2.806,0.0"
    assert settings["do_shooting_scf"] == "yes"
    assert settings["scf_has_perturbations"] == "yes"


def test_lcdm_does_not_leak_scalar_field_settings():
    p = dict(DEFAULT_PARAMS)
    p["enable_ede"] = False
    settings = build_class_settings(p)
    assert not any(key.startswith("scf_") for key in settings)
    assert "fraction_axion_ac" not in settings


def test_all_requested_redshifts_are_sent_to_class():
    p = dict(DEFAULT_PARAMS)
    p["single_z"] = 3.0
    p["z_values"] = [0.0, 1.0, 10.0]
    settings = build_class_settings(p)
    assert settings["z_pk"] == "0,1,3,10"
    assert settings["z_max_pk"] == 10.0


def test_compute_samples_each_redshift_from_class(monkeypatch):
    class FakeClass:
        def set(self, settings):
            self.settings = settings

        def compute(self):
            pass

        def pk(self, k, z):
            return (1.0 + k) / (1.0 + z) ** 2

        def get_current_derived_parameters(self, _):
            return {"h": 0.6781, "Omega_m": 0.309, "sigma8": 0.81}

        def scale_independent_growth_factor(self, z):
            return 1.0 / (1.0 + z)

        def get_background(self):
            return {
                "z": np.array([0.0, 1.0, 2.0, 3.0]),
                "H [1/Mpc]": np.array([1.0, 2.0, 3.0, 4.0]),
            }

        def struct_cleanup(self):
            pass

        def empty(self):
            pass

    monkeypatch.setitem(sys.modules, "classy", types.SimpleNamespace(Class=FakeClass))
    p = dict(DEFAULT_PARAMS)
    p.update(
        enable_ede=False,
        k_min=1e-3,
        k_max=1.0,
        k_points=5,
        z_values=[0.0, 2.0],
        single_z=0.0,
    )
    result = compute_matter_power(p)
    assert result["class_status"] == "CLASS"
    np.testing.assert_allclose(result["P_by_z"][1], result["P_by_z"][0] / 9.0)
    np.testing.assert_allclose(result["growth_class"], [1.0, 1.0 / 3.0])
    np.testing.assert_allclose(
        result["background_omega_m_by_z"], [0.309, 0.309 * 27 / 9]
    )


def test_missing_class_background_does_not_erase_valid_power_result(monkeypatch):
    class FakeClass:
        def set(self, settings):
            self.settings = settings

        def compute(self):
            pass

        def pk(self, k, z):
            return 1.0 + k

        def get_current_derived_parameters(self, _):
            return {"h": 0.6781, "Omega_m": 0.309, "sigma8": 0.81}

        def scale_independent_growth_factor(self, z):
            return 1.0 / (1.0 + z)

        def struct_cleanup(self):
            pass

        def empty(self):
            pass

    monkeypatch.setitem(sys.modules, "classy", types.SimpleNamespace(Class=FakeClass))
    p = dict(DEFAULT_PARAMS)
    p.update(
        enable_ede=True, k_min=1e-3, k_max=1, k_points=5, z_values=[0.0], single_z=0.0
    )
    result = compute_matter_power(p)
    assert result["class_status"] == "AXICLASS"
    assert result["background_omega_m_by_z"].size == 0
    assert "background table" in result["background_warning"]


def test_worker_retries_only_after_transient_crash(monkeypatch):
    """A killed worker gets one clean retry and records the execution policy."""
    monkeypatch.delitem(sys.modules, "classy", raising=False)
    monkeypatch.setenv("HALOFORGE_CLASS_TRANSIENT_RETRIES", "1")
    calls = []

    def fake_run(command, **_kwargs):
        calls.append(command)
        if len(calls) == 1:
            return subprocess.CompletedProcess(command, -9, stdout="", stderr="")
        result_path = command[-1]
        with open(result_path, "wb") as handle:
            np.savez_compressed(
                handle,
                k=np.array([0.1]),
                P=np.array([1.0]),
                P_by_z=np.array([[1.0]]),
                redshifts=np.array([0.0]),
                growth_class=np.array([1.0]),
                background_omega_m_by_z=np.array([0.3]),
                metadata=json.dumps({"class_status": "CLASS"}),
            )
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr("engine.class_runner.subprocess.run", fake_run)
    result = compute_matter_power(dict(DEFAULT_PARAMS))
    assert len(calls) == 2
    assert result["solver_execution"]["attempts"] == 2
    assert result["solver_execution"]["transient_retry_limit"] == 1


def test_worker_reported_class_error_is_not_retried(monkeypatch):
    monkeypatch.delitem(sys.modules, "classy", raising=False)
    monkeypatch.setenv("HALOFORGE_CLASS_TRANSIENT_RETRIES", "3")
    calls = []

    def fake_run(command, **_kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(
            command, 1, stdout="", stderr="invalid input"
        )

    monkeypatch.setattr("engine.class_runner.subprocess.run", fake_run)
    with pytest.raises(RuntimeError, match="invalid input"):
        compute_matter_power(dict(DEFAULT_PARAMS))
    assert len(calls) == 1
