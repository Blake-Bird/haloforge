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


def test_zero_ede_is_exactly_the_lcdm_settings_limit():
    zero = dict(DEFAULT_PARAMS, enable_ede=True, f_EDE=0.0)
    lcdm = dict(DEFAULT_PARAMS, enable_ede=False)
    assert build_class_settings(zero) == build_class_settings(lcdm)


@pytest.mark.parametrize("bad", [-1, np.inf, np.nan])
def test_solver_rejects_invalid_redshifts_instead_of_silently_dropping_them(bad):
    with pytest.raises(ValueError, match="redshifts"):
        build_class_settings(dict(DEFAULT_PARAMS, z_values=[bad]))


def test_closed_cosmology_can_have_omega_m_above_one():
    from engine.class_runner import _background_omega_m_by_z

    class ClosedBackground:
        def get_background(self):
            z = np.array([0.0, 2.0, 10.0])
            return {
                "z": z,
                "H [1/Mpc]": np.sqrt(0.3 * (1 + z) ** 3 - 0.05 * (1 + z) ** 2 + 0.75),
            }

    actual = _background_omega_m_by_z(
        ClosedBackground(), np.array([0.0, 2.0, 10.0]), 0.3
    )
    assert actual[-1] > 1
    np.testing.assert_allclose(actual[0], 0.3)


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
        params = json.loads(open(command[-2]).read())
        k = np.logspace(
            np.log10(params["k_min"]), np.log10(params["k_max"]), params["k_points"]
        )
        z = np.array(sorted({0.0, params["single_z"], *params["z_values"]}))
        with open(result_path, "wb") as handle:
            np.savez_compressed(
                handle,
                k=k,
                P=np.ones(k.size),
                P_by_z=np.ones((z.size, k.size)),
                redshifts=z,
                growth_class=np.ones(z.size),
                background_omega_m_by_z=np.full(z.size, 0.3),
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


def test_smoke_test_reports_a_solved_sample_and_does_not_inherit_redshift(monkeypatch):
    from engine.class_runner import tiny_class_smoke_test

    def compute(params):
        assert params["single_z"] == 0
        assert params["z_values"] == [0]
        assert not params["enable_ede"]
        k = np.logspace(
            np.log10(params["k_min"]), np.log10(params["k_max"]), params["k_points"]
        )
        assert np.isclose(k[6], 0.1, rtol=1e-12)
        return {"k": k, "P": k**-2}

    monkeypatch.setattr("engine.class_runner.compute_matter_power", compute)
    result = tiny_class_smoke_test(dict(DEFAULT_PARAMS, single_z=100))
    assert result["ok"]
    assert "P(0.1,0)=1.000000e+02" in result["message"]
