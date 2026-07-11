import sys
import types

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
        def set(self, settings): self.settings = settings
        def compute(self): pass
        def pk(self, k, z): return (1.0 + k) / (1.0 + z) ** 2
        def get_current_derived_parameters(self, _): return {"h": .6781, "Omega_m": .309, "sigma8": .81}
        def scale_independent_growth_factor(self, z): return 1.0 / (1.0 + z)
        def struct_cleanup(self): pass
        def empty(self): pass

    monkeypatch.setitem(sys.modules, "classy", types.SimpleNamespace(Class=FakeClass))
    p = dict(DEFAULT_PARAMS)
    p.update(enable_ede=False, k_min=1e-3, k_max=1.0, k_points=5, z_values=[0.0, 2.0], single_z=0.0)
    result = compute_matter_power(p)
    assert result["class_status"] == "CLASS"
    np.testing.assert_allclose(result["P_by_z"][1], result["P_by_z"][0] / 9.0)
    np.testing.assert_allclose(result["growth_class"], [1.0, 1.0 / 3.0])
