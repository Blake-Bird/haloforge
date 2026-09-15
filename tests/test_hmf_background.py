from copy import deepcopy

import numpy as np
import pytest

from config.defaults import DEFAULT_PARAMS
from engine.hmf import hmf_z


def test_ede_watson_uses_supplied_axiclass_background():
    params = {
        **deepcopy(DEFAULT_PARAMS),
        "window_type": "Top-hat",
        "enable_ede": True,
        "mass_definition": "so_mean",
    }
    run = {
        "params": params,
        "power_result": {
            "derived": {"h": 0.7},
            "background_omega_m_by_z": np.array([0.3, 0.73]),
        },
        "sigma_result": {
            "redshifts": np.array([0.0, 1.0]),
            "sigma_by_z": np.array([[2.0, 1.0, 0.5], [1.0, 0.5, 0.25]]),
            "M": np.array([1e10, 1e11, 1e12]),
            "M_h": np.array([0.7e10, 0.7e11, 0.7e12]),
            "rho0": 4e10,
            "window_type": "Top-hat",
        },
    }
    result = hmf_z(run, 1, "Watson SO 2013")
    assert result["omega_m_source"] == "CLASS/AxiCLASS background"
    assert result["omega_m_z"] == pytest.approx(0.73)


def test_ede_watson_fails_closed_without_background():
    params = {
        **deepcopy(DEFAULT_PARAMS),
        "window_type": "Top-hat",
        "enable_ede": True,
        "mass_definition": "so_mean",
    }
    run = {
        "params": params,
        "power_result": {"derived": {"h": 0.7}},
        "sigma_result": {
            "redshifts": np.array([0.0, 1.0]),
            "sigma_by_z": np.array([[2.0, 1.0, 0.5], [1.0, 0.5, 0.25]]),
            "M": np.array([1e10, 1e11, 1e12]),
            "M_h": np.array([0.7e10, 0.7e11, 0.7e12]),
            "rho0": 4e10,
            "window_type": "Top-hat",
        },
    }
    with pytest.raises(ValueError, match="unavailable for EDE"):
        hmf_z(run, 1, "Watson SO 2013")
