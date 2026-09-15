from copy import deepcopy

import numpy as np
import pytest

from config.defaults import DEFAULT_PARAMS
from engine.performance import PERFORMANCE_BENCHMARK_VERSION, profile_core_pipeline
from engine.sigma import compute_sigma_result


def _run():
    params = {
        **deepcopy(DEFAULT_PARAMS),
        "enable_ede": False,
        "k_points": 200,
        "mass_points": 60,
    }
    k = np.geomspace(1e-4, 30, 200)
    power = {
        "k": k,
        "P": 1e4 * k / (1 + (k / 0.1) ** 3),
        "derived": {"h": 0.6781, "sigma8": 0.8},
    }
    return {
        "params": params,
        "power_result": power,
        "sigma_result": compute_sigma_result(power, params),
    }


def test_core_performance_probe_has_explicit_budgets_and_scope():
    report = profile_core_pipeline(_run(), repetitions=1)
    assert report["benchmark_version"] == PERFORMANCE_BENCHMARK_VERSION
    assert report["core_rebuild_median_seconds"] >= 0
    assert report["python_peak_allocation_mib"] >= 0
    assert report["core_rebuild_state"] in {"pass", "review"}
    assert report["unmeasured_budgets"]


def test_core_performance_probe_validates_repetition_bounds():
    with pytest.raises(ValueError, match="between 1 and 10"):
        profile_core_pipeline(_run(), repetitions=0)
