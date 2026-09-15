import numpy as np

from config.defaults import DEFAULT_PARAMS
from engine.inspection import inspect_k_range, inspect_mass_point


def test_inspection_connects_mass_to_radius_variance_and_k_band():
    k = np.geomspace(1e-4, 10, 100)
    run = {
        "params": dict(DEFAULT_PARAMS),
        "power_result": {
            "k": k,
            "P": 1e3 * k,
            "P_by_z": np.array([1e3 * k]),
            "redshifts": np.array([0.0]),
        },
        "sigma_result": {
            "M_h": np.array([1e10, 1e12]),
            "M": np.array([1.5e10, 1.5e12]),
            "R": np.array([0.5, 2.0]),
            "sigma": np.array([2.0, 0.8]),
            "sigma_by_z": np.array([[2.0, 0.8]]),
            "redshifts": np.array([0.0]),
        },
    }
    result = inspect_mass_point(
        run, 8e11, validity={"calibrated_mask": np.array([True, False])}
    )
    assert result["mass_hinv_msun"] == 1e12
    assert result["radius_mpc"] == 2.0
    assert result["nu"] > 2
    assert 0 < result["k_10_mpc_inv"] < result["k_90_mpc_inv"]
    assert result["calibrated_at_mass"] is False


def test_k_range_inspection_ranks_mass_contributions_on_sampled_grid():
    k = np.geomspace(1e-4, 10, 100)
    run = {
        "params": dict(DEFAULT_PARAMS),
        "power_result": {
            "k": k,
            "P": 1e3 * k,
            "P_by_z": np.array([1e3 * k]),
            "redshifts": np.array([0.0]),
        },
        "sigma_result": {
            "M_h": np.array([1e10, 1e12]),
            "R": np.array([0.5, 2.0]),
            "redshifts": np.array([0.0]),
        },
    }
    report = inspect_k_range(run, 0.02, 1.0, 0.0, top_n=2)
    assert len(report["rows"]) == 2
    assert report["actual_sampled_k_start_mpc_inv"] >= 0.02
    fractions = [row["sigma2_fraction_from_selected_k"] for row in report["rows"]]
    assert fractions == sorted(fractions, reverse=True)
    assert all(0 <= value <= 1 for value in fractions)
