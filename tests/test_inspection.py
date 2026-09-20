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
    assert result["fit_checks_pass_at_mass"] is False


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


def _inspection_run():
    k = np.geomspace(0.01, 10, 100)
    return {
        "params": dict(DEFAULT_PARAMS),
        "power_result": {
            "k": k,
            "P": k**-2,
            "P_by_z": np.array([k**-2, k**-1]),
            "redshifts": [0.0, 2.0],
        },
        "sigma_result": {
            "M_h": np.array([1e10, 1e12]),
            "M": np.array([1.5e10, 1.5e12]),
            "R": np.array([0.5, 2.0]),
            "sigma": np.array([2.0, 0.8]),
            "sigma_by_z": np.array([[2.0, 0.8], [1.0, 0.4]]),
            "redshifts": [0.0, 2.0],
            "window_type": "Gaussian",
        },
    }


def test_inspection_rejects_invalid_and_unsampled_requests():
    import pytest

    run = _inspection_run()
    for mass in (0.0, -1.0, np.nan, np.inf, 1e9, 1e13):
        with pytest.raises(ValueError, match="within"):
            inspect_mass_point(run, mass)
    for count in (True, 1.5, np.nan, np.inf, 0, 101):
        with pytest.raises(ValueError, match="integer"):
            inspect_k_range(run, 0.02, 1.0, top_n=count)
    for bounds in ((0.001, 1.0), (0.02, 20.0)):
        with pytest.raises(ValueError, match="within"):
            inspect_k_range(run, *bounds)


def test_inspection_selects_power_by_its_own_redshift_coordinates():
    import pytest

    run = _inspection_run()
    reference = inspect_mass_point(run, 1e10, 2.0)
    run["power_result"]["redshifts"] = [0.0, 1.0, 2.0]
    run["power_result"]["P_by_z"] = np.insert(
        run["power_result"]["P_by_z"], 1, 100.0, axis=0
    )
    result = inspect_mass_point(run, 1e10, 2.0)
    assert result["k_10_mpc_inv"] == reference["k_10_mpc_inv"]
    assert result["k_90_mpc_inv"] == reference["k_90_mpc_inv"]
    run["power_result"]["redshifts"] = [0.0, 1.0, 3.0]
    with pytest.raises(ValueError):
        inspect_mass_point(run, 1e10, 2.0)
    with pytest.raises(ValueError):
        inspect_k_range(run, 0.02, 1.0, 2.0)


def test_inspection_uses_saved_window_and_rejects_malformed_grids():
    import pytest

    run = _inspection_run()
    reference = inspect_mass_point(run, 1e10)
    run["params"]["window_type"] = "Sharp-k"
    assert inspect_mass_point(run, 1e10) == reference
    for masses in ([1e10, np.nan], [1e12, 1e10], [1e10, 1e10]):
        run["sigma_result"]["M_h"] = masses
        with pytest.raises(ValueError, match="Halo masses"):
            inspect_mass_point(run, 1e10)
        with pytest.raises(ValueError, match="Halo masses"):
            inspect_k_range(run, 0.02, 1.0)


def test_contribution_quantiles_integrate_within_each_sample_interval():
    from engine.inspection import _central_contribution_band

    k = np.exp([0.0, 1.0])
    np.testing.assert_allclose(
        np.log(_central_contribution_band(k, [0.0, 2.0])),
        np.sqrt([0.1, 0.9]),
        rtol=1e-14,
    )
    np.testing.assert_allclose(
        np.log(_central_contribution_band(k, [2.0, 0.0])),
        1 - np.sqrt([0.9, 0.1]),
        rtol=1e-14,
    )
    np.testing.assert_allclose(
        np.log(_central_contribution_band(k, [1.0, 1.0])), [0.1, 0.9], rtol=1e-14
    )
    np.testing.assert_allclose(
        _central_contribution_band(k, [1e308, 1e308]),
        _central_contribution_band(k, [1.0, 1.0]),
    )


def test_contribution_diagnostics_reject_invalid_or_unresolved_density():
    import pytest
    from engine.inspection import _central_contribution_band

    for values in ([0.0, 0.0], [-1.0, 2.0], [np.inf, 1.0], [np.nan, 1.0], [1.0]):
        with pytest.raises(ValueError, match="contribution"):
            _central_contribution_band([1.0, 2.0], values)
