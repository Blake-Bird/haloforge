import numpy as np

from engine.sigma import sigma_numerical_diagnostics


def test_diagnostics_report_coverage_and_bounded_truncation_checks():
    mass = np.geomspace(1e8, 1e14, 48)
    k = np.geomspace(1e-4, 1e2, 600)
    power = 1e4 * k / (1 + (k / 0.2) ** 3)
    report = sigma_numerical_diagnostics(
        mass, k, power, {"h": 0.7, "Omega_m": 0.3}, "Top-hat", 64
    )
    assert "does not test power outside" in report["scope_limit"]
    assert report["coverage"]["k_samples_per_decade"] > 90
    for key in ("high_k_truncation", "low_k_truncation"):
        result = report[key]
        assert result["status"] in {"low_sensitivity", "material_sensitivity"}
        assert result["maximum_fractional_sigma_change"] >= 0
        assert mass[0] <= result["mass_at_maximum_Msun"] <= mass[-1]


def test_high_k_removal_changes_small_scale_sigma_more_than_large_scale():
    mass = np.geomspace(1e8, 1e15, 90)
    k = np.geomspace(1e-5, 1e3, 1000)
    power = 1e4 * k / (1 + (k / 0.2) ** 3)
    report = sigma_numerical_diagnostics(
        mass, k, power, {"h": 0.7, "Omega_m": 0.3}, "Top-hat", 80
    )
    assert report["high_k_truncation"]["mass_at_maximum_Msun"] < 1e11
