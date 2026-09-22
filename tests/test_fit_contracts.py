import numpy as np
import pytest

from engine.contracts import (
    MASS_DEFINITIONS,
    fit_contract,
    validity_report,
    validate_fit_configuration,
)
from engine.hmf import hmf_from_sigma


def test_sheth_tormen_does_not_claim_verified_simulation_calibration():
    assert fit_contract("Sheth-Tormen 2001").family == "semi-empirical"
    report = validity_report(
        "Sheth-Tormen 2001", [0.5, 1.0], 0, "analytic_top_hat", 200
    )
    assert report["status"] == "unverified_calibration"
    assert not report["calibrated_mask"].any()


def test_each_fit_has_a_declared_supported_mass_definition():
    from engine.fitting_functions import FITTING_NAMES

    for fit in FITTING_NAMES:
        report = validity_report(
            fit,
            [1.0],
            0,
            {"Tinker 2008": "so_mean", "Watson SO 2013": "so_mean"}.get(
                fit,
                "analytic_top_hat"
                if fit in {"Press-Schechter 1974", "Sheth-Tormen 2001"}
                else "fof_b0.2",
            ),
            200,
        )
        assert report["mass_definition"] in MASS_DEFINITIONS


@pytest.mark.parametrize(
    "fit, chosen, expected",
    [
        ("Tinker 2008", "fof_b0.2", "requires Spherical overdensity"),
        ("Watson FOF 2013", "so_mean", "requires Friends-of-friends"),
        ("Sheth-Tormen", "fof_b0.2", "requires Analytic top-hat"),
    ],
)
def test_wrong_mass_definition_is_rejected(fit, chosen, expected):
    with pytest.raises(ValueError, match=expected):
        validate_fit_configuration(fit, chosen, 200)


def test_out_of_calibration_is_visible_per_mass_sample():
    result = validity_report("Tinker 2008", np.array([0.5, 1, 2]), 0, "so_mean", 200)
    assert result["status"] == "calibrated"
    assert result["calibrated_mask"].tolist() == [True, True, True]


def test_tinker_domain_uses_published_log10_and_redshift_boundary():
    assert (
        validity_report("Tinker 2008", [1.8], 0, "so_mean", 200)["status"]
        == "calibrated"
    )
    high_z = validity_report("Tinker 2008", [1.8], 1, "so_mean", 200)
    assert high_z["status"] == "outside_calibration"
    assert "log10" in high_z["reasons"][0]


def test_watson_so_domain_marks_outside_published_ln_sigma_range():
    result = validity_report("Watson SO 2013", [0.2, 1.0], 0, "so_mean", 178)
    assert result["calibrated_mask"].tolist() == [False, True]


def test_hmf_rejects_wrong_fit_definition_before_evaluating():
    with pytest.raises(ValueError, match="requires Spherical overdensity"):
        hmf_from_sigma(
            np.geomspace(1e10, 1e12, 3),
            [2, 1, 0.5],
            4e10,
            0.7,
            "Tinker 2008",
            1.686,
            mass_definition="fof_b0.2",
        )
