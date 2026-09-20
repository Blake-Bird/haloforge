from copy import deepcopy

import numpy as np
import pytest

from state.figure_export import FigureExportError, build_publication_figures


def saved_run():
    return {
        "params": {"single_z": 2.0, "window_type": "Top-hat"},
        "arrays": {
            "k": np.array([0.01, 0.1, 1.0]),
            "P": np.array([100.0, 20.0, 1.0]),
            "P_by_z": np.array([[100.0, 20.0, 1.0], [10.0, 2.0, 0.1]]),
            "redshifts": np.array([0.0, 2.0]),
            "M_h": np.array([1e10, 1e12, 1e14]),
            "sigma": np.array([3.0, 2.0, 1.0]),
            "sigma_by_z": np.array([[3.0, 2.0, 1.0], [1.5, 1.0, 0.5]]),
            "hmf_press_schechter_z0": np.array([0.1, 0.01, 0.0]),
            "hmf_sheth_tormen_z0": np.array([0.12, 0.02, 0.0]),
        },
    }


def test_export_uses_focused_redshift_and_labels_reference_curves_separately():
    run = saved_run()
    figures = dict(build_publication_figures(run))
    np.testing.assert_array_equal(
        figures["matter_power"].data[0].y, run["arrays"]["P_by_z"][1]
    )
    np.testing.assert_array_equal(
        figures["mass_variance"].data[0].y, run["arrays"]["sigma_by_z"][1]
    )
    assert "z = 2" in figures["matter_power"].layout.title.text
    assert "z = 2" in figures["mass_variance"].layout.title.text
    assert "z = 0" in figures["analytic_hmf_reference"].layout.annotations[0].text
    assert np.isnan(figures["analytic_hmf_reference"].data[0].y[-1])
    assert run["arrays"]["hmf_press_schechter_z0"][-1] == 0


def test_grayscale_variants_preserve_every_sample_and_axis_scale():
    figures = dict(build_publication_figures(saved_run()))
    for name in ("matter_power", "mass_variance", "analytic_hmf_reference"):
        source, gray = figures[name], figures[name + "_grayscale"]
        assert source.layout.xaxis.type == gray.layout.xaxis.type == "log"
        assert source.layout.yaxis.type == gray.layout.yaxis.type == "log"
        for first, second in zip(source.data, gray.data):
            np.testing.assert_array_equal(first.x, second.x)
            np.testing.assert_array_equal(first.y, second.y)
        assert "<br>" in source.layout.annotations[0].text


@pytest.mark.parametrize(
    "key,value",
    [
        ("k", [1.0, 0.1, 0.01]),
        ("P", [100.0, np.nan, 1.0]),
        ("P_by_z", [[1.0, 2.0, 3.0]]),
        ("sigma_by_z", [[1.0, 2.0, 3.0]]),
        ("M_h", [1e10, 1e10, 1e14]),
        ("hmf_press_schechter_z0", [0.1, -0.1, 0.0]),
    ],
)
def test_invalid_export_arrays_are_rejected_instead_of_silently_omitted(key, value):
    run = saved_run()
    run["arrays"][key] = np.asarray(value)
    with pytest.raises(FigureExportError):
        build_publication_figures(run)


def test_missing_redshift_or_failed_integrity_cannot_be_exported():
    run = saved_run()
    for changes in (
        {"params": {"single_z": 1.0}},
        {"integrity_status": {"state": "invalid"}},
    ):
        with pytest.raises(FigureExportError):
            build_publication_figures(dict(run, **changes))
    legacy = deepcopy(run)
    legacy["arrays"].pop("P_by_z")
    with pytest.raises(FigureExportError, match="not stored"):
        build_publication_figures(legacy)


def test_table_exports_use_the_same_focused_samples_as_figures():
    from state.run_storage import (
        _power_result_from_run,
        _sigma_result_from_run,
        _power_table_metadata,
    )

    run = saved_run()
    run["arrays"]["dlnsigma_dlnM_by_z"] = np.array(
        [[-0.1, -0.2, -0.3], [-0.2, -0.3, -0.4]]
    )
    power, variance = _power_result_from_run(run), _sigma_result_from_run(run)
    np.testing.assert_array_equal(power["P"], run["arrays"]["P_by_z"][1])
    np.testing.assert_array_equal(variance["sigma"], run["arrays"]["sigma_by_z"][1])
    np.testing.assert_array_equal(
        variance["dlnsigma_dlnM"], run["arrays"]["dlnsigma_dlnM_by_z"][1]
    )
    assert _power_table_metadata(run)["haloforge.redshift"] == "2.0"


def test_table_export_rejects_truncated_arrays_and_missing_derivatives():
    from state.run_storage import export_power_csv, export_sigma_csv

    with pytest.raises(ValueError):
        export_power_csv({"k": [1.0, 2.0], "P": [3.0]})
    with pytest.raises(ValueError, match="derivative"):
        export_sigma_csv(
            {"M_h": [1.0, 2.0], "M": [1.0, 2.0], "R": [1.0, 2.0], "sigma": [2.0, 1.0]}
        )
