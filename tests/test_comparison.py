import numpy as np
import pytest

from engine.comparison import compare_at_point, sample_curve_at, transform_curve


def test_ratio_never_extrapolates_baseline():
    x = np.array([1.0, 2.0, 4.0, 8.0, 16.0])
    result = transform_curve(x, x**2, [2.0, 8.0], [4.0, 64.0], "Ratio")
    np.testing.assert_allclose(result[1:4], 1)
    assert np.isnan(result[[0, 4]]).all()


def test_zero_denominator_remains_undefined():
    result = transform_curve([0, 1, 2], [1, 1, 1], [0, 1, 2], [1, 0, -1], "Ratio")
    np.testing.assert_allclose(result, [1, np.nan, -1], equal_nan=True)


def test_overlay_keeps_candidate_outside_baseline():
    np.testing.assert_array_equal(
        transform_curve([1, 10], [3, 8], [2, 3], [1, 2], "Overlay"), [3, 8]
    )


def test_identical_curve_percent_difference_is_zero():
    x = np.geomspace(0.01, 100, 70)
    np.testing.assert_allclose(
        transform_curve(x, x**-1.5, x, x**-1.5, "Percent difference"), 0, atol=1e-12
    )


@pytest.mark.parametrize("grid", [[1, 1], [2, 1], [1, np.nan]])
def test_invalid_coordinates_rejected(grid):
    with pytest.raises(ValueError):
        transform_curve(grid, [1, 2], [1, 2], [1, 2], "Ratio")


def test_point_comparison_matches_stored_samples_and_reports_units_free_ratio():
    result = compare_at_point([1, 2, 4], [10, 20, 40], [1, 2, 4], [5, 10, 20], 2)
    assert result["candidate_value"] == 20
    assert result["baseline_value"] == 10
    assert result["ratio"] == 2
    assert result["percent_difference"] == 100
    assert result["candidate_method"] == "observed stored sample"


def test_point_comparison_uses_bounded_log_interpolation_and_rejects_extrapolation():
    sampled = sample_curve_at([1, 10, 100], [1, 100, 10000], 5)
    assert sampled["method"] == "log-log interpolation inside stored domain"
    assert sampled["value"] == pytest.approx(25)
    with pytest.raises(ValueError, match="stored domain"):
        compare_at_point([1, 10], [1, 10], [1, 10], [1, 10], 20)


def test_underflow_endpoint_does_not_change_positive_bracket_interpolation():
    x, y = [1, 10, 100], [1, 0.01, 0]
    assert sample_curve_at(x, y, 5)["value"] == pytest.approx(0.04)
    assert sample_curve_at(x, y, 55)["value"] == pytest.approx(0.005)
    ratios = transform_curve([1, 5, 10, 100], [1, 0.04, 0.01, 0], x, y, "Ratio")
    np.testing.assert_allclose(ratios[:3], 1)
    assert np.isnan(ratios[-1])


def test_point_comparison_rejects_overflow():
    with pytest.raises(ValueError, match="floating-point range"):
        compare_at_point([1, 2], [1e308, 1e308], [1, 2], [1e-300, 1e-300], 1)
