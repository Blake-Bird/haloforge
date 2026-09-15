import numpy as np

from engine.plot_insights import (
    largest_deviation_point,
    peak_point,
    reference_crossings,
    turning_points,
    validity_boundaries,
)


def test_peak_uses_an_observed_sample_without_interpolation():
    assert peak_point([1, 2, 3], [2, 8, 4]) == {"index": 1, "x": 2.0, "y": 8.0}


def test_validity_boundaries_mark_only_transitions():
    assert validity_boundaries([1, 2, 3, 4], [False, True, True, False]) == [2.0, 4.0]


def test_largest_deviation_ignores_nan_and_keeps_reference_explicit():
    point = largest_deviation_point([1, 2, 3], [1.0, np.nan, 1.8], reference=1.0)
    assert point["x"] == 3.0
    assert point["absolute_deviation"] == 0.8


def test_turning_points_report_only_sampled_nonflat_local_extrema():
    assert turning_points([1, 2, 3, 4, 5], [2, 5, 1, 1, 3]) == [
        {"index": 1, "x": 2.0, "y": 5.0, "kind": "sampled local maximum"},
    ]


def test_reference_crossings_do_not_invent_an_interpolated_coordinate():
    crossings = reference_crossings([1, 2, 4, 8], [0.8, 1.0, 1.4, 0.5], reference=1.0)
    assert crossings[0] == {
        "kind": "observed reference crossing",
        "index": 1,
        "x": 2.0,
        "y": 1.0,
    }
    assert crossings[1]["kind"] == "sampled crossing bracket"
    assert crossings[1]["x_lower"] == 4.0
    assert crossings[1]["x_upper"] == 8.0
