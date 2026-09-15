from engine.chart_selection import selected_x_bounds


def test_brushed_plotly_points_produce_finite_increasing_x_bounds():
    event = {
        "selection": {
            "points": [{"x": "2"}, {"x": 8.0}, {"x": float("nan")}, {"not_x": 4}]
        }
    }
    assert selected_x_bounds(event) == (2.0, 8.0)


def test_one_point_degenerate_or_non_numeric_selection_does_not_create_a_region():
    assert selected_x_bounds({"selection": {"points": [{"x": 2.0}]}}) is None
    assert (
        selected_x_bounds({"selection": {"points": [{"x": 2.0}, {"x": 2.0}]}}) is None
    )
    assert (
        selected_x_bounds({"selection": {"points": [{"x": "not a number"}, {}]}})
        is None
    )
    assert selected_x_bounds(None) is None
