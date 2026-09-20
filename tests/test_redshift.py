import numpy as np
import pytest

from engine.redshift import redshift_index


def test_exact_saved_redshift_selected_with_roundoff_tolerance():
    assert redshift_index([0.0, 1.0, 3.0], 1.0) == 1
    assert redshift_index([0.0, 1.0, 3.0], 1.0 + 1e-12) == 1


@pytest.mark.parametrize("requested", [-1, 0.5, 2.0, np.nan, np.inf])
def test_unsaved_redshift_never_selects_a_nearest_neighbour(requested):
    with pytest.raises(ValueError, match="redshift"):
        redshift_index([0.0, 1.0, 3.0], requested)


@pytest.mark.parametrize(
    "grid", [[], [0.0, 0.0], [1.0, 0.0], [0.0, np.nan], [[0.0]], [0.0, 1e-10]]
)
def test_malformed_or_ambiguous_redshift_grid_is_rejected(grid):
    with pytest.raises(ValueError):
        redshift_index(grid, 0.0)
