"""Regression against independent published coefficient nodes and spline form."""

import numpy as np
import pytest
from scipy.interpolate import InterpolatedUnivariateSpline

from engine.fitting_functions import f_tinker08


@pytest.mark.parametrize(
    "delta,A,a,b,c",
    [
        (200, 0.1858659, 1.466904, 2.571104, 1.193958),
        (800, 0.2480968, 1.869936, 1.588649, 1.581345),
        (3200, 0.2600000, 2.661983, 1.405210, 2.439729),
    ],
)
@pytest.mark.parametrize("z", [0, 1, 2.5])
def test_tinker_matches_independent_coefficient_nodes(delta, A, a, b, c, z):
    sigma = np.array([0.7, 1, 1.5, 2])
    alpha = 10 ** (-((0.75 / np.log10(delta / 75)) ** 1.2))
    expected = (
        A
        * (1 + z) ** -0.14
        * ((sigma / (b * (1 + z) ** -alpha)) ** (-a * (1 + z) ** -0.06) + 1)
        * np.exp(-c / sigma**2)
    )
    np.testing.assert_allclose(
        f_tinker08(sigma, z=z, delta_halo=delta), expected, rtol=2e-14
    )


def test_overdensity_changes_abundance_at_zero_redshift():
    assert f_tinker08(1, delta_halo=800) != pytest.approx(f_tinker08(1, delta_halo=200))


@pytest.mark.parametrize("delta", [199.9, 3200.1, np.nan, np.inf])
def test_overdensity_extrapolation_rejected(delta):
    with pytest.raises(ValueError):
        f_tinker08(1, delta_halo=delta)


def test_intermediate_overdensity_matches_independent_spline_implementation():
    from engine.fitting_functions import TINKER_OVERDENSITIES, TINKER_COEFFICIENTS

    delta = 500
    A, a, b, c = [
        InterpolatedUnivariateSpline(TINKER_OVERDENSITIES, column)(delta)
        for column in TINKER_COEFFICIENTS.T
    ]
    np.testing.assert_allclose(
        f_tinker08(1, delta_halo=delta), A * (b**a + 1) * np.exp(-c), rtol=2e-14
    )
