import numpy as np
import pytest

from engine.windows import (
    gaussian_W,
    sharp_k_W,
    top_hat_W,
    top_hat_W_series,
    window_squared,
)


def test_top_hat_zero_and_tiny_are_close_to_one():
    y = np.array([0.0, 1e-8, 1e-4])
    values = top_hat_W(y)
    assert np.all(np.isfinite(values))
    assert np.allclose(values, 1.0, atol=1e-8)


def test_top_hat_series_matches_safe_branch_near_zero():
    y = np.array([0.01, 0.05, 0.09])
    assert np.allclose(top_hat_W(y), top_hat_W_series(y))


def test_gaussian_zero_is_one():
    assert np.isclose(gaussian_W(np.array([0.0]))[0], 1.0)


def test_sharp_k_expected_values():
    assert sharp_k_W(np.array([0.0, 0.5, 1.0, 1.1])).tolist() == [1.0, 1.0, 1.0, 0.0]


def test_window_squared_nonnegative():
    y = np.logspace(-4, 2, 100)
    assert np.all(window_squared(y, "Top-hat") >= 0)


def test_windows_are_even_and_unknown_names_are_rejected():
    y = np.array([0, 0.01, 0.1, 1, 3])
    for name in ("Top-hat", "Gaussian", "Sharp-k"):
        np.testing.assert_array_equal(window_squared(y, name), window_squared(-y, name))
    with pytest.raises(ValueError, match="Unknown smoothing window"):
        window_squared(y, "Tophat typo")


def test_top_hat_does_not_evaluate_large_arguments_in_taylor_branch():
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        assert np.isfinite(top_hat_W(1e40))
        assert top_hat_W(1e-100) == 1
