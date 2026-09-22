import pytest

from engine.scientific_status import (
    APPROXIMATION,
    CALCULATED_LINEAR_THEORY,
    DEMONSTRATION,
    VALIDATED,
    weakest_status,
)


def test_weakest_status_never_upgrades_an_artifact():
    assert weakest_status(CALCULATED_LINEAR_THEORY, VALIDATED) == CALCULATED_LINEAR_THEORY
    assert weakest_status(VALIDATED, APPROXIMATION) == APPROXIMATION
    assert weakest_status(DEMONSTRATION, VALIDATED) == DEMONSTRATION


def test_unknown_status_is_rejected():
    with pytest.raises(ValueError, match="Unknown"):
        weakest_status("marketing-claim")
