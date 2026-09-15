import numpy as np
import pytest

from engine.onboarding import (
    CLUSTER_MASS_HINV_MSUN,
    REVEAL_STAGES,
    committed_prediction,
    halo_abundance_change,
    reveal_stage,
)


def test_guided_cluster_outcome_is_bounded_and_reports_units():
    mass = np.array([1e13, 1e14, 1e15])
    baseline = np.array([1e-4, 1e-6, 1e-9])
    candidate = baseline * np.array([1.0, 0.8, 0.6])

    outcome = halo_abundance_change(mass, candidate, mass, baseline)

    assert outcome["mass_hinv_msun"] == CLUSTER_MASS_HINV_MSUN
    assert outcome["candidate_value"] == pytest.approx(8e-7)
    assert outcome["percent_difference"] == pytest.approx(-20.0)
    assert outcome["unit"] == "h⁻¹ M☉"
    assert "does not establish" in outcome["scope_limit"]


def test_guided_cluster_outcome_refuses_extrapolation():
    with pytest.raises(ValueError, match="stored domain"):
        halo_abundance_change(
            [1e13, 1e14], [1e-4, 1e-6], [1e13, 1e14], [1e-4, 1e-6], 1e15
        )


def test_reveal_stages_are_stable_and_reject_invalid_ui_values():
    assert reveal_stage(0) == "1 · Early condition"
    assert reveal_stage(len(REVEAL_STAGES) - 1) == "4 · Halo abundance"
    with pytest.raises(ValueError, match="integer"):
        reveal_stage(4)
    with pytest.raises(ValueError, match="integer"):
        reveal_stage(True)


def test_guided_prediction_requires_a_deliberate_available_choice():
    choices = ["More", "Fewer", "I am not sure yet"]
    assert committed_prediction("I am not sure yet", choices) == "I am not sure yet"
    with pytest.raises(ValueError, match="Choose a prediction"):
        committed_prediction(None, choices)
    with pytest.raises(ValueError, match="Choose a prediction"):
        committed_prediction("A stale answer", choices)
