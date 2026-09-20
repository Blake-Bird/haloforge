"""Bounded evidence summaries for HaloForge's guided first experiment."""

from __future__ import annotations

from copy import deepcopy

import numpy as np

from engine.comparison import compare_at_point


CLUSTER_MASS_HINV_MSUN = 1.0e14
REVEAL_STAGES = (
    "1 · Early condition",
    "2 · Matter power",
    "3 · Smoothed variance",
    "4 · Halo abundance",
)


def guided_parameter_pair(defaults: dict, changes: dict) -> tuple[dict, dict]:
    """Construct an explicit ΛCDM reference and its controlled candidate."""
    baseline = deepcopy(defaults)
    baseline["enable_ede"] = False
    candidate = deepcopy(baseline)
    candidate.update(deepcopy(changes))
    return baseline, candidate


def committed_prediction(
    prediction: object, choices: tuple[str, ...] | list[str]
) -> str:
    """Validate a deliberate guided-experiment prediction.

    Selecting "I am not sure yet" is a valid scientific starting point; an
    unselected or stale widget value is not silently converted to a claim.
    """
    allowed = tuple(str(choice) for choice in choices)
    if not isinstance(prediction, str) or prediction not in allowed:
        raise ValueError("Choose a prediction before staging this guided experiment.")
    return prediction


def reveal_stage(index: int) -> str:
    """Return a stable guided-reveal label and reject an invalid UI state."""
    if (
        isinstance(index, bool)
        or not isinstance(index, int)
        or not 0 <= index < len(REVEAL_STAGES)
    ):
        raise ValueError(
            f"Reveal stage must be an integer from 0 to {len(REVEAL_STAGES) - 1}"
        )
    return REVEAL_STAGES[index]


def halo_abundance_change(
    candidate_mass_hinv_msun,
    candidate_hmf,
    baseline_mass_hinv_msun,
    baseline_hmf,
    mass_hinv_msun: float = CLUSTER_MASS_HINV_MSUN,
) -> dict:
    """Compare matched halo-abundance curves at a stored physical mass.

    The percentage is intentionally a bounded interpolation between sampled
    curves.  It is not a forecast uncertainty or an assertion that an
    empirical HMF fit is calibrated for the candidate cosmology.
    """
    mass = float(mass_hinv_msun)
    if not np.isfinite(mass) or mass <= 0:
        raise ValueError("The comparison mass must be finite and positive")
    comparison = compare_at_point(
        candidate_mass_hinv_msun,
        candidate_hmf,
        baseline_mass_hinv_msun,
        baseline_hmf,
        mass,
    )
    return {
        **comparison,
        "mass_hinv_msun": mass,
        "unit": "h⁻¹ M☉",
        "observable": "dn/dlnM [h³ Mpc⁻³]",
        "scope_limit": (
            "This is a bounded comparison of the two stored analytic HMF curves. "
            "It does not establish HMF calibration, cosmology support, theory uncertainty, or publication suitability."
        ),
    }
