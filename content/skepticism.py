"""Adversarial exercises that reward warranted claims over pretty curves."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SkepticismExercise:
    identifier: str
    setup: str
    prompt: str
    choices: tuple[str, ...]
    unjustified_choice: int
    explanation: str
    follow_up: str


EXERCISES = (
    SkepticismExercise(
        "smooth-is-not-calibrated",
        "Two runs produce smooth, visibly different halo-mass-function curves. The selected empirical fit reports that part of the displayed mass range is outside its calibration contract.",
        "Which conclusion is unjustified?",
        (
            "Some plotted HMF points are outside the declared fit contract.",
            "The smooth candidate curve proves the empirical halo abundance is accurate across the full displayed mass range.",
            "A controlled comparison can still describe a model-dependent difference in the plotted curves.",
        ),
        1,
        "Smooth interpolation is visual continuity, not simulation calibration. The curve can be calculated while the empirical abundance claim remains unsupported at out-of-contract points.",
        "Open the HMF validity context, then restrict the claim to supported points or choose an appropriate calibration.",
    ),
    SkepticismExercise(
        "kmax-is-not-physics",
        "Matched runs differ only in kmax. Low-mass σ(M) changes materially after the larger sampled k range is used.",
        "Which conclusion is unjustified?",
        (
            "The low-mass prediction is sensitive to sampled Fourier coverage.",
            "The higher-kmax run discovered a new physical universe with different primordial parameters.",
            "A convergence or endpoint-sensitivity test is relevant before making a low-mass abundance claim.",
        ),
        1,
        "Changing kmax changes numerical coverage of the same submitted cosmology. It can reveal an under-resolved result, but it is not itself evidence for changed physics.",
        "Run a controlled coverage sweep and record which mass scales stabilize; do not describe the numerical setting as a cosmological parameter.",
    ),
    SkepticismExercise(
        "linear-field-is-not-catalogue",
        "A matched-phase structure-field visualization shows a thinner-looking high-density tail in an EDE candidate than in a ΛCDM baseline.",
        "Which conclusion is unjustified?",
        (
            "The illustration shows how the specified linear fields differ when phases are held fixed.",
            "The image is a literal N-body halo catalogue and measures observed cluster counts.",
            "A halo-abundance interpretation needs separate fit-validity and cosmology-support evidence.",
        ),
        1,
        "The structure field is a linear Gaussian illustration. Shared phases make a visual comparison useful, but do not turn it into nonlinear simulation output or observational data.",
        "Use it to form a hypothesis, then inspect P(k), σ(M), HMF contracts, and external evidence separately.",
    ),
)


def get_exercise(identifier: str) -> SkepticismExercise:
    for exercise in EXERCISES:
        if exercise.identifier == identifier:
            return exercise
    raise KeyError(f"Unknown skepticism exercise: {identifier}")
