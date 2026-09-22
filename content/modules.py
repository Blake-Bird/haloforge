"""Prepared, falsifiable cosmology lab modules with local-only materials."""

from __future__ import annotations

import json
from hashlib import sha256
from io import BytesIO
from dataclasses import asdict, dataclass
from zipfile import ZIP_DEFLATED, ZipFile


@dataclass(frozen=True)
class LabModule:
    identifier: str
    title: str
    duration_minutes: int
    level: str
    prerequisites: tuple[str, ...]
    objectives: tuple[str, ...]
    question: str
    checkpoint: str
    student_prompt: str
    instructor_note: str
    solution_outline: str
    caution: str


@dataclass(frozen=True)
class LectureSlide:
    """A projector-ready narrative step for one prepared module.

    Slides deliberately carry scientific boundaries alongside the takeaway so a
    classroom discussion cannot silently turn a smooth visual into a claim.
    """

    title: str
    kicker: str
    body: str
    prompt: str
    speaker_note: str


MODULES = (
    LabModule(
        "primordial-tilt",
        "Why a small primordial tilt change matters",
        35,
        "Introductory cosmology",
        ("Logarithms and graphs",),
        (
            "Distinguish amplitude from spectral tilt.",
            "Trace a change from primordial power to σ(M).",
            "State which conclusion needs an HMF calibration.",
        ),
        "At fixed Aₛ, does raising nₛ affect every halo mass equally?",
        "Predict which halo scale is more sensitive before you compare the runs.",
        "Create a Planck-like baseline. Change only nₛ, record a prediction, compare P(k), σ(M), and the HMF, then identify one statement the curves do not justify.",
        "Ask students to locate the pivot before they describe a tilt. A vertically shifted plot is not a tilt change.",
        "Higher nₛ raises relative power above the pivot, so lower-mass scales often show a stronger σ response. The exact HMF response remains fit- and domain-dependent.",
        "A smooth HMF curve is not evidence that the empirical fit is calibrated for every cosmology.",
    ),
    LabModule(
        "numerical-coverage",
        "When does kmax change the answer?",
        40,
        "Computational physics",
        ("Fourier modes", "Numerical integration"),
        (
            "Relate smoothing radius to contributing Fourier modes.",
            "Use a convergence diagnostic as evidence, not decoration.",
            "Separate numerical coverage from physical effects.",
        ),
        "Can an apparently stable large-scale result conceal an unconverged low-mass prediction?",
        "Before calculating, identify whether increasing kmax changes physics or numerical coverage.",
        "Run matched cosmologies with two kmax values. Use the selected-mass inspector and Benchmark Lab. Report a claim that survives the change and a claim that does not.",
        "Require students to label kmax as a numerical decision. Do not reward a curve merely for looking smooth.",
        "Higher kmax can add sampled small-scale modes to σ(M), especially at low mass. This tests numerical coverage, not a different universe.",
        "Endpoint sensitivity cannot establish convergence beyond the original solver grid or validate a halo-fit extrapolation.",
    ),
    LabModule(
        "ede-structure",
        "Early expansion and rare structures",
        50,
        "Structure formation",
        ("Expansion history", "Linear matter power"),
        (
            "Form a falsifiable EDE hypothesis.",
            "Follow a causal chain across scales.",
            "Identify model-dependent claims about massive halos.",
        ),
        "Could a temporary change to early expansion alter a late-time cluster prediction?",
        "Commit to a prediction before enabling EDE, then name one possible alternative explanation for any difference.",
        "Use Explore to stage EDE, compare against a matched ΛCDM baseline, inspect the causal explanation, and write a conclusion plus one caveat in the Notebook.",
        "Keep the claim conditional: AxiCLASS evolves the specified linear model, while HMF calibration support is a separate question.",
        "EDE can change background and perturbation evolution, propagating into P(k) and σ(M). Whether an empirical halo-count inference is supported depends on the selected fit and calibration domain.",
        "The structure-field view is a linear Gaussian illustration, not a literal N-body universe or halo catalogue.",
    ),
    LabModule(
        "rare-tail-statistics",
        "Why small variance changes matter in the rare tail",
        45,
        "Statistics",
        ("Probability distributions", "Logarithms and graphs"),
        (
            "Connect σ(M) to the rarity of high-mass fluctuations.",
            "Distinguish a fractional change from an uncertainty estimate.",
            "State why an HMF tail is model-dependent.",
        ),
        "Why can a modest change in σ(M) correspond to a much larger change in a rare-cluster abundance?",
        "Predict whether the high-mass or low-mass end should amplify a fixed fractional σ change more strongly.",
        "Use a matched baseline and candidate. Compare σ(M), then use Compare at a physical point near 10¹⁴ h⁻¹ M☉. Record the percent difference, interpolation method, and one reason that percent is not an error bar.",
        "Contrast a measured curve ratio with a posterior or confidence interval. The goal is to teach sensitivity, not to convert an HMF curve into a probability statement about the universe.",
        "Rare high-mass objects correspond to the tail of the smoothed fluctuation distribution, so a small variance change can be amplified in a multiplicity relation. The amount of amplification depends on the collapse/HMF model and its supported domain.",
        "A displayed abundance ratio is neither observational evidence nor a calibrated uncertainty interval; it inherits the chosen mass function, mass definition, sampled k range, and cosmology-support limits.",
    ),
    LabModule(
        "numerical-methods",
        "When a stable-looking integral is not enough",
        50,
        "Numerical methods",
        ("Numerical integration", "Sampling and resolution"),
        (
            "Read endpoint-removal diagnostics without overclaiming convergence.",
            "Separate a numerical range choice from a changed physical model.",
            "Design a one-variable convergence follow-up.",
        ),
        "Can two smooth σ(M) curves still leave a low-mass conclusion numerically unsupported?",
        "Before running, predict whether removing high-k samples should matter more at low or high halo mass.",
        "Run the numerical-coverage Explore experiment, inspect the k-range-to-mass mapper and Convergence Lab, then propose one follow-up that changes only a numerical setting. State what the existing endpoint check does and does not establish.",
        "Make students name the numerical decision and the physical quantity separately. A smooth line is evidence of rendering, not automatically of integral convergence or fit validity.",
        "Smaller smoothing radii receive important contributions from higher-k modes, so low-mass σ(M) can be more exposed to finite sampled coverage. Endpoint-removal changes are useful sensitivity evidence, but not a proof of solver, physical, or calibration convergence.",
        "Changing kmax changes sampled numerical coverage; it does not automatically simulate a different universe, validate an empirical HMF fit, or establish behavior outside the original solver grid.",
    ),
)

# These names are the learner-facing experiments in Explore.  Keeping the
# mapping alongside the teaching materials makes the handoff deliberate and
# testable instead of asking a student to infer which playground to open.
GUIDED_EXPERIMENT_BY_MODULE = {
    "primordial-tilt": "More small-scale power",
    "numerical-coverage": "Why kmax matters",
    "ede-structure": "Early expansion",
    "rare-tail-statistics": "Early expansion",
    "numerical-methods": "Why kmax matters",
}


def get_module(identifier: str) -> LabModule:
    for module in MODULES:
        if module.identifier == identifier:
            return module
    raise KeyError(f"Unknown HaloForge lab module: {identifier}")


def guided_experiment_for_module(module: LabModule) -> str:
    """Return the controlled Explore experiment that begins a live module."""
    try:
        return GUIDED_EXPERIMENT_BY_MODULE[module.identifier]
    except KeyError as exc:  # Keeps new modules from silently losing a live path.
        raise ValueError(f"No guided experiment is configured for {module.identifier}") from exc


def student_handout(module: LabModule) -> str:
    return (
        "\n".join(
            [
                f"# {module.title}",
                "",
                f"**Estimated time:** {module.duration_minutes} minutes  ",
                f"**Level:** {module.level}",
                "",
                "## Question",
                module.question,
                "",
                "## Learning objectives",
                *[f"- {item}" for item in module.objectives],
                "",
                "## Prerequisites",
                *[f"- {item}" for item in module.prerequisites],
                "",
                "## Checkpoint",
                module.checkpoint,
                "",
                "## Investigation",
                module.student_prompt,
                "",
                "## Required conclusion",
                "State what changed, what was held fixed, the strongest conclusion supported by the run, and one caveat.",
                "",
                "## Scientific caution",
                module.caution,
            ]
        )
        + "\n"
    )


def instructor_guide(module: LabModule) -> str:
    return (
        "\n".join(
            [
                f"# Instructor guide — {module.title}",
                "",
                f"**Estimated time:** {module.duration_minutes} minutes",
                "",
                "## Teaching move",
                module.instructor_note,
                "",
                "## Solution outline",
                module.solution_outline,
                "",
                "## Misconception to surface",
                module.caution,
                "",
                "## Accessibility",
                "Provide the accessible chart transcript and reduced-motion mode by default; let students submit a written prediction rather than requiring hover interaction.",
                "",
                "## Privacy",
                "This local module does not create a classroom, collect student data, upload responses, or export grades.",
            ]
        )
        + "\n"
    )


def assignment_brief(module: LabModule) -> str:
    """A ready-to-distribute assignment that preserves the module's caveat."""
    return (
        "\n".join(
            [
                f"# Assignment - {module.title}",
                "",
                f"**Estimated time:** {module.duration_minutes} minutes",
                "",
                "## Your question",
                module.question,
                "",
                "## Before you run",
                module.checkpoint,
                "",
                "## What to submit",
                "",
                "1. Your prediction before calculation, including what is held fixed.",
                "2. A comparison of the requested observables with units and a stated baseline.",
                "3. The strongest conclusion supported by the evidence.",
                "4. One caveat and the next evidence step needed to strengthen the claim.",
                "",
                "## Investigation",
                module.student_prompt,
                "",
                "## Scientific boundary",
                module.caution,
            ]
        )
        + "\n"
    )


def solution_guide(module: LabModule) -> str:
    """Instructor-facing solution outline, explicitly not a grading system."""
    return (
        "\n".join(
            [
                f"# Solution guide - {module.title}",
                "",
                "## Expected reasoning",
                module.solution_outline,
                "",
                "## Misconception to surface",
                module.caution,
                "",
                "## Assessment boundary",
                "This local guide contains no student data, answer collection, automatic grading, or hidden remote solution. Use it as a discussion and feedback aid, not evidence of learning outcomes.",
            ]
        )
        + "\n"
    )


def lecture_slides(module: LabModule) -> tuple[LectureSlide, ...]:
    """Return a short, evidence-first lecture arc for a prepared module."""
    return (
        LectureSlide(
            "Start with a question",
            "01 · WONDER",
            module.question,
            "Ask for a quick instinct before naming any variables.",
            "Keep the opening qualitative. The purpose is to make a prediction feel useful, not to test prior vocabulary.",
        ),
        LectureSlide(
            "Commit to a prediction",
            "02 · PREDICT",
            module.checkpoint,
            "Have everyone choose a direction and name what they are holding fixed.",
            "Invite a written or spoken prediction; neither a hover interaction nor a color distinction is required.",
        ),
        LectureSlide(
            "Follow the causal chain",
            "03 · EXPLAIN",
            module.student_prompt,
            "Reveal one link at a time: inputs → power → smoothing → variance → halo inference.",
            "Use HaloForge's causal explanation and selected-mass inspector. Do not skip from a changed control directly to a halo-count conclusion.",
        ),
        LectureSlide(
            "Read the evidence narrowly",
            "04 · EVIDENCE",
            module.solution_outline,
            "Ask which plotted change is direct evidence, and which statement is an inference.",
            "Use a matched baseline and point comparison. Name the units, redshift, and fit before discussing a ratio.",
        ),
        LectureSlide(
            "Try to break the claim",
            "05 · SKEPTIC",
            module.caution,
            "What extra evidence would make the conclusion stronger—or show it is not supported?",
            "A smooth curve is not a validation result. Surface numerical coverage, mass-definition, and calibration-domain limits before the takeaway.",
        ),
        LectureSlide(
            "Launch the investigation",
            "06 · LAB",
            "Students now run the prepared local investigation and record a conclusion plus caveat.",
            "Route to the student handout, the reproducible notebook, or a live comparison.",
            "The local bundle contains no accounts, answer collection, surveillance, or automatic grades.",
        ),
    )


def lecture_outline(module: LabModule) -> str:
    """Return a portable, projector-friendly lecture script in Markdown."""
    lines = [
        f"# Lecture outline — {module.title}",
        "",
        f"**Estimated lab time after discussion:** {module.duration_minutes} minutes",
        "",
        "This local lecture outline is keyboard-operable in HaloForge and can also be printed or presented as static text.",
        "",
    ]
    for slide in lecture_slides(module):
        lines.extend(
            [
                f"## {slide.kicker}: {slide.title}",
                slide.body,
                "",
                "**Prompt:** " + slide.prompt,
                "",
                "**Speaker note:** " + slide.speaker_note,
                "",
            ]
        )
    return "\n".join(lines) + "\n"


def teaching_bundle(module: LabModule) -> bytes:
    """Return a self-describing, local-only 'teach this tomorrow' ZIP bundle."""
    files = {
        "README.md": "\n".join(
            [
                f"# Teach this tomorrow - {module.title}",
                "",
                "This local package contains ready-to-share course materials for one HaloForge module. It does not create a classroom, track students, upload responses, or export grades.",
                "",
                "## Contents",
                "",
                "- `student_handout.md`: investigation context and scientific caution.",
                "- `assignment.md`: concise submission brief.",
                "- `instructor_guide.md`: teaching move, solution outline, accessibility, and privacy guidance.",
                "- `solution_guide.md`: discussion-oriented expected reasoning.",
                "- `lecture_outline.md`: six-step projector narrative with prompts and speaker notes.",
                "- `lecture_slides.json`: machine-readable version of the same lecture arc.",
                "- `notebook.ipynb`: local analysis starter for an exported HaloForge bundle.",
                "- `accessibility.md`: delivery accommodations and non-surveillance guidance.",
                "- `manifest.json`: checksums for every included artifact.",
                "",
                "Export a HaloForge run first, place this notebook beside its exported Parquet tables, and make caveats part of the lesson rather than fine print.",
            ]
        )
        + "\n",
        "student_handout.md": student_handout(module),
        "assignment.md": assignment_brief(module),
        "instructor_guide.md": instructor_guide(module),
        "solution_guide.md": solution_guide(module),
        "lecture_outline.md": lecture_outline(module),
        "lecture_slides.json": json.dumps(
            [asdict(slide) for slide in lecture_slides(module)], indent=2
        )
        + "\n",
        "notebook.ipynb": notebook_template(module),
        "accessibility.md": "\n".join(
            [
                "# Accessible delivery notes",
                "",
                "- Offer accessible chart transcripts or exported tables alongside every visual.",
                "- Honor reduced-motion preferences; no learning task depends on animation or hover.",
                "- Allow written predictions and conclusions instead of requiring drag, color discrimination, or fine pointer control.",
                "- Use the grayscale-safe exports for print workflows.",
                "- This package stores no student answers or personally identifying information.",
            ]
        )
        + "\n",
    }
    manifest = {
        "schema_version": "haloforge-teaching-bundle-v2",
        "module": {
            "identifier": module.identifier,
            "title": module.title,
            "duration_minutes": module.duration_minutes,
        },
        "privacy_scope": "Local materials only; no classroom accounts, answer collection, analytics, or grade export.",
        "files": {
            name: {
                "sha256": sha256(content.encode("utf-8")).hexdigest(),
                "bytes": len(content.encode("utf-8")),
            }
            for name, content in files.items()
        },
    }
    files["manifest.json"] = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    buffer = BytesIO()
    with ZipFile(buffer, "w", ZIP_DEFLATED) as archive:
        for name, content in files.items():
            archive.writestr(name, content)
    return buffer.getvalue()


def notebook_template(module: LabModule) -> str:
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    f"# {module.title}\n",
                    "This notebook accompanies a local HaloForge run bundle.\n",
                ],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["## Question\n", module.question + "\n"],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import pandas as pd\n",
                    "power = pd.read_parquet('power_spectrum.parquet')\n",
                    "sigma = pd.read_parquet('sigma.parquet')\n",
                    "display(power.head())\n",
                    "display(sigma.head())\n",
                ],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Interpretation checkpoint\n",
                    module.checkpoint + "\n",
                    "\n",
                    "Caution: " + module.caution + "\n",
                ],
            },
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
            "haloforge_module": asdict(module),
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    return json.dumps(notebook, indent=2)
