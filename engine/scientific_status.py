"""Evidence-status contract shared by scientific views and exported artifacts."""

from __future__ import annotations


DEMONSTRATION = "demonstration"
APPROXIMATION = "approximation"
CALCULATED_LINEAR_THEORY = "calculated_linear_theory"
SIMULATION_OUTPUT = "simulation_output"
VALIDATED = "validated"
PUBLICATION_SUPPORTED = "publication_supported"

STATUS_ORDER = {
    DEMONSTRATION: 0,
    APPROXIMATION: 1,
    CALCULATED_LINEAR_THEORY: 2,
    SIMULATION_OUTPUT: 2,
    VALIDATED: 3,
    PUBLICATION_SUPPORTED: 4,
}

STATUS_LABELS = {
    DEMONSTRATION: "Demonstration",
    APPROXIMATION: "Approximation",
    CALCULATED_LINEAR_THEORY: "Calculated linear theory",
    SIMULATION_OUTPUT: "Simulation output",
    VALIDATED: "Validated",
    PUBLICATION_SUPPORTED: "Publication-supported",
}


def require_status(status: str) -> str:
    """Return a known status or fail before an artifact is labelled."""
    if status not in STATUS_ORDER:
        raise ValueError(f"Unknown scientific status: {status}")
    return status


def weakest_status(*statuses: str) -> str:
    """Return the lowest-evidence status among all non-empty inputs."""
    known = [require_status(status) for status in statuses if status]
    if not known:
        raise ValueError("At least one scientific status is required")
    return min(known, key=lambda status: STATUS_ORDER[status])
