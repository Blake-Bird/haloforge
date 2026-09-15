"""Stable, actionable categories for recoverable HaloForge failures."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FailureExplanation:
    code: str
    title: str
    summary: str
    remedy: str
    technical_detail: str


def classify_failure(exc: BaseException) -> FailureExplanation:
    """Translate known failure semantics without hiding the original detail."""
    detail = str(exc).strip() or exc.__class__.__name__
    lowered = detail.lower()
    if (
        "safety timeout" in lowered
        or "exceeded the" in lowered
        and "timeout" in lowered
    ):
        return FailureExplanation(
            "solver_timeout",
            "Calculation took too long",
            "The isolated solver was stopped at its declared safety limit; your prior completed run remains available.",
            "Reduce sampled resolution or requested redshifts, then try again. If the calculation is scientifically important, inspect the solver settings before raising the timeout.",
            detail,
        )
    if "another axiclass solve" in lowered:
        return FailureExplanation(
            "solver_busy",
            "A calculation is already running",
            "HaloForge allows one isolated solver run at a time to avoid competing writes and ambiguous state.",
            "Wait for the active calculation to finish, then submit this run again.",
            detail,
        )
    if "could not be imported" in lowered or "classy binding" in lowered:
        return FailureExplanation(
            "solver_unavailable",
            "The cosmology solver is unavailable",
            "The local CLASS/AxiCLASS binding could not be loaded, so HaloForge did not fabricate a result.",
            "Open Diagnostics to inspect the local solver installation, then restore or install the expected runtime.",
            detail,
        )
    if (
        "unsafe archive" in lowered
        or "integrity verification failed" in lowered
        or "not a readable zip" in lowered
    ):
        return FailureExplanation(
            "unsafe_import",
            "Workspace import was rejected",
            "The archive failed a local safety or integrity check before HaloForge wrote any imported files.",
            "Use an intact HaloForge workspace ZIP, or inspect the archive contents and manifest before trying again.",
            detail,
        )
    if (
        "must be" in lowered
        or "requires" in lowered
        or "must lie" in lowered
        or "range must" in lowered
    ):
        return FailureExplanation(
            "invalid_configuration",
            "The requested configuration is not supported",
            "HaloForge rejected the settings before accepting an invalid calculation or unsupported convention combination.",
            "Review the named control, mass-definition contract, redshift, or numerical range; then run a supported configuration.",
            detail,
        )
    if "integrity" in lowered or "schema" in lowered or "migration" in lowered:
        return FailureExplanation(
            "stored_data_problem",
            "Saved data needs attention",
            "HaloForge did not silently trust a stored artifact whose schema or integrity state needs review.",
            "Keep the original artifact, inspect its integrity/migration record, and restore a verified bundle if available.",
            detail,
        )
    return FailureExplanation(
        "calculation_interrupted",
        "The requested operation stopped safely",
        "HaloForge preserved prior completed work rather than presenting a partial result as completed.",
        "Read the technical detail, adjust the smallest relevant input or retry the same operation if it was an infrastructure interruption.",
        detail,
    )
