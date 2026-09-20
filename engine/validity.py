"""Versioned, machine-readable scientific validity records for saved runs.

This deliberately preserves the separate evidence claims rather than turning
them into a misleading confidence percentage.
"""

from __future__ import annotations

from engine.assurance import assurance_report
from engine.contracts import all_fit_points_checked
from engine.uncertainty import uncertainty_inventory

VALIDITY_SCHEMA_VERSION = "haloforge-scientific-validity-v3"


def scientific_validity_record(
    run: dict,
    hmf_validity: dict | None = None,
    benchmark: dict | None = None,
) -> dict:
    """Build a stable validity artifact suitable for storage and export.

    ``overall_state`` is intentionally a routing aid, not a scientific score:
    any missing publication evidence keeps it out of publication-ready state.
    """
    claims = assurance_report(run, hmf_validity)
    uncertainties = uncertainty_inventory(run, hmf_validity, benchmark)
    states = {row["claim"]: row["state"] for row in claims}
    computed = states.get("Solver completed") == "pass"
    publication = states.get("Suitable for publication") == "pass"
    if publication:
        overall = "publication_ready"
    elif not computed:
        overall = "not_computed"
    elif states.get("Numerically converged") == "review":
        overall = "computed_needs_scientific_review"
    else:
        overall = "computed_with_unresolved_numerical_evidence"
    return {
        "schema_version": VALIDITY_SCHEMA_VERSION,
        "overall_state": overall,
        "overall_detail": (
            "No publication-ready claim is made automatically; inspect the separate evidence states."
            if overall != "publication_ready"
            else "All recorded publication gates passed."
        ),
        "claims": claims,
        "uncertainties": uncertainties,
        "hmf_contract": {
            "all_evaluated_points_pass_fit_checks": all_fit_points_checked(
                hmf_validity
            ),
            "reasons": list((hmf_validity or {}).get("reasons", [])),
        },
    }
