"""Machine-readable source metadata for reproducible HaloForge exports."""

from __future__ import annotations

from engine.contracts import fit_contract
from state.notebook import normalize_notebook_entry


CITATION_SCHEMA_VERSION = "haloforge-citations-v1"


def _entry(identifier: str, citation: str, role: str, evidence: str) -> dict:
    return {"id": identifier, "citation": citation, "role": role, "evidence": evidence}


def export_citations(run: dict) -> dict:
    """Return citation metadata without fabricating external validation claims."""
    provenance = run.get("provenance", {})
    notebook = normalize_notebook_entry(run.get("notebook"))
    hmf_sources = [
        _entry(
            "press-schechter-1974",
            "Press & Schechter 1974",
            "analytic HMF reference",
            "The exported curve is the stored analytic top-hat reference, not an empirical calibration.",
        ),
        _entry(
            "sheth-tormen-2001",
            "Sheth & Tormen 2001",
            "analytic HMF reference",
            "The exported curve is the stored analytic top-hat reference, not an empirical calibration.",
        ),
    ]
    selected_fit = run.get("params", {}).get("fitting")
    if selected_fit:
        try:
            contract = fit_contract(selected_fit)
            hmf_sources.append(
                _entry(
                    f"fit-{selected_fit.lower().replace(' ', '-')}",
                    contract.citation,
                    "selected HMF contract",
                    f"Mass definition: {contract.mass_definition}; calibration status is in scientific_validity.json.",
                )
            )
        except ValueError:
            pass
    solver = _entry(
        "stored-solver-provenance",
        f"Stored {run.get('class_status', 'solver')} calculation; AxiCLASS commit {provenance.get('axiclass_commit', 'not recorded')}",
        "linear-spectrum computational provenance",
        "Exact submitted settings and software identity are in class_settings.json and provenance.json; this is not an independent external validation citation.",
    )
    figure_sources = {
        "matter_power": [solver],
        "mass_variance": [
            solver,
            _entry(
                "haloforge-sigma-method",
                "HaloForge stored log-k Simpson smoothing calculation",
                "numerical method",
                "The finite sampled-range limits and checks are in scientific_validity.json.",
            ),
        ],
        "analytic_hmf_reference": hmf_sources,
    }
    return {
        "schema_version": CITATION_SCHEMA_VERSION,
        "figures": {
            f"{stem}.{extension}": sources
            for base_stem, sources in figure_sources.items()
            for stem in (base_stem, f"{base_stem}_grayscale")
            for extension in ("pdf", "svg", "png")
        },
        "user_notebook_citations": list(notebook["citations"]),
        "scope_limit": "This metadata records provenance and cited modelling ingredients. It does not establish independent numerical agreement, fit calibration for every cosmology, or publication suitability.",
    }
