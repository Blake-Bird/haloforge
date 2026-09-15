"""Scientific contracts for halo-mass-function evaluation.

The contracts separate a computationally defined smoothing mass from a halo
finder's mass definition. HaloForge never converts one definition into
another implicitly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


MASS_DEFINITIONS = {
    "analytic_top_hat": "Analytic top-hat mass (no halo finder)",
    "fof_b0.2": "Friends-of-friends, linking length b = 0.2",
    "so_mean": "Spherical overdensity Δ relative to mean matter density",
}


@dataclass(frozen=True)
class FitContract:
    family: str
    mass_definition: str
    redshift_range: tuple[float, float] | None = None
    log_inv_sigma_range: tuple[float, float] | None = None
    delta_range: tuple[float, float] | None = None
    citation: str = ""


FIT_CONTRACTS = {
    "Press-Schechter 1974": FitContract(
        "analytic", "analytic_top_hat", citation="Press & Schechter 1974"
    ),
    "Sheth-Tormen 2001": FitContract(
        "analytic", "analytic_top_hat", citation="Sheth & Tormen 2001"
    ),
    "Jenkins 2001": FitContract(
        "empirical", "fof_b0.2", (0, 5), (-1.2, 1.05), citation="Jenkins et al. 2001"
    ),
    "Reed 2003": FitContract(
        "empirical", "fof_b0.2", (0, 15), (-1.7, 0.9), citation="Reed et al. 2003"
    ),
    "Warren 2006": FitContract(
        "empirical", "fof_b0.2", (0, 0), citation="Warren et al. 2006"
    ),
    "Reed 2007": FitContract(
        "empirical", "fof_b0.2", (0, 30), (-1.7, 0.9), citation="Reed et al. 2007"
    ),
    "Tinker 2008": FitContract(
        "empirical", "so_mean", (0, 2.5), (-0.6, 0.4), (200, 3200), "Tinker et al. 2008"
    ),
    "Crocce 2010": FitContract(
        "empirical", "fof_b0.2", (0, 2), citation="Crocce et al. 2010"
    ),
    "Courtin 2010": FitContract(
        "empirical", "fof_b0.2", (0, 0), (-0.8, 0.7), citation="Courtin et al. 2010"
    ),
    "Bhattacharya 2011": FitContract(
        "empirical", "fof_b0.2", (0, 2), citation="Bhattacharya et al. 2011"
    ),
    "Angulo 2012": FitContract(
        "empirical", "fof_b0.2", (0, 0), citation="Angulo et al. 2012"
    ),
    "Watson FOF 2013": FitContract(
        "empirical", "fof_b0.2", (0, 30), (-0.55, 1.31), citation="Watson et al. 2013"
    ),
    "Watson SO 2013": FitContract(
        "empirical",
        "so_mean",
        (0, 30),
        delta_range=(75.1, 3200),
        citation="Watson et al. 2013",
    ),
}

FIT_ALIASES = {
    "Press-Schechter": "Press-Schechter 1974",
    "Sheth-Tormen": "Sheth-Tormen 2001",
}


def fit_contract(name: str) -> FitContract:
    name = FIT_ALIASES.get(name, name)
    try:
        return FIT_CONTRACTS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown fitting function: {name}") from exc


def validate_fit_configuration(
    name: str, mass_definition: str, delta_halo: float
) -> None:
    contract = fit_contract(name)
    if mass_definition not in MASS_DEFINITIONS:
        raise ValueError(f"Unknown halo mass definition: {mass_definition}")
    if mass_definition != contract.mass_definition:
        raise ValueError(
            f"{name} requires {MASS_DEFINITIONS[contract.mass_definition]}; "
            f"the selected definition is {MASS_DEFINITIONS[mass_definition]}."
        )
    if contract.delta_range is not None:
        lo, hi = contract.delta_range
        if not np.isfinite(delta_halo) or not lo <= float(delta_halo) <= hi:
            raise ValueError(f"{name} requires {lo:g} ≤ Δmean ≤ {hi:g}.")


def validity_report(
    name: str, sigma, z: float, mass_definition: str, delta_halo: float
) -> dict:
    """Return pointwise calibration status; configuration errors raise."""
    validate_fit_configuration(name, mass_definition, delta_halo)
    contract = fit_contract(name)
    sigma = np.asarray(sigma, dtype=float)
    if np.any(~np.isfinite(sigma)) or np.any(sigma <= 0):
        raise ValueError("sigma must be finite and strictly positive")
    valid = np.ones(sigma.shape, dtype=bool)
    reasons: list[str] = []
    if contract.redshift_range is not None:
        lo, hi = contract.redshift_range
        if not lo <= float(z) <= hi:
            valid[:] = False
            reasons.append(f"z={z:g} is outside the published range {lo:g}–{hi:g}")
    if contract.log_inv_sigma_range is not None:
        lo, hi = contract.log_inv_sigma_range
        lnis = np.log(1 / sigma)
        in_range = (lnis >= lo) & (lnis <= hi)
        valid &= in_range
        if not np.all(in_range):
            reasons.append(f"some ln(σ⁻¹) values lie outside {lo:g} to {hi:g}")
    return {
        "fit": name,
        "family": contract.family,
        "mass_definition": mass_definition,
        "citation": contract.citation,
        "calibrated_mask": valid,
        "status": "calibrated" if np.all(valid) else "outside_calibration",
        "reasons": reasons,
    }
