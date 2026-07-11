"""Derived cosmological quantities for the parameter sidebar."""

from __future__ import annotations

from config.constants import (
    NEUTRINO_RADIATION_FACTOR,
    OMEGA_GAMMA_H2_DEFAULT,
    RHO_CRIT_COEFF,
    T_CMB_DEFAULT,
)


def h_from_h0(h0: float) -> float:
    """Return h = H0 / 100."""
    return float(h0) / 100.0


def omega_cdm(params: dict) -> float:
    """Return Omega_cdm from the sampled matter and baryon fractions."""
    return float(params["Omega_m"]) - float(params["Omega_b"])


def omega_radiation(params: dict) -> float:
    """Approximate present radiation density including relativistic neutrinos."""
    h = h_from_h0(float(params["H0"]))
    tcmb = float(params["Tcmb"])
    n_eff = float(params["N_eff"])
    omega_gamma_h2 = OMEGA_GAMMA_H2_DEFAULT * (tcmb / T_CMB_DEFAULT) ** 4
    omega_r_h2 = omega_gamma_h2 * (1.0 + NEUTRINO_RADIATION_FACTOR * n_eff)
    return omega_r_h2 / h**2


def omega_de(params: dict) -> float:
    """Flatness closure residual interpreted as dark energy today."""
    return 1.0 - float(params["Omega_m"]) - float(params["Omega_k"]) - omega_radiation(params)


def rho_crit(params: dict) -> float:
    """Critical density today in Msun Mpc^-3."""
    h = h_from_h0(float(params["H0"]))
    return RHO_CRIT_COEFF * h**2


def rho_matter_0(params: dict) -> float:
    """Mean matter density today in Msun Mpc^-3."""
    return float(params["Omega_m"]) * rho_crit(params)


def ede_scale(params: dict) -> dict:
    """Return the EDE critical scale factor and redshift."""
    a_c = 10.0 ** float(params["log10_a_c"])
    return {"a_c": a_c, "z_c": 1.0 / a_c - 1.0}


def parse_custom_redshifts(text: str) -> tuple[list[float], list[str]]:
    """Parse comma-separated redshifts and collect non-fatal issues."""
    values: list[float] = []
    issues: list[str] = []
    for item in text.split(","):
        stripped = item.strip()
        if not stripped:
            continue
        try:
            value = float(stripped)
        except ValueError:
            issues.append(f"Could not parse redshift value '{stripped}'.")
            continue
        if value < 0:
            issues.append(f"Ignored negative redshift value {value:g}.")
            continue
        values.append(value)
    return values, issues


def combined_redshifts(presets: list[float], custom_text: str) -> tuple[list[float], list[str]]:
    """Merge preset and custom redshifts into a sorted unique list."""
    custom, issues = parse_custom_redshifts(custom_text)
    merged = sorted({float(v) for v in [*presets, *custom]})
    return merged, issues


def derived_quantities(params: dict) -> dict:
    """Return all values requested for the derived card."""
    ede = ede_scale(params)
    return {
        "h": h_from_h0(float(params["H0"])),
        "Omega_cdm": omega_cdm(params),
        "Omega_r": omega_radiation(params),
        "Omega_de": omega_de(params),
        "rho_crit": rho_crit(params),
        "rho0": rho_matter_0(params),
        "a_c": ede["a_c"],
        "z_c": ede["z_c"],
    }

