"""Plain-language, evidence-aware explanations for cosmology parameter deltas.

This module deliberately does not infer a scientific conclusion from a pretty
plot.  It reports the direct mathematical consequence of an edited input,
then marks the parts that require calibration or a numerical check.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CausalStep:
    parameter: str
    changed: str
    direct_equation: str
    first_signal: str
    downstream: str
    robust: str
    caveat: str
    falsifier: str


_STEPS = {
    "n_s": CausalStep(
        "n_s",
        "the primordial spectral tilt",
        "𝒫ℛ(k)=Aₛ(k/kₚ)ⁿˢ⁻¹",
        "the primordial-spectrum tilt around the pivot kₚ",
        "relative small-scale power → σ(M), especially at lower mass → predicted halo abundance",
        "the direction of the primordial tilt change at fixed Aₛ and kₚ",
        "the size of a halo-count change depends on transfer physics, mass range, and the chosen HMF calibration",
        "re-run with a wider k range and compare supported HMF fits; a conclusion that disappears is not robust",
    ),
    "A_s": CausalStep(
        "A_s",
        "the primordial amplitude",
        "𝒫ℛ(k)=Aₛ(k/kₚ)ⁿˢ⁻¹",
        "the overall primordial-spectrum normalization",
        "matter-power amplitude → σ(M) → collapse rarity and halo abundance",
        "at fixed transfer physics, increasing Aₛ raises linear fluctuation amplitude",
        "a halo-count prediction is model-dependent and is not an observed halo catalogue",
        "check the same claim across the selected HMF fits and numerical-range diagnostics",
    ),
    "f_EDE": CausalStep(
        "f_EDE",
        "the peak early-dark-energy fraction",
        "H²(a) includes the EDE background contribution",
        "the early expansion history near the critical epoch",
        "transfer/growth history → matter P(k) shape and amplitude → σ(M) and halo abundance",
        "the EDE background changes the expansion history when AxiCLASS accepts the configuration",
        "mapping that change to a simulation-calibrated HMF can lie outside the fit's supported cosmologies",
        "compare against ΛCDM with matched settings and inspect calibration and convergence status before interpreting counts",
    ),
    "log10_a_c": CausalStep(
        "log10_a_c",
        "the EDE critical epoch",
        "a_c=10^log₁₀a_c",
        "when the temporary EDE contribution changes the early expansion rate",
        "the timing of transfer processing and growth → scale-dependent matter-power changes",
        "moving a_c changes the timing, not merely a late-time amplitude slider",
        "the observable effect is model- and parameter-combination-dependent",
        "hold fEDE fixed, compare a matched ΛCDM baseline, and test whether the feature survives numerical settings",
    ),
    "Omega_m": CausalStep(
        "Omega_m",
        "today's matter density",
        "ρₘ,₀=Ωₘρcrit,₀",
        "the background density, equality scale, and mass-to-radius mapping",
        "transfer shape and growth plus M↔R mapping → σ(M) → halo abundance",
        "the mass-density and radius mapping change directly",
        "several physical pathways move together, so a single curve cannot isolate the cause",
        "compare one parameter at a time and inspect the P(k), M–R, and σ(M) stages separately",
    ),
    "k_max": CausalStep(
        "k_max",
        "the largest sampled Fourier wavenumber",
        "σ²∝∫dln k k³P(k)W²(kR)",
        "the high-k endpoint of the σ(M) integral",
        "low-mass σ(M) and any derived low-mass HMF prediction",
        "raising k_max gives small smoothing scales access to more sampled modes",
        "this is a numerical coverage change, not a physical cosmology change",
        "use the endpoint-removal convergence diagnostic; do not interpret a changing result as a discovery",
    ),
}


def changed_parameters(baseline: dict, candidate: dict) -> list[str]:
    """Return meaningful scalar edits in a stable, display-friendly order."""
    keys = ["A_s", "n_s", "Omega_m", "f_EDE", "log10_a_c", "k_max"]
    return [key for key in keys if baseline.get(key) != candidate.get(key)]


def explain_change(baseline: dict, candidate: dict) -> list[CausalStep]:
    """Explain direct causes first; unknown edits remain explicit rather than guessed."""
    steps = []
    for key in changed_parameters(baseline, candidate):
        if key == "f_EDE" and not candidate.get("enable_ede", False):
            continue
        if key == "log10_a_c" and not candidate.get("enable_ede", False):
            continue
        step = _STEPS.get(key)
        if step:
            steps.append(step)
    return steps
