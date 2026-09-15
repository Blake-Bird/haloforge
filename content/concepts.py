"""Layered, context-preserving explanations for HaloForge's core concepts."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ConceptZoom:
    """One idea, expressed without conflating intuition with validation."""

    label: str
    intuition: str
    course: str
    research: str
    implementation: str


CONCEPTS = (
    ConceptZoom(
        "Primordial amplitude Aₛ",
        "Aₛ is the starting loudness of the tiny ripples that later gravity can amplify into cosmic structure.",
        "At the pivot scale kₚ, the primordial curvature spectrum is 𝒫ℛ(k)=Aₛ(k/kₚ)ⁿˢ⁻¹. Raising Aₛ rescales the initial spectrum while nₛ controls its tilt.",
        "Aₛ is dimensionless and quoted at a declared pivot; its apparent effect can be degenerate with growth, transfer physics, and other cosmological parameters. Compare matched runs and inspect stored CLASS settings rather than inferring a calibrated halo-count claim from amplitude alone.",
        "HaloForge validates and sends Aₛ plus kₚ to isolated CLASS/AxiCLASS execution, saves the exact settings and reproducibility hash, then uses the returned linear P(k) as input to the deterministic σ(M) calculation. Tests cover serialized run identity and invalid solver outcomes.",
    ),
    ConceptZoom(
        "Primordial tilt nₛ",
        "Tilt decides whether the young universe gives relatively more emphasis to small ripples or large ripples.",
        "𝒫ℛ(k)=Aₛ(k/kₚ)ⁿˢ⁻¹. At fixed Aₛ, increasing nₛ raises relative primordial power above kₚ and lowers it below kₚ.",
        "The pivot is essential: calling a tilt change a uniform amplitude change is wrong. Its later effect on σ(M) is scale-dependent, while any empirical HMF interpretation remains conditional on the selected fit's calibration domain.",
        "The parameter is stored in the submitted run bundle, passed to the solver configuration, and compared with structured causal explanations. The numerical core keeps P(k), σ(M), and HMF evaluation separate so a fitting formula cannot be mistaken for the primordial calculation.",
    ),
    ConceptZoom(
        "Transfer function",
        "The transfer function is the universe's processing filter: it reshapes the first ripples before they become the matter pattern we measure later.",
        "A common schematic is Pₘ(k,z) ∝ 𝒫ℛ(k) T²(k) D²(z), where T(k) carries early-time processing and D(z) carries later linear growth.",
        "T(k) is not an arbitrary visual filter. It depends on the model and solver settings, including radiation, baryons, neutrinos, curvature, and any supported EDE background. HaloForge records solver provenance but does not claim that an illustrative curve independently validates every physical ingredient.",
        "HaloForge delegates linear evolution to isolated CLASS/AxiCLASS, persists returned P(k) and solver settings, and checks cache identity before reuse. Its pure Python post-processing never replaces CLASS/AxiCLASS with a hidden transfer-function approximation.",
    ),
    ConceptZoom(
        "Mass variance σ(M)",
        "Imagine averaging the universe inside larger and larger spheres: σ(M) tells you how lumpy those averaged regions still are.",
        "For a smoothing radius R, σ²(R)=(2π²)⁻¹∫ dk k²P(k)W²(kR); a top-hat convention maps R to a mass M. σ(M) is dimensionless.",
        "A smooth σ(M) curve can still be incomplete if the sampled k-range misses contributing modes or resolution is inadequate. HaloForge exposes endpoint-removal checks and contribution diagnostics, but labels them sampled-range sensitivity rather than complete solver convergence.",
        "The engine uses a fixed log-k grid and Simpson integration over the stored solver spectrum. Numerical diagnostics, arrays, and method provenance are saved with each run; array invariants and integration behavior are independently tested.",
    ),
    ConceptZoom(
        "σ₈",
        "σ₈ is one standard ruler for present-day lumpiness: how uneven matter is after smoothing on a particular large cosmic scale.",
        "It is σ(R=8 h⁻¹ Mpc), evaluated from the same linear power-spectrum integral using a real-space top-hat window. It is dimensionless.",
        "Matching σ₈ does not make two cosmologies identical: their small-scale spectra, redshift evolution, transfer history, or halo-fit support can still differ. Treat it as one derived summary, not a universal validation score.",
        "HaloForge retains the solver-derived σ₈ when supplied, and its internal benchmark compares adaptive and sampled-grid estimates with explicit tolerances. The recorded value and method context travel in run provenance and exports.",
    ),
    ConceptZoom(
        "Top-hat filter",
        "A top-hat filter is a clean 'inside this sphere versus outside it' way to ask what a region of a chosen size contains.",
        "In Fourier space W(x)=3[sin(x)-x cos(x)]/x³ with x=kR. It weights the modes entering σ²(R), and its R-to-M mapping gives the analytic top-hat mass convention.",
        "The filter is a smoothing choice, not a halo finder. Analytic Press-Schechter and Sheth-Tormen references use this convention, while empirical HMF fits require their declared FOF or spherical-overdensity definitions and calibration limits.",
        "HaloForge evaluates a stable series branch near x=0 and the exact form elsewhere, tests both, and fails closed when a requested fit and mass definition conflict. Non-top-hat variance exploration does not silently unlock calibrated HMF products.",
    ),
    ConceptZoom(
        "Early dark energy",
        "Early dark energy is a temporary ingredient that can alter how quickly the young universe expands before fading into the background.",
        "The supported axion-like EDE model changes the background and linear perturbation evolution around its critical epoch; its consequences propagate through P(k), smoothing, and model-dependent halo predictions.",
        "Exact CLASS settings, model support, numerical diagnostics, and HMF calibration status are separate evidence. An EDE-induced curve difference is not by itself a calibrated prediction of real halo abundance.",
        "HaloForge constructs an AxiCLASS request in an isolated worker, records timeout/retry semantics and solver provenance, then saves arrays and validity claims atomically with the run.",
    ),
    ConceptZoom(
        "Halo mass definition",
        "A halo edge is not one universal physical line; it is a convention that has to be named before counts can be compared honestly.",
        "Common conventions include analytic top-hat mass, friends-of-friends with b=0.2, and spherical overdensity Δ relative to mean matter density. A fitted HMF is defined for particular conventions.",
        "Calibration domains include mass definition, redshift, ln(σ⁻¹), and sometimes overdensity. Outside-domain values are numerical extrapolations, not evidence that the fit applies to a new cosmology.",
        "A central fit-contract registry validates compatible definitions before HMF evaluation, produces pointwise calibration masks, and preserves analytic references separately. Unsupported combinations fail closed rather than being converted implicitly.",
    ),
)


def concept_by_label(label: str) -> ConceptZoom:
    for concept in CONCEPTS:
        if concept.label == label:
            return concept
    raise KeyError(f"Unknown HaloForge concept: {label}")
