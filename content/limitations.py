"""A versioned, user-visible registry of known HaloForge limitations."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Limitation:
    identifier: str
    area: str
    severity: str
    status: str
    statement: str
    impact: str
    mitigation: str


LIMITATIONS_VERSION = "haloforge-known-limitations-v2"

LIMITATIONS = (
    Limitation(
        "external-reference-benchmarks",
        "Scientific validation",
        "high",
        "open",
        "The frozen CAMB reference covers one flat, massless-neutrino ΛCDM cosmology at z = 0, 2, and 10; EDE, curved cosmologies, and empirical HMF calibration lack external reference coverage.",
        "Agreement in the reference cosmology does not validate other cosmologies or empirical halo abundances.",
        "Use the internal benchmark only as a numerical implementation check; independently reproduce and compare before research use.",
    ),
    Limitation(
        "ede-hmf-calibration",
        "Empirical halo mass functions",
        "high",
        "open",
        "An HMF fit can be inside its parameter-range contract while still lacking demonstrated calibration for an EDE cosmology.",
        "A predicted EDE halo-abundance shift may not be suitable for publication.",
        "Treat calibration and cosmology-support states separately; compare fits and seek an independently justified calibration.",
    ),
    Limitation(
        "finite-k-range",
        "Numerics",
        "medium",
        "open",
        "σ(M) is evaluated over the finite sampled AxiCLASS k range.",
        "Endpoint removal diagnostics cannot reveal omitted power outside the solved range.",
        "Run explicit k-range and sampling convergence studies; record the result in the Notebook and Benchmark Lab.",
    ),
    Limitation(
        "linear-structure-field",
        "Visualization",
        "medium",
        "open",
        "The structure-field view is a periodic Gaussian linear realization with shared phases, not an N-body simulation or halo catalogue.",
        "Visual density features must not be interpreted as literal formed galaxies or clusters.",
        "Use it only to compare linear amplitude and shape effects; label exported images as illustrative.",
    ),
    Limitation(
        "hosted-collaboration",
        "Privacy and collaboration",
        "high",
        "open",
        "HaloForge has no authenticated hosted workspace, roles, sharing links, or classroom roster.",
        "It cannot safely support multi-user research or classroom data sharing.",
        "Use local runs and explicit portable bundles only; do not deploy this build as an unauthenticated shared service.",
    ),
    Limitation(
        "accessibility-review",
        "Accessibility",
        "medium",
        "open",
        "The app provides local light/high-contrast/reduced-motion options and chart transcripts, but has not undergone independent assistive-technology or device testing.",
        "Accessibility behavior is not yet externally verified.",
        "Treat the controls as a starting point and obtain keyboard, screen-reader, contrast, and responsive-layout review before release.",
    ),
)


def limitations_rows() -> list[dict]:
    return [asdict(item) for item in LIMITATIONS]
