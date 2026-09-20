---
title: HaloForge
emoji: 🌌
colorFrom: cyan
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
---

# HaloForge

**Explore how cosmological parameters change matter clustering and halo abundance.**

HaloForge connects the primordial spectrum to linear matter power, smoothed mass variance, and halo mass functions in an interactive Python application. It uses the compiled [AxiCLASS](https://github.com/PoulinV/AxiCLASS) solver for both ΛCDM and axion early dark energy (EDE). Failed calculations produce diagnostics, never substitute spectra.

Use it to teach the calculation step by step, compare controlled cosmological experiments, inspect numerical sensitivity, and export calculations for further analysis. Research use requires checking numerical convergence and the calibration of the chosen halo model; the application does not certify a result for publication.

## Workspaces

- **Explore:** make a prediction, calculate a matched ΛCDM reference and candidate, and follow their power, variance, and abundance results.
- **Compare:** overlay saved runs or inspect ratios and residuals at common redshifts. Match smoothing windows and halo models; compare at a chosen physical scale.
- **Research:** inspect individual graphs, numerical diagnostics, fit conventions, Gaussian density fields, experiment notes, and export bundles. Teaching materials include student and instructor notes.

Controls are submitted explicitly. Changing a slider does not launch CLASS, and displayed results remain associated with the completed run. Each solver call runs in a separate process with a timeout so a native failure cannot terminate the application.

## Run with Docker

Install Docker Desktop, then run from this directory:

```bash
docker compose up --build -d
```

Open [localhost:7860](http://localhost:7860). The image compiles AxiCLASS revision `1b0a585f86a3dce6babd66e486535368b2799ec7` and includes Chromium for static figure export.

```bash
docker compose down              # Stop; retain saved runs.
docker compose up --build -d     # Rebuild after updating the source.
```

Saved runs live in the private `haloforge_data` Docker volume. Removing that volume deletes the results; download export bundles before removing it.

## Develop locally

Use Python 3.11 in an isolated environment. Install the application and development dependencies:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
```

Build the pinned AxiCLASS binding in the same environment. Its build requires a C/C++ compiler and `make`. The binding imports Cython during metadata generation, so its installation needs build isolation disabled:

```bash
python -m pip install "Cython==0.29.36" "numpy==1.26.4"
git clone https://github.com/PoulinV/AxiCLASS.git /tmp/AxiCLASS
cd /tmp/AxiCLASS
git checkout 1b0a585f86a3dce6babd66e486535368b2799ec7
make class libclass.a -j2
python -m pip install --no-build-isolation .
```

Return to the HaloForge directory and start the app:

```bash
streamlit run app.py
```

The Python package is named `classy`. An unrelated CLASS installation may not accept EDE settings; use the pinned AxiCLASS source. Static SVG/PNG exports also require a Chrome/Chromium installation; the Docker setup includes it.

The project’s `.streamlit/config.toml` disables Streamlit usage statistics and sets the shared interface theme. The same configuration is included in the Docker image.

## Scientific methods and conventions

All power spectra are **linear**, in synchronous gauge, sampled directly at every requested redshift. The supported species are baryons, cold dark matter, photons, and massless relativistic neutrinos. Massive neutrinos and nonlinear power prescriptions are not exposed.

| Quantity | Convention |
| --- | --- |
| Primordial curvature power | Dimensionless 𝒫ℛ(k) = Aₛ(k/kₚ)ⁿˢ⁻¹ |
| Wavenumber k | Mpc⁻¹, not h Mpc⁻¹ |
| Matter power P(k,z) | Mpc³ |
| Dimensionless matter power | Δ²(k,z) = k³P(k,z)/(2π²) |
| Displayed mass Mₕ | Numerical mass in h⁻¹ M☉; physical M = Mₕ/h |
| Mean density | ρₘ,₀ = Ωₘ × 2.775 × 10¹¹ h² M☉ Mpc⁻³ |
| Top-hat radius | R = [3M/(4πρₘ,₀)]¹ᐟ³, in Mpc |
| Differential abundance | dn/dlnM in h³ Mpc⁻³ |
| Cumulative abundance | Integral from M to the largest sampled mass, not to infinity |

The variance is

```text
σ²(R,z) = ∫ dln k [k³ P(k,z) / (2π²)] W²(kR).
```

- **Top-hat:** W(y) = 3(sin y − y cos y)/y³, with a small-argument series to avoid cancellation.
- **Gaussian:** W(y) = exp(−y²/2).
- **Sharp-k:** W(y) = 1 for |y| ≤ 1 and zero otherwise.

Smooth windows use Simpson integration over the sampled logarithmic k grid. Sharp-k uses an analytic integral of the piecewise power-law spectrum up to `min(1/R, k_max)`, including the final partial interval. This prevents artificial steps as the cutoff crosses grid samples. Neither method extrapolates outside the computed k range. σ₈ always uses the top-hat radius 8/h Mpc, regardless of the selected exploration window.

The halo calculation is

```text
dn/dlnM = (ρₘ,₀/M) f(σ) |dlnσ/dlnM|.
```

Cumulative counts integrate each positive interval as a power law in mass, exactly matching its log-log interpolant. Intervals with a zero endpoint use a linear interpolant in ln M. The integral ends at the largest sampled mass and is exactly zero there.

Logarithmic derivatives use second-order finite differences, including at the mass-grid endpoints. Reed 2007 additionally uses n_eff = −6 dlnσ/dlnM − 3. Watson SO uses Ωₘ(z) from the solver background, including EDE and curvature. The exact zero-EDE fraction uses ΛCDM settings without scalar-field shooting.

Halo abundance requires **real-space top-hat smoothing**. Gaussian and sharp-k outputs are variance explorations at the same reference R(M); they do not supply calibrated alternative halo mass assignments. Empirical fits retain their friends-of-friends or spherical-overdensity mass definitions. HaloForge does not silently convert between them. Calibration masks describe the implemented range checks, not complete validation of a cosmology. See [scientific references and corrections](docs/SCIENCE-REFERENCES.md) and the [mass-definition design decision](docs/adr/0002-halo-mass-definition-and-calibration-contracts.md).

## Saved runs and exports

Runs retain submitted parameters, solver settings, arrays, notes, lineage, and numerical diagnostics. New calculations record software provenance and a reproducibility hash. Data writes are atomic, and stored arrays are checked for integrity when loaded.

Export bundles include CSV/Parquet tables, parameter and provenance records, a manifest with artifact checksums, a run report, and eligible figures in PDF/SVG/PNG. Static power and variance figures use the saved run’s focused redshift, stated in their titles. Analytic HMF reference figures remain explicitly at z = 0. Zero abundances are retained in exported data and omitted from logarithmic traces. Recreation starters and teaching notebooks support further work outside the app. The bundle records what was calculated; independent reproduction remains the researcher's responsibility.

Without configuration, local storage uses the operating system's per-user application-data directory, such as `~/Library/Application Support/HaloForge` on macOS. Set `HALOFORGE_DATA_DIR` to use another private directory. Runs, exports, caches, and secrets are excluded from the source distribution.

**This build is for single-user local use.** `HALOFORGE_DEPLOYMENT=hosted` stops the app before accessing data. Shared hosting requires authenticated, isolated per-user storage and access control, which are not implemented.

## Validation

```bash
ruff format --check app.py config content engine state tests scripts
ruff check app.py config content engine state tests scripts
pytest -q
HALOFORGE_TEST_CLASS=1 pytest -q tests/test_class_integration.py
pip-audit -r requirements.txt
docker compose config
```

The ordinary suite covers analytic variance and derivative identities, HMF equations, units, calibration contracts, saved-run integrity, exports, failure handling, and assembled application workspaces. The separate solver suite requires the compiled binding and checks ΛCDM/EDE execution, the zero-EDE limit, curved backgrounds, growth, pipeline agreement with CLASS σ₈, and independent CAMB P(k), σ(M), and σ₈ reference tables. CI enables that suite in the production-derived test image and verifies the shipped image's health, solver, and static exports.

For a research result, additionally:

1. Vary k limits, k sampling, and mass resolution until the quantities used in the conclusion converge.
2. Check σ₈ against the solver and inspect any integration warnings.
3. Verify the halo definition and the published mass, redshift, overdensity, and cosmology calibration of the selected fit.
4. Reproduce the relevant observables with an independent implementation or reference dataset.
5. Retain the exported settings, provenance, numerical evidence, and citations with the analysis.

## Limits

- EDE linear spectra do not establish EDE calibration of empirical halo mass functions.
- Endpoint-removal diagnostics measure sensitivity within the sampled range; they cannot detect missing power outside it.
- Gaussian structure images are linear random-field slices with shared Fourier phases, not N-body simulations or halo catalogues.
- Standalone multiplicity plots label fixed n_eff or Ωₘ reference slices where needed. They are not substitutes for the run-dependent HMF calculation.
- The [frozen CAMB reference](tests/reference/README.md) covers one flat ΛCDM cosmology. External EDE/HMF validation and comprehensive accessibility review remain outstanding.

See [known limitations](docs/KNOWN-LIMITATIONS.md), [implementation status](docs/IMPLEMENTATION-STATUS.md), and [release policy](docs/RELEASE-POLICY.md).

## References

- Lesgourgues (2011), [CLASS: overview](https://arxiv.org/abs/1104.2932); Blas, Lesgourgues & Tram (2011), [CLASS: approximation schemes](https://arxiv.org/abs/1104.2933).
- Poulin et al. (2019), [Early Dark Energy Can Resolve the Hubble Tension](https://arxiv.org/abs/1811.04083); [AxiCLASS source](https://github.com/PoulinV/AxiCLASS).
- Press & Schechter (1974), [Formation of Galaxies and Clusters of Galaxies by Self-Similar Gravitational Condensation](https://doi.org/10.1086/152650).
- Sheth & Tormen (1999), [Large-scale bias and the peak background split](https://arxiv.org/abs/astro-ph/9901122).
- Reed et al. (2003), [Evolution of the Mass Function of Dark Matter Haloes](https://arxiv.org/abs/astro-ph/0301270); Reed et al. (2007), [The halo mass function from the dark ages through the present day](https://arxiv.org/abs/astro-ph/0607150).
- Tinker et al. (2008), [Toward a halo mass function for precision cosmology](https://arxiv.org/abs/0803.2706).
- Schneider, Smith & Reed (2013), [Halo mass function and the free streaming scale](https://academic.oup.com/mnras/article/433/2/1573/1750290).

Each fit's source and mass definition are also recorded in its application contract and export metadata. Cite the solver and the specific halo model used, rather than this list as a whole.

## Contributing and license

See [CONTRIBUTING.md](CONTRIBUTING.md), [SECURITY.md](SECURITY.md), and [CHANGELOG.md](CHANGELOG.md). Citation metadata is in [CITATION.cff](CITATION.cff).

No open-source license has been selected. Until the rights holder adds one, do not assume permission to reuse, redistribute, or host the source. License selection and third-party attribution review remain prerequisites for an open-source release.
