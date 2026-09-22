# Implementation status

Updated 2026-09-22. Requirements in [OBJECTIVE.md](OBJECTIVE.md) and [REQUIREMENTS.md](REQUIREMENTS.md) are tracked here. Work is ongoing; release approval is incomplete.

## Verification

- **Local suite:** 377 passed, six optional solver tests skipped. The six real AxiCLASS/CAMB tests also pass with the current source mounted read-only into the test runtime (31.04 seconds), including execution of an exported recreation script at z = 2. The older runtime retains the harmless pytest cache-option warning removed from the current Dockerfile. Ruff and formatting pass across 110 Python files.
- **Container:** the 2026-09-18 production and test images build. The production image passes its health check, an isolated real CLASS calculation, and Unicode PDF report generation, and PDF/SVG/PNG static figure exports. Production image: `sha256:00b08d24618d3620248012750aa94f6c76fe4efd5dc576a0de8ddd60c8c49ee2`. The complete test-image suite passed **377 tests** in 300.67 seconds. Its only warning came from a cache-directory option while the pytest cache plugin was disabled; that conflicting option has been removed from the Dockerfile. These container results precede the validity-schema changes below.
- **Browser:** a real matched ΛCDM/EDE experiment completed and opened its saved comparison. The user subsequently exposed a comparison-control crash. It was reproduced in an interaction test, fixed in source, and patched into the running port-8505 instance. The server restarted without changing saved runs. A separate browser session exercised ratio, percent-difference, and redshift changes.
- **Figures:** all 18 PDF/SVG/PNG variants rendered with current export source mounted into the earlier production runtime. Power, variance, and HMF PNGs were inspected. A two-page PDF report containing scientific symbols and literal markup was also rendered and inspected.

The live port-8505 container carries the comparison fix on the older image. It is not the newly built production image. Older checkpoint counts are superseded by the results above.

## Implemented corrections

### Scientific calculations

- Contribution-band percentiles now invert the same piecewise-linear density used for interval integration. Analytic rising, falling, constant, and extreme-amplitude tests pass. Fraction diagnostics reject invalid or unresolved contributions instead of emitting NaNs. The 27 targeted inspection/workspace tests pass; the full source suite is being rerun.

- Inspection rejects non-finite or out-of-range masses, out-of-spectrum Fourier requests, malformed grids, and non-integer result counts. Power and variance select their redshift coordinates independently; the saved smoothing window takes precedence over draft metadata. The dashboard labels nearest-sample selection and formats values with explicit precision. The full suite passes 377 tests after these changes; the temporary preview starts with empty storage, so completed-panel evidence comes from assembled-workspace tests.

- Validity schema v3 requires nonempty Boolean fit-range evidence, distinguishes range checks from simulation calibration, and identifies simulation-fitted Sheth-Tormen coefficients in the assurance panel. The full source suite passes, and 29 targeted validity, uncertainty, inspection, and assembled-workspace tests pass after the final label change.

- Exact piecewise power-law sharp-k integration, including partial cutoff intervals; stable top-hat evaluation and explicit window validation.
- Corrected Reed 2003 and slope-dependent Reed 2007 multiplicities; log-domain evaluation handles extreme positive variance without polynomial overflow.
- Positive cumulative-HMF intervals integrate exactly under piecewise power-law interpolation. Zero endpoints use linear interpolation in log mass; the finite upper mass bound remains explicit.
- Shared strict saved-redshift selection replaces nearest-sample substitution in HMF, variance diagnostics, inspection, sensitivity, and exports.
- Shared physical input contracts and complete solver/cache array checks reject non-finite values, inconsistent grids, invalid shapes, and metadata that attempts to replace scientific arrays.
- The zero-EDE limit follows the ΛCDM path. Closed-universe background matter fractions above one remain valid.
- Sensitivity comparisons reject out-of-range mass queries and confounded runs, and support zero-valued parameter baselines using absolute parameter changes.
- Structure-field tests check Fourier normalization, amplitude scaling, slice RMS, coordinate orientation, and physical plot bounds.

### Persistence and exports

- Source/dependency fingerprints, array checksums, and declared calculation identity accompany saved runs. Legacy identities retain their original verification rules.
- Damaged records are excluded from startup hydration and analysis. Save, export, rename, duplicate, metadata, and baseline operations cannot replace evidence of corruption with a new checksum.
- Malformed drafts recover visibly without overwriting the source file. Custom redshifts and values outside slider presets remain available.
- Power and variance tables and static figures use the declared analysis redshift. Reference HMF figures explicitly state z = 0. Zero tails remain in data tables and are omitted from logarithmic traces.
- Generated recreation scripts now parse their embedded JSON and CSV fields correctly and evaluate the exported redshift. A real-solver regression checks every reproduced power sample.
- PDF reports escape literal notebook text, wrap table cells, split long evidence rows, and embed fonts with scientific-symbol coverage. Font notices are included under `assets/fonts`.

### Application behavior

- Guided EDE comparisons now calculate a genuine zero-EDE baseline and preserve the matched pair through navigation.
- Comparison widgets store scalar run IDs instead of dictionaries containing NumPy arrays. Tests exercise repeated mode, redshift, and baseline changes.
- Damaged-run inspection remains available without offering scientific downloads or mutation actions. Workspace backup and deletion remain available.
- Responsive layouts, plot bounds, light/high-contrast figures, and caption contrast have targeted fixes and tests. Comprehensive accessibility and device review remains incomplete.
- Shipped Streamlit configuration disables framework usage statistics and uses the minimal toolbar. Docker includes Chromium for static exports and runs the application as a non-root user.

## Remaining work

The [repository review](REPOSITORY-REVIEW.md) tracks remaining source, scientific-contract, persistence, UI, and release checks. The independent CAMB fixture covers one documented ΛCDM model; EDE calibration, broader cosmologies, simulation-based halo abundances, and end-to-end accessibility are not established by the present evidence. No publication-ready claim follows automatically from a successful solve or a green test suite.
