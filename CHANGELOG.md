# Changelog

This project does not yet have a tagged public release.

## Unreleased

- Validate physical inputs before launching CLASS, with shared checks for app and direct solver calls. Reject malformed EDE initial conditions, non-finite cosmological parameters, and fractional sample counts.
- Add a frozen independent CAMB reference for flat ΛCDM matter power and top-hat variance at three redshifts, with a reproducible generator and explicit tolerances.
- Correct structure-field RMS reporting, coordinate orientation, and plot bounds; test Fourier normalization and shared-phase amplitude scaling.

- Added a guided, prediction-first Explore experience with a matched baseline/candidate reveal, a bounded cluster-scale abundance comparison, direct next-route navigation, and staged causal evidence views.
- Guided experiments now begin with no preselected outcome. A learner must deliberately choose an expected outcome—including the scientifically valid “I am not sure yet”—before the safe one-change experiment can be staged or calculated.
- Added eight four-layer conceptual-zoom entries, prepared modules across five subject areas, projector-oriented lecture mode, and local lab-section artifacts with explicit no-authentication/no-hidden-solution boundaries.
- Added scientific cursor explanations, accessible chart transcripts, selectable mass/range synchronization, and explicit brush-to-notebook annotations that require a user note and confirmation.
- Added a versioned canonical validation-case registry for ΛCDM, EDE, curvature, high redshift, low/high mass, and a visible empirical-fit boundary case. These cases are configuration coverage targets, not asserted external reference agreement.
- Added durable table/data-dictionary metadata, PDF/SVG/PNG and grayscale figure exports, citation metadata, and integrity-aware reproducibility bundles.
- Added structured error taxonomy/recovery guidance and opt-in, local-only operational diagnostics. Diagnostics are disabled by default and never collect scientific parameters, run identifiers, notes, IP addresses, or network telemetry.
- Added a per-run, append-only local mutation history for creation, notebook metadata updates, renames, baseline changes, and forks. The hash-linked records use field names rather than notebook or parameter values, appear in the Notebook, and travel as `audit_trail.json` in reproducibility bundles. This detects changes/reordering of retained records, but cannot prevent filesystem edits; it is not authenticated identity, authorization, or shared-workspace audit infrastructure.
- The Notebook now shows the exported artifacts attached to its selected saved run, including local availability and byte size, plus a portable attachment manifest. Missing local files remain visible rather than being treated as attached; artifact attachment is not scientific validation.
- Added conservative graph-insight markers for sampled turning points and comparison-reference crossings. Turning markers are never fit/smoothed extrema; comparison crossings are shown as exact stored hits or amber endpoint brackets, never as fabricated interpolated locations.
- Corrected bounded Tinker08 overdensity interpolation and added fit/mass-definition contracts.
- Removed artificial HMF floors and made comparison-domain gaps explicit.
- Recorded AxiCLASS-derived background matter fractions for supported downstream use.
- Added numerical range-sensitivity diagnostics, provenance records, export manifests, and reproducibility hashes.
- Moved the local default data location outside the source checkout and fail-closed unauthenticated hosted filesystem persistence.
- Added deterministic tests and GitHub Actions verification.
- Strengthened container CI from image construction alone to a bounded startup health check plus a real packaged AxiCLASS smoke calculation, so the shipped image and solver binding are exercised together on every change.
- Updated Pytest and PyArrow to the advisory-fixed releases reported by the dependency audit.
- Bundled Chromium in the production image and added in-image static-export tests, fixing Docker-only PDF/SVG/PNG export failures from Kaleido's missing browser runtime.

## Evidence and release boundary

- [Implementation status](docs/IMPLEMENTATION-STATUS.md) distinguishes completed local work from unimplemented requirements.
- [Known limitations](docs/KNOWN-LIMITATIONS.md) lists open scientific, collaboration, and accessibility limits.
- [Release policy](docs/RELEASE-POLICY.md) states the independent evidence required before public release.
- This changelog reports repository changes only. It does not establish external scientific agreement, classroom effectiveness, accessibility conformance, security review, or public deployment readiness.
