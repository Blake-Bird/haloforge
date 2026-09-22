# Workspace audit

Updated 2026-09-22. This is a product-facing audit of what each visible
workspace does today. It separates **rendered and automated-tested** behavior
from work that needs a solver, an external binary, or a classroom deployment.
It is not a claim that a rendered control establishes scientific validity.

## Evidence used

- Live local-app walkthrough: Explore, quick navigation, Dashboard, Graph
  Studio, and Teaching Lab.
- `tests/test_app_workspaces.py`: every Research workspace and both Compare
  workspaces render against synthetic, integrity-checked saved science.
- Focused teaching, onboarding, and navigation tests.

## What each screen is for

| Workspace | What is present now | Audit status / next expansion |
| --- | --- | --- |
| Explore | Prediction-led curated experiment, matched baseline/candidate run, staged causal reveal, four concept depths. | Works as an onboarding flow. A real solver is needed to complete the calculation. |
| Dashboard | Active-run summary; P(k), σ(M), mode contribution, and HMF figures; validity and uncertainty panels. | Rendered with saved data. It is dense, and should remain the core summary rather than become another control wall. |
| Graph Studio | Chosen graph modules, layout choices, fit/redshift selection, recipes, axes, transcripts, and exports. | Rendered with saved data. It has real figures; adding more charts should follow a scientific question, not fill space. |
| Compare Lab | Matched-run overlays/ratios/differences, point comparison, units, and caveats. | Rendered with saved pair. It correctly excludes invalid/confounded comparisons. |
| Sensitivity Explorer | One-parameter saved-run finite differences and counterfactual lookup. | Rendered with saved pair. It cannot invent unrun thresholds or replace a posterior. |
| Structure Field | Shared-phase linear density slices and quantitative RMS comparison. | Rendered and numerically tested. It is explicitly not an N-body catalogue. |
| Fit + Window Atlas | Fit contracts, mass definitions, smoothing windows, multiplicity and validity context. | Rendered with saved data. This is the appropriate place for modelling caveats. |
| Design Experiment | One-change planning, held-fixed evidence, pre-run hypothesis, and staging. | Rendered; plan construction has unit coverage. |
| Benchmark Lab | Adaptive σ₈ integration check and validation case matrix. | Works on an existing run. It checks an internal numerical quantity, not external agreement by itself. |
| Performance Lab | Reproducible timing/allocation probe. | Works on an existing run; it is intentionally separate from science validation. |
| Convergence Lab | k-range and sampled-calculation checks with suggested controlled follow-ups. | Works on an existing run; does not prove calibration or physical convergence. |
| Evolution Studio | Multi-redshift linear spectra, scrubber, contact sheet, frame table, and video export. | Rendered with saved data. It is linear-theory evolution, not nonlinear structure evolution. |
| Campaign Lab | LHS/Sobol/grid campaign definition, durable member states, bounded batches, and response views. | UI and orchestration are tested. Actual members need the configured CLASS/AxiCLASS environment. |
| Simulation Lab | Hardware diagnostics, periodic-box planning, IC generation, snapshot/catalogue import, FOF and HMF comparison. | Planning and analysis are implemented; a real GADGET-4 execution path requires its validated external image and acceptance evidence. |
| Teaching Lab | Five modules, live question, lecture, student/instructor materials, local sections, notebooks, and skepticism exercises. | Materials and bundles are tested. It is local teaching support, not a hosted classroom, LMS, or gradebook. |
| Learn the Pipeline | Seven-step visual course: primordial conditions through halo abundance. | Works with a saved run; shows a truthful pre-run stop at the first uncalculated step. |
| Notebook | Questions, predictions, annotations, links, attached artifacts, citations, and exports. | Rendered with saved data; collaboration remains intentionally out of scope without auth/storage boundaries. |
| Runs + Export | Load, rename, duplicate, baseline, trash recovery, import/export, and reproducibility bundles. | Rendered with saved data and integrity checks. Shared hosted storage is intentionally guarded. |
| Diagnostics | Runtime, solver, validity, and range diagnostics. | Rendered with saved data; smoke testing requires a local solver. |
| Known Limitations | Versioned limitation registry and download. | Always available; it is a guardrail, not a substitute for fixing a limitation. |

## Corrections made during this audit

1. Search and navigation now begin with meaningful destinations rather than an
   alphabetical dump.
2. Quick-action navigation no longer produces a Streamlit state error or a
   visible selector/page mismatch.
3. A fresh Research visit opens Dashboard instead of an empty recovery screen.
4. Every prepared teaching module now opens its matching guided experiment,
   retaining the prediction-before-calculation workflow.

## Important gaps that should remain visible

- The navigation tool is a keyboard-operable sidebar search, not yet a
  system-wide Cmd/Ctrl-K palette.
- Teaching does not collect work, track progress, or grade students. Adding
  those needs authenticated, privacy-reviewed accounts and storage boundaries.
- GADGET-4 execution, Rockstar comparison, and production 2LPT are not ready
  to be represented as completed simulation workflows.
- No rendered chart proves a fit is calibrated, an experiment converged, or a
  result is publication-ready; the app should keep showing those distinctions.
