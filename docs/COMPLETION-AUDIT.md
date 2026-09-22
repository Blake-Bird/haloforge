# HaloForge completion audit

Updated 2026-09-22. This document records the comprehensive implementation and verification
status of the HaloForge cosmology laboratory, addressing the release-blocking defects, UI
responsiveness, mathematical rigor, and the nonlinear GADGET-4 simulation subsystem.
The canonical atomic ledger remains [REQUIREMENTS.md](REQUIREMENTS.md).

## Status vocabulary

- **Verified implemented** — source code and automated tests exist in the workspace, and were executed and passed during this verification run.
- **Implemented with declared boundaries** — core algorithm, data structures, and tests exist; execution on external multi-node clusters or third-party binaries requires their external environment.
- **Open / external** — requires external hardware, multi-node HPC clusters, external human peer review, or external user accessibility testing.

---

## What this audit implemented and verified

| Subsystem / Concern | Status | Verification & Evidence |
| --- | --- | --- |
| **Saved-run “Benchmark this run” crash** | **Verified implemented** | Resolved `KeyError: 'sigma_result'` by implementing `engine/saved_run.py` and fallback derived `h = H0 / 100.0`. Tested in `tests/test_benchmark_interaction.py`. |
| **One-change experiment planner** | **Verified implemented** | Fixed unwanted multi-parameter drift. Explicit starting baseline contracts implemented (`active`, `named_baseline`, `canonical_preset`) in `engine/experiment_design.py`. Tested in `tests/test_experiment_planner.py`. |
| **Run deletion confirmation & trash recovery** | **Verified implemented** | Added confirmation modal, prevented deletion of sole baseline, and implemented local trash/undo recovery in `app.py`. |
| **Responsive CSS & Overflow containment** | **Implemented with declared boundaries** | CSS contains containment and responsive rules, but browser-level visual regression across every target viewport has not been completed. |
| **Workspace-specific sidebar controls** | **Verified implemented** | Hides the 50-parameter cosmology form from unrelated pages (Teach, Known Limitations, Notebook, Runs, Diagnostics, Evolution, Campaign, Simulation). |
| **Persistent project header** | **Verified implemented** | Displays active run, baseline, draft state, validity badge, and target redshift at top of every research view. |
| **Truncated controls cleanup** | **Verified implemented** | Replaced truncated labels with a clean `"Configure axes"` popover for plot customizations. |
| **Recoverable error boundary** | **Verified implemented** | Catches unhandled Streamlit exceptions with a friendly diagnostic panel and downloadable error log in production mode. |
| **Redshift Evolution Studio ($z=20\rightarrow0$)** | **Implemented with declared boundaries** | Supports uniform in $a$, $z$, and $\ln(a)$ sampling, stored multi-redshift spectra, fixed-axis Plotly views, contact sheets, data tables, and reproducible MP4/WebM/GIF rendering. Exact stored CLASS/AxiCLASS samples are marked as calculated linear theory; interpolated frames are explicitly marked as approximations and report withheld-source validation where possible. EDE frame times come from the solver background; historical EDE runs without it show time as unavailable. EDE-specific growth validation and visual review on all target displays remain open. |
| **Multi-cosmology parameter campaign orchestrator** | **Implemented with declared boundaries** | Supports real isolated CLASS/AxiCLASS members, bounded worker concurrency, Cartesian increments, atomic local campaign records, pause/resume/retry/cancel, and timing estimates only when completed local members supply measurements. Cluster scheduling and independent scaling validation remain open. |
| **GADGET-4 integration architecture** | **Open / external** | The repository creates planning files and reads real HDF5 snapshots, but no immutable image plus official GADGET-4 acceptance run is recorded. |
| **1LPT / 2LPT Initial Conditions generator** | **Implemented with declared boundaries** | Generates normalized first-order fields and an explicit periodic second-order LPT kernel from an exact stored P(k,z_start) slice, with measured Fourier-shell diagnostics and basic invariants. The production UI still exposes 1LPT only. EDE refuses ΛCDM shortcuts and requires solver-derived first- and second-order growth inputs; transfer-function/gauge and independent power-recovery validation remain open. |
| **Periodic-box setup & softening policy** | **Verified implemented** | Built `engine/nbody_setup.py` calculating particle mass, mean separation, Nyquist mode, softening policy (Power et al. 2003), and halo particle resolution thresholds ($M_{20}, M_{100}, M_{300}, M_{1000}$). Tested in `tests/test_nbody_setup.py`. |
| **Periodic FOF halo catalogue generator** | **Implemented with declared boundaries** | Reads validated GADGET-4 PartType1 HDF5 particles, performs periodic FOF at linking length $b=0.20$, and retains only the FOF mass it actually measures. External halo-finder fixture validation remains open. |
| **Finite-bin HMF vs. N-body comparison** | **Implemented with declared boundaries** | Requires the active run’s stored $\sigma(M,z)$, an exact-redshift snapshot, FOF-$b=0.2$ fit selection, and a SHA-256 snapshot-to-run sidecar manifest. It shows Poisson-only descriptive residuals, not a validated accuracy claim. |
| **Automated regression suite** | **Verified implemented** | The current full local suite passed 459 tests with 6 explicitly opt-in AxiCLASS tests skipped on 2026-09-22. The pinned-worker integration suite was then run separately and passed 6/6, including the frozen independent CAMB comparison and exported recreation. Passing tests do not replace external GADGET-4 evidence. |

---

## Detailed checklist reconciliation

### 1. Release blockers & core UX
- **Benchmark crash:** Resolved. Saved runs without stored $\sigma(M)$ derive $h$ accurately from $H_0 / 100.0$ and recompute needed validity metrics.
- **One-change planner:** Preserves explicit parameter anchors and avoids unintended multi-parameter diffs.
- **Destructive run deletion:** Requires explicit confirmation, protects the sole baseline from deletion, and supports trash restoration.
- **Layout & overflow:** Popovers and tables bounded to viewport width. Truncated axes controls replaced with concise popover triggers.
- **Sidebar decluttering:** Universal cosmology form only appears in workspaces where cosmology parameters are directly modified.
- **Production error boundary:** Raw Python tracebacks intercepted by a styled diagnostic panel with download capability.
- **Calculation action:** The primary action is labelled “Calculate & save run,” making the compute-and-durable-save behavior explicit rather than hiding it behind “auto-save.”

### 2. Redshift evolution movie ($z=20 \rightarrow 0$)
- **Sampling schemes:** Uniform in scale factor $a$ (default, prevents visual compression of late-time structure formation), uniform in $z$, and uniform in $\ln(a)$.
- **Fixed axes:** Axis bounds held static during frame playback to convey genuine physical growth rather than misleading auto-scaled visual artifacts.
- **Frame data & contact sheets:** Static multi-panel contact sheets and exact per-frame data tables are provided alongside the interactive scrubber.
- **Video delivery:** The exact precomputed frame sequence can be rendered as MP4, WebM, or GIF with fixed axes and recorded export settings.
- **Scientific provenance:** A frame that exactly matches a stored solver redshift is labelled calculated linear theory. A frame reconstructed between stored redshifts is labelled an approximation; the manifest includes a withheld-source interpolation check when the source grid permits it. For EDE, cosmic time is read from the stored solver background. It is deliberately unavailable for legacy EDE runs that do not contain that background quantity.

### 3. Parameter campaigns & resource orchestration
- **Sampling strategies:** Latin Hypercube Sampling (LHS), Sobol quasi-random sequences, and Cartesian grids.
- **Resource protection:** Worker count is bounded; elapsed-time estimates remain unavailable until local completed solver members provide a measured calibration. Memory and disk remain planning estimates.
- **Sensitivity & trends:** Automated detection of monotonic response, extrema, and parameter sensitivities.
- **Baseline contract:** Campaign members carry only their declared parameter delta. The saved campaign baseline is merged once by the caller; global defaults cannot silently overwrite EDE or numerical settings. A three-member concurrent real CLASS H₀ sweep at 0.05 increments completed successfully on 2026-09-21.

### 4. GADGET-4, 2LPT initial conditions & halo finding
- **GADGET-4 adapter:** Planning configuration and snapshot parsing are present; an immutable image and official acceptance run are still required before execution claims. The visible planning workflow is explicitly collisionless DM-only; gas/SPH and baryonic-feedback workflows are neither configured nor presented as supported. The generated runtime plan now includes the valid plain-ASCII `output_times.txt` companion required by `OutputListFilename`, rather than a dangling reference. It emits one consistent documented unit contract across the parameter file, IC export, and snapshot reader (Mpc/h, $10^{10}\,M_\odot/h$, km/s), and uses the documented `Softening*Class0` parameter names; FoF linking/type/minimum settings are correctly emitted as `Config.sh` compile-time options rather than invented runtime keys. The doctor requires a complete acceptance manifest (`fixture_id`, snapshot SHA-256, completion time, matching image digest, and matching pinned source revision) in addition to Docker and the local immutable image. The local check on 2026-09-21 found Docker 29.6.1, Open MPI 5.0.11, 16 CPU cores, 128 GB RAM, and 556.7 GB free disk, but no configured or locally available immutable GADGET-4 image and no acceptance manifest. Its `ready_for_simulation` result is therefore correctly `false`.
- **EDE GADGET-4 dynamics:** The phenomenological H(a) planning table and invented `ExpansionHistoryFile` parameter were removed. HaloForge now refuses to emit an EDE GADGET-4 runtime path until an externally validated implementation, explicit extension contract, and acceptance data exist.
- **IC generation:** Produces normalized first-order Zel'dovich fields and a separately tested periodic 2LPT mathematical kernel from an exact stored P(k,z_start) slice. The current UI intentionally exposes only the 1LPT path. Transfer-function/gauge validation, independent power-recovery validation, and validated EDE second-order growth inputs remain open; no unvalidated path is presented as production-ready.
- **Periodic FOF:** Friends-of-friends halo finder using periodic distance trees with linking length $b=0.20 \times \bar{d}$, FOF mass calculation only, and high-$z$ resolution reporting. It does not infer $M_{200m}$, $M_{200c}$, radii, or physical velocity units.
- **HMF comparison:** Direct logarithmic binning of FOF-$b=0.2$ masses against a matching FOF fit, Poisson error calculation, standardized residuals, exact saved-run/snapshot linkage, and explicit omitted-systematics warning. No mass-definition conversion is performed.

---

## Test suite execution evidence

```
============================= test session starts ==============================
platform darwin -- Python 3.13.9, pytest-8.4.2, pluggy-1.6.0
rootdir: /Users/bbird/Downloads/Research/haloforge-main
configfile: pyproject.toml
================== 459 passed, 6 skipped in 55.96s ====================
```

The opt-in real solver suite was also executed against the locally resolved
`/Users/bbird/miniforge3/envs/dmresearch/bin/python` worker on 2026-09-22:

```
HALOFORGE_TEST_CLASS=1 pytest -q tests/test_class_integration.py
============================= 6 passed in 8.69s ==============================
```

This verifies the bounded ΛCDM, EDE, curvature, frozen-CAMB, and export
recreation contracts. It does not validate GADGET-4, nonlinear evolution, or
EDE dynamics inside GADGET-4.
