# HaloForge master acceptance checklist

Working draft, 2026-09-22. This consolidates the external code/science review, the [workspace audit](WORKSPACE-AUDIT.md), the [384-row requirements ledger](REQUIREMENTS.md), the [known limitations](KNOWN-LIMITATIONS.md), and the [release policy](RELEASE-POLICY.md).

This is a checklist of **evidence to obtain**, not a promise of a numerical score. A test passing means its stated behavior passed; it does not prove a cosmology is calibrated, a simulation is physically valid, or a lesson works for students. “100/100” can only mean that an agreed, bounded scope has passed published acceptance gates and independent review. It cannot mean that all scientific uncertainty has disappeared.

**Current baseline:** the local deterministic suite passed 485 tests with 6 opt-in solver tests skipped on 2026-09-22. The IC (h)-unit and GADGET velocity defects were corrected, Evolution Studio now shares core HMF routines, the unused “Go to…” control was removed, and students can export their own saved work. Recorded snapshot particle/density views and GIF/MP4 movie export now have synthetic-file tests; saved FOF Parquet catalogues carry box size and units. Those changes are implemented in the current working tree; real GADGET/Rockstar execution, native finder comparison, independent IC/GADGET acceptance, broad external science benchmarks, and a public release review remain open. P0 means a blocker for the stated scientific claim; P1 means a blocker for a polished public release; P2 means optional portfolio evidence beyond HaloForge.

## How to use this checklist

- [ ] For every item, record an owner, priority, evidence link, review date, and decision in `REQUIREMENTS.md` or a linked evidence record.
- [ ] Use statuses **verified**, **implemented but unverified**, **blocked by external evidence**, **deferred**, and **removed from scope**. Do not call an item complete because a related screen renders.
- [ ] Keep unit tests, solver tests, visual checks, user studies, and independent scientific comparisons separate in the evidence record.
- [ ] Freeze a release candidate before final validation; record its commit, dependency lock, solver revision, container digest, reference-data version, and test environment.
- [ ] Re-run affected acceptance gates after every later change to equations, units, defaults, calibration, storage schemas, or teaching claims.
- [ ] Define the precise supported scientific domain before seeking a “publication-ready” verdict. Everything outside that domain must be visibly unsupported or exploratory.

## 0. Reconcile scope and the old ledger — prerequisite to every score

- [ ] Walk all **R001–R384** rows and map each to a current feature, evidence gap, deliberate deferral, or removal from scope. Preserve the original text in history; publish the revised decision.
- [ ] Update `COMPLETION-AUDIT.md` and other historical test-count claims so their dates, revisions, and evidence are not confused with the current working tree.
- [ ] Remove the obsolete assumption that every aspirational sentence in `OBJECTIVE.md` must be built before release.
- [ ] Reconcile R151–R167 and related classroom language with the current decision: students run their own labs and export their own work; there is no grading, grade export, response collection, or student surveillance.
- [ ] Decide whether instructor materials remain as local discussion aids; remove “hidden solutions,” classroom dashboards, and answer analytics from the acceptance target unless the product direction explicitly changes.
- [ ] Decide whether authenticated **research** collaboration belongs in this release. If deferred, mark hosted sharing requirements deferred and keep hosted mode disabled; do not count absent collaboration as a defect in a local-only release.
- [ ] Decide whether gallery, viral, challenge, sound, and ML/emulator ideas are release requirements, experiments, or out of scope. Do not build them merely to satisfy old prose.
- [ ] Resolve the conflict between “browser-private vault” language and the actual local filesystem vault; document the real storage and privacy boundary.
- [ ] Replace vague targets such as “beautiful,” “memorable,” and “Jane Street-grade” with observable review criteria and named reviewers; retain qualitative judgment where it cannot be measured.
- [ ] Define separate acceptance verdicts for **teaching**, **linear-theory research diagnostics**, **HMF predictions**, and **N-body simulation**. Passing one must never upgrade another.
- [ ] Publish a one-page supported/unsupported capability matrix in the app, README, exports, and release notes.

## 1. Immediate scientific blockers and unit contracts — P0

- [ ] Audit every boundary between physical Mpc and Mpc/h, Mpc⁻¹ and h Mpc⁻¹, Mpc³ and (Mpc/h)³, physical and h-scaled masses, and physical peculiar and GADGET stored velocity.
- [ ] Keep the new CLASS-to-box (k,P) conversion and nonzero GADGET velocity tests as permanent regressions; add randomized (h\ne1) cases and coverage-edge cases.
- [ ] Independently recover the physical input power from generated ICs at several (h), box sizes, grid sizes, spectral shapes, seeds, and redshifts. Predeclare statistical tolerances.
- [ ] Compare identical cosmology, phases, box, transfer function, and starting redshift against an established independent IC generator; record mode-level power, displacement, velocity, and cross-correlation residuals.
- [ ] Verify displacement and velocity conventions against the actual GADGET-4 input path, including scale factor, Hubble units, coordinate units, and header metadata.
- [ ] Run a nonzero-velocity IC through an immutable GADGET-4 build and confirm the ingested values and first-step behavior independently of HaloForge's reader.
- [ ] Test the IC fundamental and Nyquist coverage checks in **physical** units, including exact endpoints and a deliberately insufficient CLASS range.
- [ ] Test Fourier Hermitian constraints, DC/Nyquist handling, normalization, periodic wrapping, center-of-mass momentum, paired-fixed phases, and seed reproducibility against independent calculations.
- [ ] Decide whether the current 1LPT path is suitable only for demonstration or for validated simulation starts; label the UI and exports accordingly.
- [ ] Keep 2LPT out of production claims until second-order growth, gauge/transfer choice, displacement sign and normalization, EDE treatment, and independent generator comparisons pass.
- [ ] Validate snapshot coordinate, mass, velocity, redshift, ID, and particle-type interpretation with official-format fixtures, not only files written by HaloForge.
- [ ] Version the entire IC unit contract and include it in every generated IC manifest and exported run report.

## 2. Linear spectra, cosmology, and variance — P0

- [ ] Recheck the CLASS/AxiCLASS input/output contract against the pinned solver revision: species, gauge, redshift grid, units, (P(k,z)), growth, and background fields.
- [ ] Freeze independent reference tables for several flat ΛCDM cosmologies, not only the current single CAMB fixture. Include changes in (H_0), (Omega_m), (Omega_b), (A_s), (n_s), and relevant edge cases.
- [ ] Add independently generated curvature and high-redshift references with documented solver settings and discrepancy budgets.
- [ ] For EDE, compare AxiCLASS spectra, growth, and background against an independently configured reference or a second implementation where one exists; identify any quantities that cannot be independently checked.
- [ ] Verify exact zero-EDE recovery, very small EDE fractions, invalid EDE parameters, and solver failure isolation with the compiled solver in CI.
- [ ] Predeclare tolerances by observable and regime; explain numerical, solver, and reference uncertainty rather than using a single global percentage.
- [ ] Recheck top-hat, Gaussian, and sharp-k windows with analytic limits, small-argument behavior, and broad power-law examples.
- [ ] Verify (sigma_8) at (R=8/h) Mpc against both the solver and an independent integral for every benchmark case.
- [ ] Perform explicit low- and high-(k) extension studies for each claimed mass range; endpoint removal alone is insufficient evidence about omitted power.
- [ ] Map each displayed mass range to the Fourier support that dominates its variance; block or flag masses whose support is not adequate.
- [ ] Compare Simpson and sharp-k piecewise integration against independent high-accuracy quadrature on smooth, oscillatory, steep, and truncated spectra.
- [ ] Verify (d\ln\sigma/d\ln M) at interior and endpoint masses under grid refinement; quantify rare-tail amplification of derivative error.
- [ ] Define and test how invalid, nonfinite, negative, duplicate, unordered, or insufficient solver samples fail closed.
- [ ] Distinguish exact stored redshift samples from interpolated frames everywhere, including exported figures and video manifests.

## 3. HMF formulas, mass definitions, and calibration — P0

- [ ] Audit every implemented multiplicity formula against its **primary paper**, including coefficient precision, redshift dependence, logarithm base, overdensity reference, and stated range.
- [ ] Build a fit-by-fit evidence table: source equation/table, halo finder, mass definition, overdensity, cosmology set, redshift, (sigma) or mass range, interpolation rule, and validation residual.
- [ ] Finish the Sheth–Tormen audit. Keep it semi-empirical and unverified until a defensible catalogue-mass interpretation and simulation calibration range are documented; do not turn an analytic smoothing mass into a FOF or SO mass silently.
- [ ] Verify that Press–Schechter is described as an analytic reference, not a simulation-calibrated accuracy claim.
- [ ] Validate Tinker, Watson, Reed, Jenkins, Warren, Crocce, Courtin, Bhattacharya, and Angulo implementations against independently coded reference values at interior points and boundaries.
- [ ] Compare supported fits against at least one external HMF implementation after matching definitions, overdensity convention, redshift, units, and parameter options.
- [ ] Keep pointwise domain masks, cosmology-support state, numerical-convergence state, and publication suitability as separate fields; prevent an “inside range” mask from becoming an overall calibrated verdict.
- [ ] For EDE, document which empirical fits have no demonstrated calibration and block publication-style claims unless external simulation or literature evidence supports them.
- [ ] Verify (Omega_m(z)) for background-dependent fits from stored solver data, including curvature and EDE. No ΛCDM closure shortcut may enter the EDE Watson path.
- [ ] Verify the HMF (h^3\mathrm{Mpc}^{-3}) conversion and mass convention against independent dimensional calculations.
- [ ] Test the cumulative HMF against analytic functions, extreme dynamic range, zero tails, and the exact zero at the finite upper mass boundary.
- [ ] Make every cumulative plot and export state “integrated to sampled (M_\max),” including behavior on log axes where zero is omitted.
- [ ] Add fit-choice sensitivity views that show how a conclusion changes across **compatible** models without pretending their disagreement is a calibrated error bar.
- [ ] Review every chart, teaching explanation, and export for mass-definition or EDE-calibration overclaims.

## 4. Evolution Studio and simulation science — P0/P1

The detailed clean-machine GADGET-4/FoF/Rockstar implementation and acceptance gates are in [NBODY-PLATFORM-CHECKLIST.md](NBODY-PLATFORM-CHECKLIST.md).

- [ ] Confirm Evolution Studio uses the authoritative (sigma), differential HMF, cumulative HMF, and solver-background paths for every frame and fit; prevent another duplicate physics path from appearing.
- [ ] Test its selected fit and mass definition against the active run and label them in frames, transcripts, tables, and video exports.
- [ ] Test EDE frames with and without complete background data; unavailable quantities must remain unavailable instead of acquiring a ΛCDM fallback.
- [ ] Audit the cosmic-time approximation for non-EDE cosmologies, especially radiation and curvature; give its scope in the UI.
- [ ] Verify interpolation error estimates on withheld solver redshifts and set a visible threshold for when a movie may imply a smooth evolution.
- [ ] Keep “linear (Delta^2\approx1)” labelled as a heuristic nonlinear scale, never a halo-collapse criterion.
- [ ] Ensure linear density slices, particle ICs, simulated snapshots, and halo catalogues have distinct names, legends, exports, and provenance.
- [ ] Obtain the validated, pinned GADGET-4 image/source revision and an acceptance manifest with input, output hashes, commands, configuration, and completion evidence.
- [ ] Run convergence studies over particle number, box size, softening, initial redshift, time stepping, and force accuracy before claiming quantitative N-body results.
- [ ] Compare HaloForge FOF catalogues with an independent halo finder on shared snapshots, including boundary-crossing groups and resolution cuts.
- [ ] Add Rockstar only after its actual execution and catalogue interpretation are validated; otherwise remove it from release promises.
- [ ] Evaluate finite-volume variance, Poisson error, force resolution, missing long modes, and halo-finder systematics in HMF-versus-simulation comparisons.
- [ ] For EDE simulations, require a validated expansion history and dynamics contract accepted by the chosen N-body code; keep the current fail-closed behavior until then.

## 5. Independent benchmarks and scientific release evidence — P0

- [ ] Publish versioned frozen inputs, expected outputs, tolerances, generation scripts, and citations for every external reference.
- [ ] Include canonical cases spanning baseline ΛCDM, parameter extremes, curvature, high redshift, finite (k)-range, low/high masses, and invalid fit domains.
- [ ] Compare (P(k,z)), (sigma(M,z)), (sigma_8), (dn/d\ln M), cumulative counts, and background quantities separately; report signed residuals by point and summary metrics.
- [ ] Have a second person independently reproduce at least one case from the exported settings without using HaloForge internals.
- [ ] Investigate and publish every benchmark failure; do not loosen a tolerance without a scientific explanation and review.
- [ ] Run the compiled-solver suite in the pinned production-derived image for every scientific release candidate; attach logs and image digest.
- [ ] Add visual regression artifacts to scientific pull requests so changed curves and validity masks are inspectable.
- [ ] Run CPU/memory and timing baselines separately from science benchmarks; never present speed as accuracy.
- [ ] Commission a domain expert to review equations, conventions, calibration claims, ICs, and representative exports; preserve their findings and the response to each.
- [ ] Write a sample publication-style analysis using HaloForge, reproduce its numbers independently, and state which claims are supported and which are exploratory.

## 6. Architecture, tests, and numerical discipline — P1

- [ ] Continue extracting Research and Compare workspaces from `app.py` into focused UI modules; leave scientific calculations in `engine/` and storage rules in `state/`.
- [ ] Make the scientific core importable and usable without Streamlit; add a documented pure-Python API example.
- [ ] Centralize units, cosmology defaults, fit contracts, citations, and conversions in one authoritative layer.
- [ ] Replace implicit session-state transitions with named, testable actions for calculate, save, compare, export, import, and recovery.
- [ ] Introduce typed validated schemas for parameter sets, solver outputs, saved runs, figures, ICs, and manifests, with explicit version migration.
- [ ] Name every numerical invariant and test it: positivity, monotonic axes, physical dimensions, conservation, finite results, exact endpoints, and provenance binding.
- [ ] Add property-based and metamorphic tests for scaling, grid refinement, duplication, invalid combinations, and no hidden extrapolation.
- [ ] Fuzz archive import, JSON metadata, filename handling, solver-output validation, and HDF5 readers within bounded resource limits.
- [ ] Add meaningful integration tests that cross modules without merely repeating implementation formulas.
- [ ] Run static analysis, formatting, unit tests, solver tests, export smoke tests, dependency audit, and container startup in CI on a pinned environment.
- [ ] Set documented performance budgets for startup, UI interaction, calculation setup, figure export, memory, and disk use; benchmark before/after optimizations.
- [ ] Verify every cache key includes input identity, solver/config version, schema, and ownership scope; test invalidation after code or data changes.
- [ ] Review dependencies for necessity, version pinning, security fixes, license compatibility, and reproducible clean installation.
- [ ] Keep architecture decisions for unit conventions, mass definitions, storage integrity, solver isolation, and simulation scope up to date.

## 7. Storage, exports, and reproducibility — P1

- [ ] Test atomic save interruption at each write step and recovery from partial metadata, arrays, exports, and manifest writes.
- [ ] Test saved-array structural validation, checksum failures, schema migrations, legacy records, trash recovery, and restore collisions.
- [ ] Demonstrate that an invalid saved run cannot silently enter comparison, teaching export, or scientific figures.
- [ ] Make the student export include the **latest** saved answers, numerical tables, units, run settings, provenance, citations, limitations, and analysis notebook; round-trip inspect the actual ZIP.
- [ ] Verify CSV, Parquet, JSON, SVG, PDF, PNG, notebook, and ZIP contents against the displayed run at several redshifts and fits.
- [ ] Put fit validity, linear-versus-simulation status, finite integration bound, interpolation status, and known caveats in every relevant exported artifact.
- [ ] Check color-safe, grayscale, font embedding, equation labels, captions, and readable units at publication and classroom sizes.
- [ ] Run exported recreation code from a **clean environment** and compare its results and provenance to the original; do not call a script reproducible only because it was generated.
- [ ] Provide a human-readable data dictionary with units, shapes, coordinate conventions, redshift identity, and descriptions for every table column.
- [ ] Test that exported manifests detect changed, missing, substituted, duplicate, and path-traversal members.
- [ ] Define backup, retention, disk-quota, deletion, and portability behavior for the local vault; document recovery limits.
- [ ] Prepare a DOI-ready archival bundle only after the scientific domain, licensing, and long-term metadata are stable.

## 8. Product flow and all 20 audited workspaces — P1

- [ ] **Explore:** test first-run completion with a real solver, truthful progress/failure states, prediction before calculation, staged causal reveal, and no jargon barrier.
- [ ] **Dashboard:** show one primary scientific takeaway, active-run identity, validity and uncertainty beside results; test empty, invalid, and dense states.
- [ ] **Graph Studio:** test module selection, axes, units, redshift and fit controls, transcript parity, exports, and useful default layouts on large/small screens.
- [ ] **Compare Lab:** test exact matched-run rules, baseline changes, ratios with zero denominators, redshift mismatch, units, and excluded confounded comparisons.
- [ ] **Sensitivity Explorer:** verify finite-difference labels, one-change requirement, bounded lookup, zero-baseline behavior, and no invented posterior or threshold.
- [ ] **Structure Field:** validate shared-phase RMS numerically; keep linear-field labels and illustrative watermarks on screen and in downloads.
- [ ] **Fit + Window Atlas:** make fit family, mass definition, smoothing window, source, calibration bounds, and unsupported combinations understandable without implying interchangeability.
- [ ] **Design Experiment:** test one-change plans, held-fixed evidence, pre-run hypothesis, staging, cancellation, and changed-baseline recovery.
- [ ] **Benchmark Lab:** separate internal integration checks from external-reference agreement; show exact case version, tolerance, discrepancy, and failure.
- [ ] **Performance Lab:** record environment, warm-up, timing distribution, allocation, and reproducibility; never label a fast run scientifically valid.
- [ ] **Convergence Lab:** show what numerical choices were varied and what remains untested; require follow-up runs for a convergence claim.
- [ ] **Evolution Studio:** verify scrubber, fixed axes, contact sheet, table, video, exact/interpolated frame labels, and scientific model wording.
- [ ] **Campaign Lab:** test LHS/Sobol/grid membership, durable pause/resume/retry/cancel, bounded workers, failed members, provenance, and actual solver runs.
- [ ] **Simulation Lab:** keep planning distinct from execution; test hardware doctor, IC acceptance, imports, FOF, HMF comparison, and fail-closed absence of a validated GADGET image.
- [ ] **Teaching Lab:** test all five modules with students, local section bundles, self-directed prompts, accessible exercises, fresh-answer export, and no grading or collection path.
- [ ] **Learn the Pipeline:** test seven-step continuity, the pre-run stop, links to actual saved evidence, and misconceptions at every step.
- [ ] **Notebook:** test question, prediction, annotation, citation, artifact, lineage, conclusion, caveat, and export persistence through restart/import.
- [ ] **Runs + Export:** test load, rename, duplicate, baseline, trash, import/export, collision handling, corrupted files, and fresh-session recovery.
- [ ] **Diagnostics:** test actionable solver/runtime/range errors, preserved context, and a real local smoke run.
- [ ] **Known Limitations:** version and link each limitation to affected screens, mitigation, owner, and closure evidence; verify downloads match the current registry.
- [ ] Verify the simple Explore / Compare / Research navigation plus teaching handoff with keyboard and browser back/refresh. Do not reintroduce a “Go to…” control without a demonstrated user need.
- [ ] Conduct end-to-end browser tests on representative desktop, laptop, tablet, and narrow viewport sizes; cover long labels, large data, no data, solver failure, and interrupted sessions.

## 9. Teaching effectiveness, accessibility, and visual design — P1

- [ ] Define measurable learning objectives for each module, including prediction, causal explanation, units, limitations, and evidence-based conclusion.
- [ ] Observe real students completing each lab without help; record task completion, misconceptions, time, and where instructions fail.
- [ ] Compare learning before/after or against a reasonable alternative using consented, privacy-conscious evaluation; do not infer effectiveness from attractive materials.
- [ ] Ask instructors to run the local materials in an actual class and review preparation time, pacing, projector use, and the quality of student exports.
- [ ] Keep instructor guides as discussion aids; do not add grades, hidden student records, or monitoring to satisfy old requirements.
- [ ] Audit all science copy for “linear field = simulated universe,” “(sigma(M)) = observed catalogue,” universal HMF-fit claims, EDE-as-amplitude-only claims, and (Delta^2=1) collapse language.
- [ ] Test every interactive flow by keyboard alone, including focus order, visible focus, form errors, chart alternatives, and export controls.
- [ ] Commission screen-reader testing for navigation, tables, plots, tabs, equations, status messages, and downloads; fix issues found.
- [ ] Measure text and chart contrast in light, dark, high-contrast, and grayscale modes; verify reduced motion and no information conveyed by color alone.
- [ ] Review responsive layouts and touch targets on real devices, not only synthetic screenshots.
- [ ] Establish a visual system for typography, spacing, baseline/candidate colors, warnings, invalid states, figures, and exported media.
- [ ] Run a design critique and a novice usability study on the first 90 seconds; use findings to remove clutter and improve the causal story.
- [ ] Test that every graph has a clear question, units, provenance, accessible table/transcript, “why it matters,” and “what may mislead” note.

## 10. Security, privacy, operations, and release — P1

- [ ] Retain the single-user local deployment boundary until authenticated multi-user isolation has been designed and reviewed.
- [ ] Threat-model archive import, HDF5 parsing, file paths, formula/script exports, untrusted run names, external solver execution, and resource exhaustion.
- [ ] Test upload size limits, decompression limits, filename/path validation, ZIP bombs, malformed metadata, and denial-of-service cases.
- [ ] Verify secrets and local data never enter logs, telemetry, screenshots, exported bundles, or the source distribution unintentionally.
- [ ] Keep telemetry opt-in and local/privacy-preserving; publish exactly what is collected and how to turn it off.
- [ ] Review storage permissions, backup/restore, deletion, retention, and multi-user host behavior before any hosted deployment.
- [ ] Resolve code, documentation, generated-figure, font, solver, and dataset ownership; choose a license only with rights-holder approval.
- [ ] Run dependency and container vulnerability scans, image provenance checks, and a clean build on a supported platform.
- [ ] Run a clean-install release rehearsal from documented instructions, then solver smoke test, exports, import, teaching flow, and failure recovery.
- [ ] Publish a tagged release with change log, exact dependency/solver revisions, image digest, tests, scientific benchmark residuals, limitations, and migration instructions.
- [ ] Make public claims no stronger than the released evidence. Recheck README, screenshots, demos, portfolio descriptions, and downloadable reports against the final acceptance record.

## 11. External review and honest score evidence — P1/P2

- [ ] Obtain an independent physicist/cosmologist review of the core equations, units, calibration contracts, and displayed claims.
- [ ] Obtain independent N-body/IC review and an established-generator comparison before raising the simulation-readiness verdict.
- [ ] Have a researcher complete a real exploratory workflow and independently reproduce a published-style figure or table.
- [ ] Have an instructor and students use a complete lab and report observed learning and usability problems.
- [ ] Obtain an accessibility review by someone who uses assistive technology, plus a design review by someone outside the project.
- [ ] Publish issue-by-issue responses to outside findings, including unresolved disagreements and rejected suggestions.
- [ ] Repeat the original review against a frozen release candidate using the same rubric and **new evidence**; let the reviewer, not the project, assign new numbers.
- [ ] Treat a claimed 100 in publication readiness as bounded to a specific supported analysis and validation domain, never as a blanket certification of every cosmology or simulation workflow.

## 12. Portfolio scores that HaloForge alone cannot earn — separate track

- [ ] **Quant research:** build a separate, small rigorous market-research project covering time-series leakage, nonstationarity, execution costs, microstructure, dependent-data validation, risk limits, and live-versus-backtest drift. Have a domain reviewer examine it. Do not bolt finance features onto HaloForge merely to raise a portfolio score.
- [ ] **ML/training:** build a separate serious PyTorch/JAX project with reproducible training, evaluation methodology, ablations, data provenance, GPU profiling, failure analysis, and ideally distributed or large-scale systems evidence. Have an ML engineer review it.
- [ ] **Research-engineering portfolio:** write concise case studies explaining the two IC defects, the corrected derivation, test design, remaining validation gap, and how outside feedback changed the system.
- [ ] **Ownership evidence:** retain commit history, design notes, tradeoffs, review responses, and demonstrations where the author can derive the formulas and explain why each scientific boundary exists.

## Release verdict template

The checklist is not “all done” when every box has code. It is done for an agreed release scope when:

1. Every applicable item has linked evidence and a reviewer; removed/deferred items have a visible rationale.
2. All P0 scientific blockers pass independent checks in the supported domain.
3. All 20 workspaces pass their stated behavior, accessibility, and failure-path checks.
4. The compiled solver and release container pass the frozen regression suite.
5. External science, teaching, accessibility, and design reviews have been answered.
6. The release is legally distributable, reproducible from a clean environment, and candid about remaining limits.

The reviewer’s original scores remain historical baselines: core physics 91, numerical engineering 94, HMF calibration 82, N-body/IC 58, teaching 90, publication readiness 77, architecture 81, reproducibility 95, portfolio 89, quant 79, research engineering 86, and ML/training 69. Re-score only after the release evidence above exists.
