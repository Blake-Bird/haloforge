# N-body platform: download-to-working release checklist

Status: **not yet an executable GADGET-4 platform** (2026-09-22). This is the implementation and acceptance plan for a person who clones HaloForge and expects to run a dark-matter-only periodic ΛCDM simulation, view its particles, find halos with GADGET-4 FoF/Subfind and Rockstar, and compare scientifically compatible abundance estimates. An EDE N-body claim is a separate research/validation milestone, not part of the first working release.

Legend: `[x]` exists in the current source and has a local synthetic/unit test; `[ ]` incomplete, absent, or not demonstrated on a clean machine. A checked item does **not** certify physical accuracy. For each completed item, add a CI job or versioned evidence artifact and test date.

## 0. Present gaps and release boundary

- [x] A box/resolution planner, 1LPT IC writer, Config.sh/parameter/output-time generators, strict single-file PartType1 reader, in-app periodic FoF, saved-catalogue provenance, HMF comparison guard, recorded-particle cube/density views, and recorded-frame GIF/MP4 export exist.
- [ ] A fresh clone has a documented, one-command path to a *real* GADGET-4 executable. The current `Dockerfile` builds AxiCLASS, not GADGET-4 or Rockstar. It now copies `ui/`, checks its import, and pins the previously omitted `h5py` dependency. On 2026-09-22, the Linux arm64 test and production images built from this checkout; focused container tests passed and the non-root production health endpoint returned `ok`. This does not establish a clean-clone GADGET run or other architecture support.
- [ ] Provide an in-app run controller. Today the Simulation Lab prepares text files and imports external snapshots; it does not launch, queue, monitor, stop, or resume GADGET-4.
- [ ] Support native GADGET-4 multi-file snapshots and FoF/Subfind catalogues. The snapshot reader now assembles declared HDF5 PartType1 shards and rejects partial/inconsistent sets in synthetic tests; native output and group catalogues remain untested.
- [ ] Detect, build/run, import, and validate Rockstar. No working Rockstar adapter or FoF-versus-Rockstar screen exists.
- [ ] Publish a clearly scoped support matrix: macOS arm64/x86_64, Linux x86_64/arm64 where tested, Windows via WSL2, local desktop versus server/HPC, maximum tested particle count, and ΛCDM DM-only versus unsupported EDE/hydro modes.

## 1. Clean-machine installation and dependency gate (P0)

- [ ] Pin the GADGET-4 source commit, build configuration, compiler/MPI/HDF5/GSL/FFTW dependencies, and the resulting image **digest**, not merely a tag. Record upstream license and citation.
- [ ] Add a reproducible GADGET-4 build stage or separately pinned simulation image. Build for each supported CPU architecture; prove the executable starts and runs an official example or declared acceptance fixture.
- [ ] Fix HaloForge's production image contents, including `ui/`, tests of Python imports, non-root permissions, writable run volume, and local-only Docker/daemon integration. Do not mount an unrestricted host Docker socket into a publicly reachable app without a security design.
- [ ] Decide one supported execution path for first release: local container with a tightly scoped runner, or native executable. Make the installer/doctor test the *same* path used by the Start button.
- [ ] Provide `install`, `doctor`, `smoke`, and `uninstall` instructions for macOS, Linux, and WSL2, including disk/RAM/CPU requirements, Apple Silicon caveats, and failed-build troubleshooting.
- [ ] Doctor verifies running container daemon, immutable image digest, executable, MPI ranks, HDF5 support, writable/adequate disk, RAM, architecture compatibility, and a real acceptance result. Its “ready” state must not be driven by an editable JSON manifest alone.
- [ ] CI builds a new image from a clean checkout, launches the app, runs a tiny solver calculation, starts and finishes a tiny GADGET simulation, and reads its outputs. Keep any larger physical benchmark as a separate scheduled job.

## 2. Physical model and initial conditions (P0)

- [ ] Choose and display an initial supported cosmology: flat ΛCDM, collisionless PartType1 only, specified transfer function/gauge, and explicit redshift range. Reject EDE and hydro in the executable path until independently validated.
- [ ] Bind each simulation draft to one immutable, integrity-checked CLASS/AxiCLASS result; retain exact `P(k,z_start)`, Ωm, Ωb, ΩΛ, h, A_s, n_s, normalization, solver version, and its reproducibility hash.
- [ ] Validate box `L`, `N³`, Nyquist/fundamental k, starting redshift, PM mesh, softening, mass resolution, memory/disk estimate, output schedule, and final redshift before writing any run. Offer a tiny safe preset and measured larger presets.
- [ ] Make the IC order explicit. The current UI generates **1LPT** files and `generate_config_sh()` now defaults to 1LPT and rejects `SECOND_ORDER_LPT_ICS`, which upstream says requires specially constructed Jenkins ICs. Validate the entire generated configuration with a real GADGET build before closing this item.
- [ ] Independently verify Mpc ↔ Mpc/h, k and P(k) conversion, mass `10^10 M☉/h` storage, GADGET velocity convention, scale factor, particle IDs, periodic coordinates, header particle counts, and mean density with actual GADGET reads, not solely round-trips through HaloForge.
- [ ] Compare the generated IC power, displacement and velocities to an independent IC generator across multiple seeds, h values, grids, boxes and epochs, with published tolerances. Prove no hidden k extrapolation.
- [ ] Record the seed, phase convention, generator revision, input spectrum hash, all particle-array hashes, and an IC acceptance report in the run folder.

## 3. Build/configuration and durable run folder (P0)

- [ ] Compile and validate every emitted `Config.sh` option against the pinned GADGET-4 revision. Keep FoF/Subfind compile-time options separate from runtime parameters; reject unsupported combinations.
- [ ] Parse/validate the generated parameter file with the actual GADGET-4 executable before allowing a long run. Verify filename stems, units, time bounds, softening, precision, MPI/I/O settings, and output-times file.
- [ ] Make output folder and names configurable but safe; reserve a unique run ID and create `inputs/`, `config/`, `logs/`, `restarts/`, `snapshots/`, `catalogues/`, `figures/`, `exports/` with a versioned manifest and no overwrite of another run.
- [ ] Save effective configuration and executable/image digest, host/CPU/MPI/HDF5 versions, exact source-run hash, cosmology, `L` and units, `N³`, particle mass, start/final scale factors, snapshot schedule, seed, and hashes of every generated input.
- [ ] Estimate disk and memory from the chosen `N³` and number of output files. Require free-space headroom and set resource limits; do not offer an unbounded “large box” button on a laptop.
- [ ] Preserve GADGET's own HDF5 `Header`, `Parameters`, and `Config` metadata in the provenance chain; compare it against HaloForge's effective run manifest.

## 4. Execution controller and recovery (P0)

- [ ] Implement a bounded local job queue with states `draft → validated → queued → starting → running → stopping/checkpointing → completed/failed/interrupted`; persist transitions across app restarts.
- [ ] Launch GADGET as an argument vector in an isolated per-run working directory, with explicit MPI ranks, file mounts, CPU/RAM limits, and no shell interpolation of user paths.
- [ ] Stream stdout/stderr and GADGET progress files into a log panel; show elapsed time, current simulation time/redshift, completed snapshots, live disk use, and a measured remaining-time range (never a fake precise ETA).
- [ ] Implement cancel with graceful checkpoint request, timeout/escalation, and status reconciliation. Never label a terminated/incomplete run as scientifically completed.
- [ ] Implement resume from GADGET restart files and from a snapshot as separate workflows. Restart-file resume must preserve the required MPI rank count/build compatibility; snapshot resume must avoid output-name collisions. Test interrupted, missing, stale, and corrupt checkpoints.
- [ ] Verify exit code, terminal log marker, expected output times, and every snapshot/header before marking success. Detect partial/missing files and clean up only run-owned temporary files.
- [ ] Permit users to reopen a run folder after app restart and continue analysis without recomputing. Support export/import of a portable run bundle with a manifest verifier.

## 5. Real snapshot and density explorer (P0/P1)

- [ ] Support single-file and split HDF5 snapshots (`snap_###.N.hdf5`), header totals, duplicate IDs, type selection, unit conventions, variable masses where applicable, and partial-file detection. The current reader covers equal-mass PartType1 shards with synthetic tests, including cross-shard duplicate IDs and byte hashes; native fixture, variable-mass support, and large-file streaming remain.
- [ ] Verify each snapshot belongs to the exact run: hashes, embedded Parameters/Config, `L`, cosmology, scale factor/redshift, particle count/mass, units, and planned output schedule.
- [x] Render a bounded sample of *recorded* particle coordinates, a full-box projected density grid, physical axes, and file-derived box size; do not call the initial lattice an evolved field.
- [ ] Add selectable orthogonal slices, adjustable thickness, smoothing/density estimator, color scale, periodic boundaries, camera reset, and a clear sampled-versus-all-particles indicator; test mobile/keyboard and large-file performance.
- [ ] Render a movie from all validated real snapshots with a fixed physical/color scale, exact frame redshifts, playback/scrubbing, and export manifest. Current movie path is limited to 2–12 single-file snapshots; expand it safely for production schedules and test WebM as well as GIF/MP4.
- [ ] Provide catalogue history and evolution plots across outputs (counts and abundances versus z; growth only with a validated tree or explicitly labeled spatial matching).

## 6. GADGET-4 FoF/Subfind and Rockstar (P0)

- [ ] Offer the pinned GADGET-4 built-in FoF (and optional Subfind) at planned snapshots or postprocessing. Import **native** HDF5 group catalogues, including multi-file groups and offsets; preserve original column names/units and keep host groups separate from subhalos.
- [ ] Keep the in-app periodic cKDTree FoF as a small-box cross-check only. Add limits/streaming or fail-closed warnings for large `N³`; benchmark boundary-spanning groups against GADGET FoF.
- [ ] Pin/build Rockstar with HDF5 support and a versioned config. For GADGET HDF5 input, validate the Rockstar fork's **AREPO** reader path and length/mass/velocity conversions rather than assuming its binary `GADGET` reader works for HDF5. Test one-snapshot single-worker and periodic parallel modes separately.
- [ ] Detect Rockstar executable/configuration, run it from the same validated snapshots, capture server/worker logs and outputs, support restart, and fail if no catalogue is produced. Keep its resources and output folder isolated per simulation.
- [ ] Import Rockstar ASCII/list (and any binary format explicitly supported) using the file's declared columns/units, scale factor, cosmology, box size, parent/host identity, mass definition, unbinding and `STRICT_SO_MASSES` setting. Reject missing/contradictory metadata; keep exact catalogue hashes.
- [ ] Compare both finders on **the same snapshot**, same periodic box, redshift, particle threshold and host/subhalo selection. Show positional crossmatches, unmatched halos, mass-definition labels, counts, resolution cuts, and uncertainty. Do not compare FoF `b=0.2` group masses numerically with Rockstar `Mvir`/`M200c` as if identical.
- [ ] Test adversarial fixtures: group crossing a periodic face, close pairs, zero halos, subhalos, split files, duplicate IDs, unit mismatch, wrong z, swapped snapshot, missing parent IDs, and multiple plausible matches.

## 7. HMF and cosmology-matched analysis (P0)

- [ ] Enforce one immutable linear-run ↔ IC ↔ GADGET snapshots ↔ both catalogues identity. No “nearest z” substitution, editable draft cosmology, or mismatched box/mass units in a scientific comparison.
- [x] The in-app FoF-versus-HMF path requires a verified snapshot sidecar and exact stored σ(M,z), and restricts fits to a FoF-compatible mass definition. Extend these gates to native GADGET FoF and Rockstar.
- [ ] For each finder compute differential/cumulative abundance with explicit `L³`, bin edges, independent counts, Poisson uncertainty, finite-volume/cosmic-variance estimate, particle-mass resolution, incompleteness, fit calibration range, and redshift interpolation status.
- [ ] Use SO-calibrated HMF fits only with matching SO mass definitions and explicitly validated Rockstar strict-SO/catalogue settings; document whether subhalos are excluded. If no compatible fit exists, show the finder comparison without an invalid analytic overlay.
- [ ] Add three coordinated dashboards: GADGET FoF versus in-app FoF, GADGET FoF versus Rockstar spatial/selection comparison, and each *compatible* catalogue versus HMF; all include units, matching rules, provenance, and limitations.
- [ ] Quantify numerical convergence across box size, `N³`, start redshift, softening, time stepping and force accuracy. Separate “run completed” from “result converged/calibrated/publication-suitable.”

## 8. Downloadable product and release acceptance (P0/P1)

- [ ] A new user can clone the repo, run one documented setup command, read a diagnostic that identifies missing prerequisites, launch the tiny ΛCDM example from the UI, watch it finish, view the 3D particles/density/movie, run both finders, see safe HMF comparisons, and export a self-contained run **without manually editing paths or inventing metadata**.
- [ ] Run that journey on each claimed platform using a clean user account and blank data directory. Publish elapsed time, peak memory, disk use, exact commands, dependency versions, image digests and fixture hashes.
- [ ] CI covers config validation, IC/header units, official GADGET smoke run, restart/resume, split snapshot/group import, Rockstar HDF5 conversion, finder match, mass-definition refusal, HMF gate, every export, and browser workflow; destructive/corrupt inputs must fail closed.
- [ ] Review long-run security: allowed directories, path traversal, job limits, process ownership, log redaction, disk quota, cancellation, container isolation, and privacy for multi-user/server mode. Keep local-only mode explicit until users have isolated storage and compute.
- [ ] Publish a versioned example run and independent validation report; update README, installer, screenshots, capability matrix, known limitations, and citations to distinguish built/tested features from planned ones.
- [ ] EDE promotion gate, **separate from ΛCDM readiness**: specify a GADGET-compatible EDE background/perturbation dynamics contract, implement it in a pinned fork or validated interface, compare its expansion/linear response to AxiCLASS, run independent reference simulations, and repeat finder/HMF convergence tests. Until then, the Start button must reject EDE.

## Definition of “fully working”

The N-body feature is not complete when a config is printable, a synthetic HDF5 file passes tests, or a plot renders. It is complete for the declared ΛCDM DM-only scope only when the clean-machine journey in section 8 passes with *actual* GADGET-4 and Rockstar outputs on every advertised platform, the exported files preserve the true box/cosmology/units and file hashes, and the comparison UI refuses invalid mass definitions or mismatched runs. Independent scientific validation is an additional claim with its own evidence.

## Upstream references used for this checklist

- [GADGET-4 build, run and restart guide](https://wwwmpa.mpa-garching.mpg.de/gadget4/02_running/)
- [GADGET-4 runtime parameters and split HDF5 output](https://wwwmpa.mpa-garching.mpg.de/gadget4/05_parameterfile/)
- [GADGET-4 snapshot metadata and units](https://wwwmpa.mpa-garching.mpg.de/gadget4/06_snapshotformat/)
- [GADGET-4 FoF/Subfind catalogue format](https://wwwmpa.mpa-garching.mpg.de/gadget4/08_groupfiles/)
- [Rockstar setup, HDF5 input, outputs and mass-definition cautions](https://github.com/eelregit/rockstar)
