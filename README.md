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

HaloForge is an interactive, research-oriented path from primordial parameters to the linear matter power spectrum, mass variance, and halo mass function. Every cosmology run uses the real `classy.Class()` binding compiled from PoulinV/AxiCLASS revision `1b0a585f86a3dce6babd66e486535368b2799ec7`. There is no toy-spectrum fallback.

## Current safety and reproducibility posture

- **Single-user local app only.** Runs, drafts, cache, and exports live outside the cloned source tree by default. Docker Compose uses a private named Docker volume. A public/hosted deployment fails closed before writing any data, because the current filesystem vault is not an authenticated per-user store.
- **No accidental repository data.** The source tree excludes personal experiments, exports, cache, local environments, and secrets from Git and Docker contexts.
- **Reproducible bundles.** Every newly computed run records a versioned provenance envelope and reproducibility hash. Its export bundle includes `provenance.json` and a manifest containing a hash and byte count for every exported artifact.
- **Slider movement cannot launch expensive work.** Cosmology controls live inside a form. Values are staged in the browser and only run when **Run & auto-save** is pressed.
- **CLASS is crash-isolated.** Each CLASS/AxiCLASS solve runs in a dedicated subprocess. A native backend exit or segmentation fault ends the worker rather than the Streamlit app, and the previous completed run remains untouched.
- **Writes are atomic.** Run metadata, compressed arrays, cache files, and state pointers are written to temporary files and atomically replaced, preventing half-written files after interruption.
- **Storage decisions are explicit.** The durable-run/export design and its scientific-scope boundary are recorded in [ADR 0001](docs/adr/0001-run-export-integrity-and-provenance.md).
- **HMF conventions fail closed.** Halo mass-definition and calibration-contract decisions are recorded in [ADR 0002](docs/adr/0002-halo-mass-definition-and-calibration-contracts.md); a plotted extrapolation is not a calibration claim.
- **σ(M) integration is deterministic.** HaloForge integrates the complete sampled CLASS grid in `ln k` with a vectorized Simpson rule. This avoids repeated adaptive-quadrature interpolation and subdivision failures.
- **Comparison baselines are matched correctly.** Ratio and residual plots compare each curve against the baseline curve with the same smoothing window and HMF fit.
- **Structure fields preserve amplitude.** The app generates a periodic 3D Gaussian linear-density realization with identical Fourier phases for every run. Panels share one baseline normalization, so Aₛ and growth differences are not normalized away.
- **Graph layout is responsive.** Titles, legends, axis labels, captions, and margins are separated. Graph Studio and Compare Lab include per-chart axis controls, while Plotly zoom and high-resolution PNG export remain available.
- **Evidence stays explicit.** See the [repository review](docs/REPOSITORY-REVIEW.md), [changelog](CHANGELOG.md), [implementation status](docs/IMPLEMENTATION-STATUS.md), [known limitations](docs/KNOWN-LIMITATIONS.md), and [release policy](docs/RELEASE-POLICY.md) for completed local work versus evidence still required before release.

## Scientific conventions

- AxiCLASS returns linear `P(k,z)` in `Mpc³` for `k` in `Mpc⁻¹`.
- Halo masses displayed as `M_h` are numerical values in `h⁻¹ M☉`; physical mass is `M=M_h/h` in `M☉`.
- `ρcrit,0 = 2.775×10¹¹ h² M☉ Mpc⁻³` and `ρm,0=Ωm ρcrit,0`.
- The default real-space top-hat has `W(y)=3(sin y-y cos y)/y³`, evaluated with a small-y series.
- `σ²(R,z)=(2π²)⁻¹∫k²P(k,z)W²(kR)dk`, evaluated over the finite sampled CLASS k range using Simpson integration in `ln k`.
- `dn/dlnM=(ρm,0/M) f(σ) |dlnσ/dlnM|`; the result is reported in `h³ Mpc⁻³` against `M_h`.
- Every selected redshift is sampled directly from AxiCLASS. EDE runs are not scaled using a ΛCDM growth approximation.
- The structure-field page is a linear Gaussian realization, not an N-body simulation or literal halo catalogue. Its RMS and correlation table preserve the computed relative amplitudes.

Empirical HMF fits are not universally interchangeable. Match each fit's halo definition, overdensity convention, calibration cosmology, redshift, and validity range before using it in a paper.

## Run locally with Docker

1. Install and open Docker Desktop.
2. Open a terminal in this folder.
3. Run:

```bash
docker compose up --build -d
```

4. Open `http://localhost:7860`.

Your durable local results are stored in the `haloforge_data` Docker volume, not the repository. To see volumes, use Docker Desktop; do not delete `haloforge_data` when updating the app. Exports can always be downloaded from the Run Vault.

To stop the app without deleting results:

```bash
docker compose down
```

To rebuild after pulling a new GitHub version while retaining results:

```bash
git pull --ff-only
docker compose up --build -d
```

## One-command GitHub start on a Mac

```bash
open -a Docker && until docker info >/dev/null 2>&1; do sleep 2; done; if [ ! -d "$HOME/haloforge/.git" ]; then git clone https://github.com/Blake-Bird/haloforge.git "$HOME/haloforge"; fi; cd "$HOME/haloforge" && git pull --ff-only && docker compose up --build -d && open http://localhost:7860
```

## Local non-Docker development

Use Python 3.11. Build the exact AxiCLASS revision named in the Dockerfile before starting the app. Its Python binding imports Cython during metadata generation but does not declare Cython as an isolated build dependency, so install it with build isolation disabled:

```bash
git clone https://github.com/PoulinV/AxiCLASS.git
cd AxiCLASS
git checkout 1b0a585f86a3dce6babd66e486535368b2799ec7
python -m pip install "Cython==0.29.36" "numpy==1.26.4"
make class libclass.a -j2
python -m pip install --no-build-isolation .
```

Then install `requirements.txt` in the same environment and run `streamlit run app.py`. The binding must be importable as `classy`; HaloForge will show an actionable diagnostic and will not fabricate a spectrum if it is unavailable.

Without `HALOFORGE_DATA_DIR`, the local app uses your operating system’s per-user application-data location (for example, `~/Library/Application Support/HaloForge` on macOS), never `./data` in the cloned repository. Set `HALOFORGE_DATA_DIR` only to a private directory you control.

## Hosted deployments

Do not deploy this build as a shared public Streamlit service. Setting `HALOFORGE_DEPLOYMENT=hosted` intentionally stops the app before it reads or writes experiments: a server filesystem is shared by visitors and is not browser-private storage. A hosted release requires a separate authenticated per-user storage adapter, access control, retention policy, quota enforcement, and security review. Until then, use the local application and explicit export/import bundles.

## License and release boundary

No open-source license has been selected for this repository yet. Until the rights holder adds one, the source is shared for review only; do not assume permission to reuse, redistribute, or host it. Before a public GitHub release, choose a license, confirm third-party attribution requirements, and replace this notice with the selected license text.

## Maintainer quality gates

Before opening a pull request or preparing a release, run:

```bash
ruff format --check app.py config content engine state tests
ruff check app.py config content engine state tests
pytest -q
docker compose config
```

The CI workflow also runs these checks, a dependency audit, and an image build. A real Docker/AxiCLASS smoke test remains required on a machine with a running Docker daemon.

## Publication checklist

- Record the AxiCLASS commit written to `/opt/AXICLASS_COMMIT` during the image build.
- Export parameters, exact CLASS settings, sampled spectra, σ(M,z), fitting-function name, `δc`, halo overdensity, window, and numerical ranges.
- Demonstrate k-range, k-sampling, and mass-grid convergence.
- Compare pipeline σ₈ against AxiCLASS and explain the finite k integration range.
- Confirm that a selected empirical HMF fit is used inside its published mass, redshift, halo-definition, and cosmology calibration range.
- Cite CLASS, AxiCLASS/EDE papers, and the selected HMF calibration paper.
