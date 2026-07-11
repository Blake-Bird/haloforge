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

## What changed in this stable release

- **Runs auto-save locally.** Every completed run is written to `./data/saved_runs`, with exports in `./data/exports`. Docker Compose mounts `./data` from the host, so browser refreshes, browser closes, Streamlit restarts, container restarts, rebuilds, and Git pulls do not erase completed runs.
- **The latest run restores automatically.** HaloForge records the active run in `./data/state/last_run.json` and reloads it when a new browser session opens.
- **Slider movement cannot launch expensive work.** Cosmology controls live inside a form. Values are staged in the browser and only run when **Run & auto-save** is pressed.
- **CLASS is crash-isolated.** Each CLASS/AxiCLASS solve runs in a dedicated subprocess. A native backend exit or segmentation fault ends the worker rather than the Streamlit app, and the previous completed run remains untouched.
- **Writes are atomic.** Run metadata, compressed arrays, cache files, and state pointers are written to temporary files and atomically replaced, preventing half-written files after interruption.
- **σ(M) integration is deterministic.** HaloForge integrates the complete sampled CLASS grid in `ln k` with a vectorized Simpson rule. This avoids repeated adaptive-quadrature interpolation and subdivision failures.
- **Comparison baselines are matched correctly.** Ratio and residual plots compare each curve against the baseline curve with the same smoothing window and HMF fit.
- **Structure fields preserve amplitude.** The app generates a periodic 3D Gaussian linear-density realization with identical Fourier phases for every run. Panels share one baseline normalization, so Aₛ and growth differences are not normalized away.
- **Graph layout is responsive.** Titles, legends, axis labels, captions, and margins are separated. Graph Studio and Compare Lab include per-chart axis controls, while Plotly zoom and high-resolution PNG export remain available.

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

Your durable local results are stored in the repository’s `data` folder. Do not delete that folder when updating the code.

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

Use Python 3.11. Compile AxiCLASS first, ensure its generated `python/classy*.so` is on `PYTHONPATH`, install `requirements.txt`, and run `streamlit run app.py`.

## Publication checklist

- Record the AxiCLASS commit written to `/opt/AXICLASS_COMMIT` during the image build.
- Export parameters, exact CLASS settings, sampled spectra, σ(M,z), fitting-function name, `δc`, halo overdensity, window, and numerical ranges.
- Demonstrate k-range, k-sampling, and mass-grid convergence.
- Compare pipeline σ₈ against AxiCLASS and explain the finite k integration range.
- Confirm that a selected empirical HMF fit is used inside its published mass, redshift, halo-definition, and cosmology calibration range.
- Cite CLASS, AxiCLASS/EDE papers, and the selected HMF calibration paper.
