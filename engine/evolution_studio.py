"""Redshift evolution movie studio engine: z=20 -> 0 exact frame calculations.

Provides synchronized observables across scale-factor-aware frame sequences:
matter power P(k,z), dimensionless Delta^2(k,z), growth D(z), variance sigma(M,z),
and differential/cumulative halo mass functions, with fixed-axes physical animation
and accessible static contact sheets.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import plotly.graph_objects as go

from engine.fitting_functions import fitting_values
from engine.sigma import dlog_sigma_dlog_M
from engine.windows import window_squared


@dataclass(frozen=True)
class EvolutionMilestone:
    """A physical milestone epoch in cosmic history."""

    name: str
    redshift: float
    description: str
    active_in_frame: bool = False


def cosmic_time_gyr(z: float, params: dict) -> float:
    """Approximate cosmic age in Gyr at redshift z for standard or EDE cosmology."""
    h = float(params.get("H0", 67.36)) / 100.0
    omega_m = float(params.get("Omega_m", 0.315))
    omega_l = 1.0 - omega_m - float(params.get("Omega_k", 0.0))
    # Hubble time in Gyr: 1 / H0 ~ 9.778 / h Gyr
    hubble_time = 9.778 / h
    # Flat Lambda-CDM analytic age formula:
    # t(a) = (2 / (3 * H0 * sqrt(Omega_L))) * asinh(sqrt(Omega_L / Omega_m) * a^(3/2))
    a = 1.0 / (1.0 + float(z))
    if omega_l > 0 and omega_m > 0:
        arg = np.sqrt(omega_l / omega_m) * (a**1.5)
        t = (2.0 / (3.0 * np.sqrt(omega_l))) * np.arcsinh(arg) * hubble_time
    else:
        # Matter-dominated Einstein-de Sitter approximation
        t = (2.0 / 3.0) * (a**1.5) * hubble_time
    return max(float(t), 1e-4)


def nonlinear_scale_knl(k: np.ndarray, delta2: np.ndarray) -> float | None:
    """Find the wavenumber k where dimensionless variance Delta^2(k) reaches unity."""
    mask = delta2 >= 1.0
    if np.any(mask):
        idx = int(np.argmax(mask))
        if idx > 0:
            k0, k1 = k[idx - 1], k[idx]
            d0, d1 = delta2[idx - 1], delta2[idx]
            if d1 > d0:
                frac = (1.0 - d0) / (d1 - d0)
                return float(k0 + frac * (k1 - k0))
        return float(k[idx])
    return None


def calculate_evolution_frames(
    power_k: np.ndarray,
    power_p_z0: np.ndarray,
    redshifts: np.ndarray,
    growth_factors: np.ndarray | None,
    params: dict,
    *,
    mass_min_exp: float = 8.0,
    mass_max_exp: float = 16.0,
    mass_points: int = 100,
    fitting: str = "Tinker 2008",
) -> list[dict[str, Any]]:
    """Compute physical observable frames across an exact redshift sequence.

    Uses linear growth scaling D(z) for P(k,z) when pre-solved Boltzmann grids
    do not contain arbitrary intermediate frames, or uses explicit Boltzmann inputs.
    """
    k = np.asarray(power_k, dtype=float)
    p0 = np.asarray(power_p_z0, dtype=float)
    zs = np.asarray(redshifts, dtype=float)
    h = float(params.get("H0", 67.36)) / 100.0
    omega_m0 = float(params.get("Omega_m", 0.315))
    rho_crit_0 = 2.775e11 * (h**2)  # Msun / Mpc^3
    rho0 = omega_m0 * rho_crit_0

    # Mass coordinates
    M_h = np.logspace(mass_min_exp, mass_max_exp, mass_points)
    M = M_h / h

    # Precalculate growth factors if not supplied
    if growth_factors is None or len(growth_factors) != zs.size:
        # Standard LambdaCDM growth factor approximation (Carrol, Press & Turner 1992)
        growth = []
        omega_l0 = 1.0 - omega_m0 - float(params.get("Omega_k", 0.0))
        for z_val in zs:
            a_val = 1.0 / (1.0 + z_val)
            ez2 = omega_m0 * (1.0 + z_val) ** 3 + omega_l0
            om_z = omega_m0 * (1.0 + z_val) ** 3 / ez2
            ol_z = omega_l0 / ez2
            d_z = (
                (5.0 / 2.0)
                * om_z
                * a_val
                / (
                    (om_z ** (4.0 / 7.0))
                    - ol_z
                    + (1.0 + om_z / 2.0) * (1.0 + ol_z / 70.0)
                )
            )
            growth.append(d_z)
        growth = np.asarray(growth, dtype=float)
        # Normalize to D(z=0) = 1.0
        if growth[-1] > 0:
            growth /= growth[-1]
    else:
        growth = np.asarray(growth_factors, dtype=float)
        if growth[-1] > 0:
            growth /= growth[-1]

    # EDE active milestone if configured
    ede_enabled = bool(params.get("enable_ede", False))
    log10_ac = float(params.get("log10_a_c", -3.5))
    z_c = float(10.0 ** (-log10_ac) - 1.0) if ede_enabled else None

    frames = []
    for idx, z_val in enumerate(zs):
        scale_factor = 1.0 / (1.0 + float(z_val))
        d_val = float(growth[idx])
        p_z = p0 * (d_val**2)
        delta2 = (k**3) * p_z / (2.0 * (np.pi**2))
        knl = nonlinear_scale_knl(k, delta2)

        # Compute sigma(M, z)
        # Using top-hat smoothing:
        R = (3.0 * M / (4.0 * np.pi * rho0)) ** (1.0 / 3.0)
        sigmas = []
        for r_val in R:
            # sigma^2 = integral dk / (2 pi^2) k^2 P(k,z) W^2(kR)
            integrand = (
                (k**2) * p_z * window_squared(k * r_val, "Top-hat") / (2.0 * np.pi**2)
            )
            var = np.trapezoid(integrand, k)
            sigmas.append(np.sqrt(max(var, 1e-30)))
        sigmas = np.asarray(sigmas, dtype=float)
        dlnsigma = dlog_sigma_dlog_M(M, sigmas)

        # HMF: dn / dln M
        delta_c = float(params.get("delta_c", 1.686))
        ez2 = omega_m0 * (1.0 + z_val) ** 3 + (1.0 - omega_m0)
        omega_m_z = omega_m0 * (1.0 + z_val) ** 3 / ez2
        f_mult = fitting_values(
            sigmas,
            delta_c,
            fitting,
            z=float(z_val),
            omega_m_z=float(omega_m_z),
            delta_halo=200.0,
            neff=-6.0 * dlnsigma - 3.0,
        )
        hmf_diff = (rho0 / M) * f_mult * np.abs(dlnsigma) / (h**3)

        # Cumulative HMF: n(>M) = integral_M^Mmax (dn/dln M') dln M'
        log_M = np.log(M)
        cum_hmf = np.zeros_like(hmf_diff)
        for m_i in range(len(M) - 1):
            cum_hmf[m_i] = np.trapezoid(hmf_diff[m_i:], log_M[m_i:])
        cum_hmf[-1] = hmf_diff[-1] * 0.01

        # Milestones
        milestones = []
        if z_c is not None:
            is_active = abs(float(z_val) - z_c) / (1.0 + z_c) < 0.25
            milestones.append(
                EvolutionMilestone(
                    name="EDE Peak Activity",
                    redshift=z_c,
                    description=f"Axion EDE energy fraction peaks around z_c={z_c:.1f} (a_c=10^{log10_ac:.1f}).",
                    active_in_frame=is_active,
                )
            )
        if knl is not None:
            milestones.append(
                EvolutionMilestone(
                    name="Nonlinear Scale Entry",
                    redshift=float(z_val),
                    description=f"Modes with k >= {knl:.3f} Mpc^-1 have collapsed into the non-linear regime (Delta^2 >= 1).",
                    active_in_frame=True,
                )
            )

        frame_data = {
            "frame_index": idx,
            "redshift": float(z_val),
            "scale_factor": scale_factor,
            "cosmic_time_gyr": cosmic_time_gyr(z_val, params),
            "growth_factor": d_val,
            "k": k.tolist(),
            "P_k": p_z.tolist(),
            "delta2": delta2.tolist(),
            "knl": knl,
            "M_h": M_h.tolist(),
            "M": M.tolist(),
            "sigma": sigmas.tolist(),
            "dlnsigma_dlnM": dlnsigma.tolist(),
            "hmf_diff": hmf_diff.tolist(),
            "hmf_cum": cum_hmf.tolist(),
            "milestones": [asdict(m) for m in milestones],
        }
        frames.append(frame_data)

    return frames


def build_evolution_figure(
    frames: list[dict[str, Any]],
    current_frame_idx: int,
    observable: str,
    *,
    baseline_frames: list[dict[str, Any]] | None = None,
    mode: str = "Overlay",
    fixed_axes: bool = True,
    theme: str = "Dark",
) -> go.Figure:
    """Construct a high-contrast physical evolution frame figure with fixed coordinate bounds."""
    if not frames or current_frame_idx >= len(frames):
        raise ValueError("Invalid frames or frame index")

    curr = frames[current_frame_idx]
    fig = go.Figure()

    # Determine axis properties and traces based on observable
    if observable == "Matter Power P(k)":
        x = np.asarray(curr["k"])
        y = np.asarray(curr["P_k"])
        x_label = "k [Mpc⁻¹]"
        y_label = "P(k,z) [Mpc³]"
        title = f"Matter Power Spectrum P(k, z={curr['redshift']:.2f})"
        is_log_x, is_log_y = True, True
        # Base trace
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                name=f"Candidate (z={curr['redshift']:.2f})",
                line=dict(color="#35d7e5", width=3.0),
                hovertemplate="k=%{x:.3e} Mpc⁻¹<br>P(k)=%{y:.3e} Mpc³<extra></extra>",
            )
        )
        if baseline_frames and current_frame_idx < len(baseline_frames):
            base = baseline_frames[current_frame_idx]
            if mode == "Ratio":
                ratio = y / np.maximum(np.asarray(base["P_k"]), 1e-30)
                fig.data[0].y = ratio
                y_label = "P(k) / P_baseline(k)"
                is_log_y = False
            else:
                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=np.asarray(base["P_k"]),
                        mode="lines",
                        name=f"ΛCDM Baseline (z={base['redshift']:.2f})",
                        line=dict(color="#ffb454", width=2.2, dash="dash"),
                        hovertemplate="k=%{x:.3e} Mpc⁻¹<br>P_base=%{y:.3e} Mpc³<extra></extra>",
                    )
                )

    elif observable == "Dimensionless Power Δ²(k)":
        x = np.asarray(curr["k"])
        y = np.asarray(curr["delta2"])
        x_label = "k [Mpc⁻¹]"
        y_label = "Δ²(k,z) = k³ P(k) / (2π²)"
        title = f"Dimensionless Variance Δ²(k, z={curr['redshift']:.2f})"
        is_log_x, is_log_y = True, True
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                name=f"Candidate (z={curr['redshift']:.2f})",
                line=dict(color="#35d7e5", width=3.0),
                hovertemplate="k=%{x:.3e} Mpc⁻¹<br>Δ²=%{y:.3e}<extra></extra>",
            )
        )
        # Linear threshold line Delta^2 = 1
        fig.add_hline(
            y=1.0,
            line_dash="dot",
            line_color="#ff718b",
            annotation_text="Linear collapse threshold (Δ² = 1)",
            annotation_position="top left",
        )

    elif observable == "Mass Variance σ(M)":
        x = np.asarray(curr["M_h"])
        y = np.asarray(curr["sigma"])
        x_label = "M [h⁻¹ M☉]"
        y_label = "σ(M, z)"
        title = f"RMS Fluctuation Strength σ(M, z={curr['redshift']:.2f})"
        is_log_x, is_log_y = True, True
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                name=f"Candidate (z={curr['redshift']:.2f})",
                line=dict(color="#45d49d", width=3.0),
                hovertemplate="M=%{x:.3e} h⁻¹ M☉<br>σ=%{y:.4f}<extra></extra>",
            )
        )
        if baseline_frames and current_frame_idx < len(baseline_frames):
            base = baseline_frames[current_frame_idx]
            if mode == "Ratio":
                ratio = y / np.maximum(np.asarray(base["sigma"]), 1e-30)
                fig.data[0].y = ratio
                y_label = "σ(M) / σ_baseline(M)"
                is_log_y = False
            else:
                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=np.asarray(base["sigma"]),
                        mode="lines",
                        name="ΛCDM Baseline",
                        line=dict(color="#ffb454", width=2.2, dash="dash"),
                    )
                )

    elif observable == "Differential HMF dn/dlnM":
        x = np.asarray(curr["M_h"])
        y = np.asarray(curr["hmf_diff"])
        x_label = "M [h⁻¹ M☉]"
        y_label = "dn/dlnM [h³ Mpc⁻³]"
        title = f"Differential Halo Mass Function (z={curr['redshift']:.2f})"
        is_log_x, is_log_y = True, True
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                name=f"Candidate (z={curr['redshift']:.2f})",
                line=dict(color="#35d7e5", width=3.0),
                hovertemplate="M=%{x:.3e} h⁻¹ M☉<br>dn/dlnM=%{y:.3e} h³ Mpc⁻³<extra></extra>",
            )
        )
        if baseline_frames and current_frame_idx < len(baseline_frames):
            base = baseline_frames[current_frame_idx]
            if mode == "Ratio":
                ratio = y / np.maximum(np.asarray(base["hmf_diff"]), 1e-30)
                fig.data[0].y = ratio
                y_label = "Ratio to ΛCDM Baseline"
                is_log_y = False
            else:
                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=np.asarray(base["hmf_diff"]),
                        mode="lines",
                        name="ΛCDM Baseline",
                        line=dict(color="#ffb454", width=2.2, dash="dash"),
                    )
                )

    else:  # Cumulative HMF
        x = np.asarray(curr["M_h"])
        y = np.asarray(curr["hmf_cum"])
        x_label = "M [h⁻¹ M☉]"
        y_label = "n(>M) [h³ Mpc⁻³]"
        title = f"Cumulative Halo Number Density n(>M, z={curr['redshift']:.2f})"
        is_log_x, is_log_y = True, True
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                name=f"Candidate (z={curr['redshift']:.2f})",
                line=dict(color="#a78bfa", width=3.0),
                hovertemplate="M=%{x:.3e} h⁻¹ M☉<br>n(>M)=%{y:.3e} h³ Mpc⁻³<extra></extra>",
            )
        )

    # Physical fixed axis bounds across all frames
    if fixed_axes:
        if observable == "Matter Power P(k)":
            fig.update_xaxes(range=[np.log10(x[0]), np.log10(x[-1])])
            fig.update_yaxes(range=[-1.0, 5.0] if is_log_y else [0.5, 1.5])
        elif observable == "Dimensionless Power Δ²(k)":
            fig.update_xaxes(range=[np.log10(x[0]), np.log10(x[-1])])
            fig.update_yaxes(range=[-5.0, 2.5])
        elif observable == "Mass Variance σ(M)":
            fig.update_xaxes(range=[np.log10(x[0]), np.log10(x[-1])])
            fig.update_yaxes(range=[-1.5, 1.2] if is_log_y else [0.7, 1.3])
        elif observable in {"Differential HMF dn/dlnM", "Cumulative HMF n(>M)"}:
            fig.update_xaxes(range=[np.log10(x[0]), np.log10(x[-1])])
            fig.update_yaxes(range=[-14.0, 1.0] if is_log_y else [0.0, 2.0])

    fig.update_xaxes(type="log" if is_log_x else "linear", title=x_label)
    fig.update_yaxes(type="log" if is_log_y else "linear", title=y_label)

    # Annotate frame epoch
    epoch_text = (
        f"<b>z = {curr['redshift']:.2f}</b> · a = {curr['scale_factor']:.3f} · t = {curr['cosmic_time_gyr']:.2f} Gyr<br>"
        f"Linear Growth D(z) = {curr['growth_factor']:.3f}"
    )
    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=0.98,
        y=0.96,
        text=epoch_text,
        showarrow=False,
        align="right",
        bgcolor="rgba(10, 21, 27, 0.82)"
        if theme != "Light"
        else "rgba(255, 255, 255, 0.88)",
        bordercolor="#344a53" if theme != "Light" else "#b7c2c3",
        font=dict(size=11, family="ui-monospace, monospace"),
    )

    bg_color = "rgba(6, 16, 21, 0.95)" if theme != "Light" else "#ffffff"
    text_color = "#edf1ef" if theme != "Light" else "#132126"
    grid_color = (
        "rgba(255, 255, 255, 0.08)" if theme != "Light" else "rgba(0, 0, 0, 0.08)"
    )

    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color=text_color)),
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        font=dict(color=text_color),
        margin=dict(l=65, r=25, t=55, b=50),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            bgcolor="rgba(0,0,0,0)",
        ),
        xaxis=dict(gridcolor=grid_color),
        yaxis=dict(gridcolor=grid_color),
    )
    return fig


def evolution_contact_sheet(
    frames: list[dict[str, Any]],
    observable: str,
    *,
    num_panels: int = 4,
    theme: str = "Dark",
) -> go.Figure:
    """Generate a small multiples contact sheet showing evolution across key snapshots."""
    from plotly.subplots import make_subplots

    if not frames:
        raise ValueError("Empty frames list")

    step = max(1, len(frames) // num_panels)
    indices = [min(i * step, len(frames) - 1) for i in range(num_panels)]
    indices = sorted(list(set(indices)))

    fig = make_subplots(
        rows=1,
        cols=len(indices),
        subplot_titles=[f"z = {frames[idx]['redshift']:.1f}" for idx in indices],
        shared_yaxes=True,
    )

    for col_idx, f_idx in enumerate(indices):
        frame = frames[f_idx]
        if observable == "Matter Power P(k)":
            x, y = frame["k"], frame["P_k"]
        elif observable == "Dimensionless Power Δ²(k)":
            x, y = frame["k"], frame["delta2"]
        elif observable == "Mass Variance σ(M)":
            x, y = frame["M_h"], frame["sigma"]
        else:
            x, y = frame["M_h"], frame["hmf_diff"]

        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                line=dict(color="#35d7e5", width=2.4),
                name=f"z={frame['redshift']:.1f}",
                showlegend=False,
            ),
            row=1,
            col=col_idx + 1,
        )
        fig.update_xaxes(type="log", row=1, col=col_idx + 1)
        fig.update_yaxes(type="log", row=1, col=col_idx + 1)

    bg_color = "rgba(6, 16, 21, 0.95)" if theme != "Light" else "#ffffff"
    text_color = "#edf1ef" if theme != "Light" else "#132126"
    fig.update_layout(
        title=f"Evolution Contact Sheet: {observable} (z={frames[indices[0]]['redshift']:.1f} → z={frames[indices[-1]]['redshift']:.1f})",
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        font=dict(color=text_color),
        height=320,
        margin=dict(l=50, r=20, t=50, b=40),
    )
    return fig


def evolution_frame_summary_table(frames: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return a clean, compact tabular transcript of cosmic evolution across frames."""
    rows = []
    for f in frames:
        rows.append(
            {
                "Frame": f["frame_index"],
                "Redshift z": f"{f['redshift']:.2f}",
                "Scale factor a": f"{f['scale_factor']:.3f}",
                "Cosmic Age [Gyr]": f"{f['cosmic_time_gyr']:.3f}",
                "Growth D(z)": f"{f['growth_factor']:.3f}",
                "Nonlinear Scale knl [Mpc⁻¹]": f"{f['knl']:.3f}"
                if f["knl"]
                else "All linear",
                "Milestones": "; ".join(m["name"] for m in f["milestones"]) or "None",
            }
        )
    return rows
