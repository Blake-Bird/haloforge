"""HaloForge — stable interactive AxiCLASS structure laboratory."""

from __future__ import annotations

import json
from copy import deepcopy
from io import BytesIO
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config.defaults import DEFAULT_PARAMS
from config.ranges import CONTROL_RANGES
from engine.class_runner import ClassRuntimeError, build_class_settings, environment_diagnostics, tiny_class_smoke_test
from engine.cosmology import derived_quantities
from engine.fitting_functions import FITTING_NAMES, FIT_METADATA, fitting_values
from engine.hmf import cumulative_hmf, hmf_z
from engine.sigma import sigma_grid, sigma_integrand_per_logk
from engine.windows import top_hat_W_exact, top_hat_W_series, window_W, window_squared
import state.run_storage as run_storage
from state.run_storage import (
    delete_run,
    duplicate_run,
    get_run_label,
    load_all_runs,
    rename_run,
    save_draft_params,
    set_baseline,
)
from state.session import (
    clear_loaded_run,
    current_pipeline_run,
    get_active_params,
    get_matter_power_result,
    get_params,
    get_sigma_result,
    init_session,
    load_run_into_session,
    reset_params,
    run_new_cosmology,
    slow_parameters_changed,
)

APP_ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="HaloForge", page_icon="◉", layout="wide", initial_sidebar_state="expanded")
st.markdown(f"<style>{(APP_ROOT / 'assets/custom.css').read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)
init_session()
params = get_params()

COLORS = ["#35d7e5", "#ffb454", "#a78bfa", "#ff718b", "#45d49d", "#67a7ff", "#f5df70", "#d879ff"]
WINDOWS = ["Top-hat", "Gaussian", "Sharp-k"]

PARAM_INFO = {
    "A_s": ("Primordial amplitude Aₛ", "Sets the overall normalization of primordial curvature fluctuations.", "Increasing Aₛ shifts the entire primordial spectrum upward at every k; nₛ controls the tilt.", "amplitude"),
    "n_s": ("Spectral tilt nₛ", "Controls the relative primordial power on scales above and below the pivot kₚ.", "Changing nₛ rotates the spectrum around kₚ: larger nₛ gives relatively more high-k power.", "tilt"),
    "k_pivot": ("Pivot scale kₚ", "The reference wavenumber where the amplitude Aₛ is defined.", "It is an anchor for the power-law parameterization, not a new physical feature in P(k).", "pivot"),
    "H0": ("Expansion rate H₀", "The present expansion rate and the source of h = H₀/100.", "Changing H₀ changes physical densities and characteristic scales; the guide shows faster versus slower expansion, not a fake P(k) line.", "expansion"),
    "Omega_m": ("Matter density Ωₘ", "Fraction of today's critical density in total matter.", "It changes equality, growth, the mean matter density, the M↔R mapping, and halo abundance.", "matter"),
    "Omega_b": ("Baryon density Ωᵦ", "The baryonic part of matter; Ωcdm is computed as Ωₘ−Ωᵦ.", "It changes baryon loading and the scale-dependent oscillatory structure in the processed matter spectrum.", "baryon"),
    "Omega_k": ("Curvature Ωₖ", "Contribution of spatial curvature to the background geometry.", "Zero is spatially flat; positive and negative values change geometry, distances, and growth.", "curvature"),
    "N_eff": ("Relativistic species N_eff", "Sets the effective early radiation content beyond photons.", "More radiation delays matter–radiation equality and changes early-time processing of perturbations.", "radiation"),
    "Tcmb": ("CMB temperature Tcmb", "Sets today's photon temperature and therefore the photon energy density.", "A higher Tcmb raises radiation density and shifts early expansion and equality.", "temperature"),
    "tau_reio": ("Optical depth τ", "Integrated probability that CMB photons rescattered after reionization.", "It is central for CMB observables; its direct influence on the linear matter spectrum is limited in this app.", "reionization"),
    "f_EDE": ("EDE peak fraction fEDE", "Maximum temporary early-dark-energy fraction near the critical epoch.", "A larger value raises the height of the early expansion pulse; it does not act like dark energy today.", "ede_amp"),
    "log10_a_c": ("EDE epoch log₁₀aᶜ", "Locates when the axion-like field becomes dynamical.", "More negative values move the temporary EDE pulse earlier in cosmic history.", "ede_epoch"),
    "k_min": ("Minimum k", "Largest Fourier scale retained in the sampled spectrum.", "Raising kmin discards large-scale modes and can bias very large smoothing radii.", "kmin"),
    "k_max": ("Maximum k", "Smallest Fourier scale retained in the sampled spectrum.", "Low-mass halos need sufficiently high kmax for converged σ(M).", "kmax"),
    "k_points": ("k samples", "Number of logarithmically spaced AxiCLASS P(k) samples.", "More points resolve shape and stabilize fixed-grid integration, but require more backend evaluations.", "samples"),
    "quad_limit": ("Integration batch size", "Safe vectorization batch used by the fixed log-k Simpson integrator.", "It controls memory batching only; the complete sampled CLASS k grid is always integrated.", "accuracy"),
    "delta_c": ("Collapse threshold δc", "Linear overdensity threshold used by multiplicity models.", "A higher threshold makes collapse rarer and suppresses the high-mass tail.", "threshold"),
    "mass_min_exp": ("Minimum halo mass", "Lower log₁₀ mass bound in h⁻¹M☉.", "Smaller masses correspond to smaller smoothing radii and greater sensitivity to high k.", "mass_range"),
    "mass_max_exp": ("Maximum halo mass", "Upper log₁₀ mass bound in h⁻¹M☉.", "Larger masses probe rare peaks and the exponentially falling HMF tail.", "mass_range"),
    "mass_points": ("Mass samples", "Number of logarithmic M samples used for σ(M), its derivative, and the HMF.", "A denser grid stabilizes numerical derivatives and curve detail.", "samples"),
    "selected_mass_exp": ("Inspection mass", "Mass highlighted in the σ integrand and learning views.", "The marker moves across the mass range and reveals which k modes contribute to that scale.", "inspect_mass"),
    "single_z": ("Analysis redshift z", "Epoch used by focused σ and HMF diagnostics.", "At larger z, linear growth is smaller and massive halos are much rarer.", "redshift"),
}

VISUALS = {
    "amplitude": '''<svg viewBox="0 0 300 105" aria-label="A_s shifts the primordial spectrum vertically"><path class="axis" d="M28 12V84H282"/><path class="guide" d="M38 38C102 44 184 54 272 64"/><path class="accent" d="M38 22C102 28 184 38 272 48"/><path class="arrow" d="M238 61V49M233 54l5-5 5 5"/><text x="41" y="18">higher Aₛ</text><text x="41" y="52">lower Aₛ</text><text x="244" y="99">log k</text><text x="4" y="17">𝒫ℛ</text></svg>''',
    "tilt": '''<svg viewBox="0 0 300 105" aria-label="n_s pivots the primordial spectrum"><path class="axis" d="M28 12V84H282"/><path class="guide" d="M38 25L272 66"/><path class="accent" d="M38 66L272 28"/><line class="marker" x1="155" y1="12" x2="155" y2="84"/><circle class="dot" cx="155" cy="46" r="4"/><text x="162" y="18">pivot kₚ</text><text x="205" y="27">larger nₛ</text><text x="205" y="74">smaller nₛ</text></svg>''',
    "pivot": '''<svg viewBox="0 0 300 105"><path class="axis" d="M28 12V84H282"/><path class="accent" d="M38 28C105 33 188 46 272 65"/><line class="marker" x1="130" y1="12" x2="130" y2="84"/><line class="marker amber" x1="205" y1="12" x2="205" y2="84"/><circle class="dot" cx="130" cy="40" r="4"/><circle class="dot amber-fill" cx="205" cy="51" r="4"/><text x="100" y="98">different anchors, same law</text></svg>''',
    "expansion": '''<svg viewBox="0 0 300 105" aria-label="H0 changes present expansion rate"><circle class="universe slow" cx="82" cy="52" r="20"/><circle class="universe fast" cx="215" cy="52" r="20"/><path class="arrow" d="M82 18V5M77 10l5-5 5 5M82 86v13M77 94l5 5 5-5M48 52H35M40 47l-5 5 5 5M116 52h13M124 47l5 5-5 5"/><path class="arrow amber" d="M215 12V2M210 7l5-5 5 5M215 92v11M210 98l5 5 5-5M175 52h-15M165 47l-5 5 5 5M255 52h15M265 47l5 5-5 5"/><text x="50" y="100">lower H₀</text><text x="187" y="100">higher H₀</text></svg>''',
    "matter": '''<svg viewBox="0 0 300 105"><path class="axis" d="M28 12V84H282"/><path class="guide" d="M38 78C65 38 92 20 123 25C168 32 211 56 272 75"/><path class="accent" d="M38 79C79 31 115 18 153 28C195 39 229 59 272 73"/><line class="marker" x1="123" y1="18" x2="123" y2="84"/><line class="marker amber" x1="153" y1="18" x2="153" y2="84"/><text x="68" y="99">equality/turnover shifts</text></svg>''',
    "baryon": '''<svg viewBox="0 0 300 105"><path class="axis" d="M28 12V84H282"/><path class="guide" d="M38 64C65 44 81 64 106 48S146 62 169 48S211 59 236 48S260 52 272 48"/><path class="accent" d="M38 58C63 31 82 69 106 39S147 67 170 39S213 65 237 39S261 51 272 39"/><text x="77" y="97">baryon loading changes wiggles</text></svg>''',
    "curvature": '''<svg viewBox="0 0 300 105"><path class="guide" d="M30 72Q82 22 134 72"/><path class="accent" d="M164 72Q216 112 268 72"/><path class="axis" d="M30 52H134M164 52H268"/><text x="55" y="93">closed-like</text><text x="197" y="93">open-like</text><text x="126" y="16">Ωₖ=0 is flat</text></svg>''',
    "radiation": '''<svg viewBox="0 0 300 105"><path class="axis" d="M25 55H280"/><circle class="dot" cx="88" cy="55" r="5"/><circle class="dot amber-fill" cx="154" cy="55" r="5"/><path class="arrow" d="M88 35H154M148 30l6 5-6 5"/><text x="39" y="26">less radiation</text><text x="153" y="26">more radiation</text><text x="63" y="82">equality</text><text x="132" y="82">later equality</text></svg>''',
    "temperature": '''<svg viewBox="0 0 300 105"><path class="guide" d="M28 42C46 16 64 68 82 42S118 16 136 42"/><path class="accent" d="M164 42C181 8 198 76 215 42S249 8 266 42"/><circle class="dot" cx="82" cy="76" r="8"/><circle class="dot amber-fill" cx="215" cy="76" r="13"/><text x="34" y="99">lower photon density</text><text x="169" y="99">higher photon density</text></svg>''',
    "reionization": '''<svg viewBox="0 0 300 105"><path class="accent" d="M20 52H278"/><circle class="dot" cx="105" cy="52" r="6"/><circle class="dot amber-fill" cx="190" cy="52" r="6"/><path class="guide" d="M105 52L72 23M190 52L224 24"/><text x="52" y="94">larger τ → more rescattering</text></svg>''',
    "ede_amp": '''<svg viewBox="0 0 300 105"><path class="axis" d="M28 12V84H282"/><path class="guide" d="M35 80C92 80 102 76 120 43S147 80 276 80"/><path class="accent" d="M35 80C92 80 101 75 120 16S148 80 276 80"/><text x="82" y="99">temporary early pulse</text><text x="136" y="21">larger fEDE</text></svg>''',
    "ede_epoch": '''<svg viewBox="0 0 300 105"><path class="axis" d="M25 82H282"/><path class="guide" d="M35 80C58 80 64 72 78 23S101 80 130 80"/><path class="accent" d="M155 80C178 80 184 72 198 23S221 80 274 80"/><path class="arrow" d="M92 13H185M179 8l6 5-6 5"/><text x="45" y="99">earlier</text><text x="225" y="99">later</text></svg>''',
    "kmin": '''<svg viewBox="0 0 300 105"><path class="axis" d="M28 12V84H282"/><path class="accent" d="M35 70C80 35 128 25 178 39S240 65 275 72"/><rect class="shade" x="30" y="12" width="70" height="72"/><line class="marker" x1="100" y1="12" x2="100" y2="84"/><text x="36" y="99">discarded largest scales</text></svg>''',
    "kmax": '''<svg viewBox="0 0 300 105"><path class="axis" d="M28 12V84H282"/><path class="accent" d="M35 70C80 35 128 25 178 39S240 65 275 72"/><rect class="shade" x="220" y="12" width="58" height="72"/><line class="marker" x1="220" y1="12" x2="220" y2="84"/><text x="119" y="99">discarded smallest scales</text></svg>''',
    "samples": '''<svg viewBox="0 0 300 105"><path class="guide" d="M25 76C82 22 158 24 278 66"/><g class="sparse"><circle cx="25" cy="76" r="4"/><circle cx="90" cy="31" r="4"/><circle cx="176" cy="31" r="4"/><circle cx="278" cy="66" r="4"/></g><g class="dense"><circle cx="48" cy="56" r="2"/><circle cx="70" cy="41" r="2"/><circle cx="112" cy="26" r="2"/><circle cx="140" cy="25" r="2"/><circle cx="205" cy="39" r="2"/><circle cx="244" cy="54" r="2"/></g><text x="65" y="99">denser sampling follows the same curve</text></svg>''',
    "accuracy": '''<svg viewBox="0 0 300 105"><path class="axis" d="M28 12V84H282"/><path class="accent" d="M34 78C72 42 108 26 150 35S225 71 276 74"/><path class="bars" d="M45 78V70M65 78V57M85 78V45M105 78V36M125 78V33M145 78V35M165 78V42M185 78V51M205 78V61M225 78V68M245 78V72"/><text x="56" y="99">complete log-k grid, safe batches</text></svg>''',
    "threshold": '''<svg viewBox="0 0 300 105"><path class="axis" d="M25 82H282"/><path class="accent" d="M28 82C77 82 89 18 150 18S225 82 278 82"/><line class="marker" x1="214" y1="14" x2="214" y2="82"/><path class="tail" d="M214 62C233 75 255 81 278 82V82H214Z"/><text x="201" y="11">δc</text><text x="221" y="99">collapse tail</text></svg>''',
    "mass_range": '''<svg viewBox="0 0 300 105"><path class="axis" d="M28 55H282"/><line class="marker" x1="75" y1="25" x2="75" y2="82"/><line class="marker amber" x1="238" y1="25" x2="238" y2="82"/><path class="arrow" d="M75 34H238M81 29l-6 5 6 5M232 29l6 5-6 5"/><text x="39" y="99">dwarf</text><text x="135" y="99">galaxy/group</text><text x="235" y="99">cluster</text></svg>''',
    "inspect_mass": '''<svg viewBox="0 0 300 105"><path class="axis" d="M28 55H282"/><line class="marker moving" x1="158" y1="18" x2="158" y2="82"/><circle class="dot" cx="158" cy="55" r="6"/><text x="87" y="99">selected M chooses smoothing R</text></svg>''',
    "redshift": '''<svg viewBox="0 0 300 105"><path class="axis" d="M28 12V84H282"/><path class="accent" d="M35 28C105 34 185 48 275 70"/><path class="guide" d="M35 48C105 53 185 65 275 78"/><text x="40" y="23">today: more growth</text><text x="151" y="96">higher z: less growth</text></svg>''',
}

GRAPH_CAPTIONS = {
    "Primordial spectrum": "Aₛ fixes the amplitude at kₚ; nₛ controls the power-law tilt around that pivot.",
    "Matter P(k)": "Linear matter power sampled directly from the completed CLASS/AxiCLASS run in Mpc units.",
    "Dimensionless Δ²(k)": "Δ²=k³P/(2π²) is the contribution to variance per logarithmic interval before smoothing.",
    "Processing shape": "A normalized T²-like shape proxy, P(k)/kⁿˢ, isolates scale-dependent processing from primordial tilt.",
    "Window W": "Smoothing kernels weight Fourier modes by kR; the top-hat oscillates and can become negative.",
    "Window W²": "σ² depends on W², so negative top-hat lobes contribute positively after squaring.",
    "Top-hat series error": "The small-y series avoids catastrophic cancellation and is checked against the exact expression.",
    "Mass–radius map": "Each physical mass corresponds to a comoving top-hat radius at the run's mean matter density.",
    "σ(M)": "RMS linear density contrast after smoothing. Larger masses average larger volumes and usually have smaller σ.",
    "σ slope": "The absolute logarithmic slope is one of the three factors entering dn/dlnM.",
    "σ integrand": "This curve shows exactly which k modes contribute to σ² for the selected mass and redshift.",
    "Growth": "Independent CLASS growth and the σ₈ ratio should track each other when the finite k integral is converged.",
    "Multiplicity f(σ)": "Collapse prescriptions and simulation-calibrated fits map fluctuation rarity into halo multiplicity.",
    "HMF": "Differential halo abundance per logarithmic mass interval in h³ Mpc⁻³.",
    "Cumulative HMF": "Number density of halos above mass M, integrated from the differential HMF.",
}


def set_plot(fig: go.Figure, title: str, caption: str) -> go.Figure:
    fig.update_layout(title=title, meta={"caption": caption})
    return fig


def _trace_values(fig: go.Figure, axis: str) -> np.ndarray:
    values: list[float] = []
    for trace in fig.data:
        raw = getattr(trace, axis, None)
        if raw is None:
            continue
        try:
            arr = np.asarray(raw, dtype=float).ravel()
        except (TypeError, ValueError):
            continue
        values.extend(arr[np.isfinite(arr)].tolist())
    return np.asarray(values, dtype=float)


def _apply_custom_axis(fig: go.Figure, axis: str, minimum: str, maximum: str) -> None:
    try:
        lo = float(minimum) if minimum.strip() else None
        hi = float(maximum) if maximum.strip() else None
    except ValueError:
        return
    if lo is None and hi is None:
        return
    axis_obj = fig.layout.xaxis if axis == "x" else fig.layout.yaxis
    axis_type = axis_obj.type or "linear"
    current = _trace_values(fig, axis)
    if current.size == 0:
        return
    data_lo = float(np.nanmin(current[current > 0])) if axis_type == "log" and np.any(current > 0) else float(np.nanmin(current))
    data_hi = float(np.nanmax(current))
    lo = data_lo if lo is None else lo
    hi = data_hi if hi is None else hi
    if hi <= lo or (axis_type == "log" and lo <= 0):
        return
    target = [float(np.log10(lo)), float(np.log10(hi))] if axis_type == "log" else [lo, hi]
    (fig.update_xaxes if axis == "x" else fig.update_yaxes)(range=target, autorange=False)


def chart(fig: go.Figure, height: int = 430, key: str | None = None, axis_controls: bool = False) -> None:
    title = str(fig.layout.title.text or "")
    caption = ""
    if isinstance(fig.layout.meta, dict):
        caption = str(fig.layout.meta.get("caption", ""))
    show_legend = sum(1 for trace in fig.data if getattr(trace, "showlegend", True) is not False) > 1
    bottom = 98 if show_legend else 62
    fig.update_layout(
        height=height,
        margin=dict(l=70, r=28, t=84, b=bottom),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#091218",
        font=dict(family="IBM Plex Sans, sans-serif", color="#c9d3d9", size=12),
        title=dict(text=title, x=0.02, xanchor="left", y=0.97, yanchor="top", font=dict(size=17 if len(title) < 54 else 14, color="#f5f1e8")),
        hovermode="closest",
        hoverlabel=dict(bgcolor="#111d23", bordercolor="#4d626b", font_color="#f5f7f4"),
        legend=dict(
            bgcolor="rgba(9,18,24,.92)", bordercolor="#2b3d45", borderwidth=1,
            orientation="h", y=-0.20, yanchor="top", x=0, xanchor="left",
            font=dict(size=10), tracegroupgap=4,
        ),
        showlegend=show_legend,
    )
    fig.update_xaxes(gridcolor="#1c2a31", zerolinecolor="#3b4a52", showspikes=True, spikemode="across", spikesnap="cursor", spikecolor="#74878f", automargin=True, exponentformat="power")
    fig.update_yaxes(gridcolor="#1c2a31", zerolinecolor="#3b4a52", showspikes=True, spikemode="across", spikesnap="cursor", spikecolor="#74878f", automargin=True, exponentformat="power")

    if fig.layout.yaxis.type == "log" and fig.layout.yaxis.range is None:
        values = _trace_values(fig, "y")
        values = values[values > 0]
        if values.size:
            vmax = float(np.max(values))
            meaningful = values[values >= vmax * 1e-14]
            vmin = float(np.min(meaningful if meaningful.size else values))
            lo, hi = np.log10(vmin), np.log10(vmax)
            pad = max(0.18, 0.06 * max(hi - lo, 1.0))
            fig.update_yaxes(range=[lo - pad, hi + pad])

    if axis_controls and key:
        with st.popover(f"Axes · {title[:24]}"):
            mode = st.selectbox("Range mode", ["Best default", "Full data", "Custom"], key=f"{key}_range_mode")
            axes = st.columns(2)
            with axes[0]:
                xtype = st.selectbox("x scale", ["Keep", "Linear", "Log"], key=f"{key}_xtype")
            with axes[1]:
                ytype = st.selectbox("y scale", ["Keep", "Linear", "Log"], key=f"{key}_ytype")
            if xtype != "Keep":
                fig.update_xaxes(type=xtype.lower(), autorange=True)
            if ytype != "Keep":
                fig.update_yaxes(type=ytype.lower(), autorange=True)
            if mode == "Full data":
                fig.update_xaxes(autorange=True)
                fig.update_yaxes(autorange=True)
            elif mode == "Custom":
                xcols = st.columns(2)
                xmin = xcols[0].text_input("x minimum", key=f"{key}_xmin")
                xmax = xcols[1].text_input("x maximum", key=f"{key}_xmax")
                ycols = st.columns(2)
                ymin = ycols[0].text_input("y minimum", key=f"{key}_ymin")
                ymax = ycols[1].text_input("y maximum", key=f"{key}_ymax")
                _apply_custom_axis(fig, "x", xmin, xmax)
                _apply_custom_axis(fig, "y", ymin, ymax)

    st.plotly_chart(fig, width="stretch", key=key, config={
        "displaylogo": False, "responsive": True, "scrollZoom": True,
        "modeBarButtonsToAdd": ["drawline", "eraseshape"],
        "toImageButtonOptions": {"format": "png", "scale": 2},
    })
    if caption:
        st.caption(caption)


def info_label(key: str, title: str | None = None) -> None:
    heading, meaning, downstream, kind = PARAM_INFO[key]
    st.markdown(
        f'''<div class="parameter-lens"><span>{title or heading}<i>?</i></span><div class="lens-card"><div class="lens-kicker">PHYSICS GUIDE</div><h4>{heading}</h4><div class="lens-visual">{VISUALS[kind]}</div><p>{meaning}</p><b>{downstream}</b></div></div>''',
        unsafe_allow_html=True,
    )


def slider(key: str, title: str | None = None) -> None:
    spec = CONTROL_RANGES[key]
    info_label(key, title)
    kwargs = dict(
        label=title or key, min_value=spec["min"], max_value=spec["max"],
        value=params.get(key, spec["default"]), step=spec["step"],
        label_visibility="collapsed", key=f"hf_{key}",
    )
    if "format" in spec:
        kwargs["format"] = spec["format"]
    params[key] = st.slider(**kwargs)


def clear_widgets() -> None:
    for key in list(st.session_state):
        if key.startswith("hf_"):
            del st.session_state[key]


def _prepare_multiselect_state(key: str, valid_values) -> None:
    if key not in st.session_state:
        return

    stored = st.session_state[key]
    valid = set(valid_values)

    if not isinstance(stored, (list, tuple)):
        del st.session_state[key]
        return

    cleaned = []
    changed = False

    for value in stored:
        if isinstance(value, (str, int, float, bool)) and value in valid:
            cleaned.append(value)
        else:
            changed = True

    if changed:
        if cleaned:
            st.session_state[key] = cleaned
        else:
            del st.session_state[key]


def _prepare_scalar_widget_state(key: str, valid_values) -> None:
    if key not in st.session_state:
        return

    value = st.session_state[key]
    valid = set(valid_values)

    if not isinstance(value, (str, int, float, bool)) or value not in valid:
        del st.session_state[key]


def apply_preset(name: str) -> None:
    clear_widgets()
    params.clear()
    params.update(deepcopy(DEFAULT_PARAMS))
    if name == "LCDM":
        params["enable_ede"] = False
        st.session_state["run_name_draft"] = "Baseline LCDM"
    elif name == "EDE":
        params.update(enable_ede=True, f_EDE=0.10, log10_a_c=-3.5, n_EDE=3)
        st.session_state["run_name_draft"] = "EDE f0.10"
    elif name == "High amplitude":
        params.update(enable_ede=False, A_s=2.35e-9)
        st.session_state["run_name_draft"] = "High A_s"
    elif name == "Blue tilt":
        params.update(enable_ede=False, n_s=0.99)
        st.session_state["run_name_draft"] = "Blue tilt"
    save_draft_params(params)


def sidebar() -> str:
    with st.sidebar:
        st.markdown('<div class="brand"><i></i><div><b>HALOFORGE</b><small>AXICLASS STRUCTURE LAB</small></div></div>', unsafe_allow_html=True)
        section = st.radio("Navigation", ["Dashboard", "Graph studio", "Compare lab", "Structure field", "Learn the pipeline", "Fit + window atlas", "Runs + export", "Diagnostics"], label_visibility="collapsed")
        st.markdown('<div class="preset-label">QUICK UNIVERSES</div>', unsafe_allow_html=True)
        pcols = st.columns(2)
        for i, preset in enumerate(["LCDM", "EDE", "High amplitude", "Blue tilt"]):
            if pcols[i % 2].button(preset, key=f"preset_{i}", width="stretch"):
                apply_preset(preset)
                st.rerun()
        st.caption("Presets load stable starting values. Nothing expensive runs until you press Run & auto-save.")

        with st.form("cosmology_controls", clear_on_submit=False):
            run_name = st.text_input("Run name", value=st.session_state.get("run_name_draft", ""), key="hf_run_name", placeholder="Leave blank for an automatic name")
            with st.expander("Primordial field", expanded=True):
                slider("A_s"); slider("n_s"); slider("k_pivot")
            with st.expander("Background cosmology", expanded=True):
                slider("H0"); slider("Omega_m"); slider("Omega_b")
                slider("Omega_k"); slider("N_eff"); slider("Tcmb"); slider("tau_reio")
                d = derived_quantities(params)
                st.markdown(f'<div class="mini-readout"><span>h</span><b>{d["h"]:.4f}</b><span>Ωcdm</span><b>{d["Omega_cdm"]:.4f}</b><span>Ωr</span><b>{d["Omega_r"]:.2e}</b></div>', unsafe_allow_html=True)
            with st.expander("Axion early dark energy", expanded=True):
                params["enable_ede"] = st.toggle("Enable EDE", value=bool(params.get("enable_ede", True)), key="hf_enable_ede")
                slider("f_EDE"); slider("log10_a_c")
                params["n_EDE"] = st.select_slider("Potential index n", options=[2, 3, 4, 5, 6], value=int(params.get("n_EDE", 3)), key="hf_n_EDE")
                params["scf_parameters"] = st.text_input("Initial field θᵢ, θ̇ᵢ", value=str(params.get("scf_parameters", "2.806,0.0")), key="hf_scf_parameters")
                d = derived_quantities(params)
                st.caption(f"aᶜ = {d['a_c']:.3e}  ·  zᶜ = {d['z_c']:.1f}")
            with st.expander("Halo analysis", expanded=True):
                params["window_type"] = st.selectbox("Primary window", WINDOWS, index=WINDOWS.index(params.get("window_type", "Top-hat")), key="hf_window_type")
                params["fitting"] = st.selectbox("Primary HMF fit", FITTING_NAMES, index=FITTING_NAMES.index(params.get("fitting", FITTING_NAMES[1])), key="hf_fitting")
                slider("delta_c"); slider("mass_min_exp"); slider("mass_max_exp"); slider("mass_points"); slider("selected_mass_exp"); slider("single_z")
                params["delta_halo"] = st.number_input("Halo overdensity Δ", 75.1, 3200.0, float(params.get("delta_halo", 200.0)), 1.0, key="hf_delta_halo")
            with st.expander("CLASS + integration numerics", expanded=False):
                slider("k_min"); slider("k_max"); slider("k_points"); slider("quad_limit", "Integration batch size")
                st.caption("σ(M) now uses the complete sampled log-k grid with Simpson integration. This removes adaptive-integrator instability.")
            selected_z = st.multiselect("Redshifts sent to CLASS", [0, 0.5, 1, 2, 5, 10, 100], default=params.get("z_values", [0, 0.5, 1, 2, 5, 10, 100]), key="hf_z_values")
            params["z_values"] = sorted(set(float(z) for z in [0, *selected_z, float(params["single_z"])]))
            submitted = st.form_submit_button("Run & auto-save", type="primary", width="stretch")

        reset_col, status_col = st.columns([1, 1.7])
        if reset_col.button("Reset controls", width="stretch"):
            reset_params(); clear_widgets(); st.rerun()
        status_col.caption("Slider edits are staged in the browser and cannot trigger calculations by themselves.")

        if submitted:
            try:
                with st.spinner("Running isolated AxiCLASS, integrating σ(M,z), and saving atomically…"):
                    result = run_new_cosmology(run_name)
                st.session_state["run_flash"] = f"Saved {result['saved_run']['name']} locally."
                clear_widgets()
                st.rerun()
            except (ClassRuntimeError, ValueError) as exc:
                st.error(str(exc))
                st.info("The previous completed run is still loaded and its files were not changed.")
            except Exception as exc:
                st.error(f"The run was stopped safely: {exc}")
                st.info("The previous completed run is still loaded and its files were not changed.")

        if st.session_state.get("run_flash"):
            st.success(st.session_state.pop("run_flash"))
        if slow_parameters_changed():
            st.warning("Draft controls differ from the completed run. Graphs remain tied to the saved run until you run the new settings.")
        result = get_matter_power_result()
        if result:
            source = "cache" if result.get("from_cache") else "computed"
            active = current_pipeline_run()
            active_name = st.session_state.get("last_auto_saved_run", {}).get("name", "restored run")
            st.markdown(f'<div class="runtime-ok"><i></i>{result["class_status"]} · {source} · {len(result["redshifts"])} z<br><span>{active_name}</span></div>', unsafe_allow_html=True)
    return section


def require_run():
    result = get_matter_power_result()
    sigma = get_sigma_result()
    run = current_pipeline_run()
    if result is None or sigma is None or run is None:
        st.markdown('<div class="empty"><div class="empty-orbit"><i></i></div><b>Forge the first universe</b><span>Choose a preset or tune the staged controls, then press Run & auto-save. Completed runs persist in the local data folder.</span></div>', unsafe_allow_html=True)
        return None
    return run, result, sigma


def z_index(redshifts, z):
    arr = np.asarray(redshifts, float)
    return int(np.argmin(np.abs(arr - float(z))))


def primordial_fig(p):
    k = np.logspace(-5, 1, 500)
    y = float(p["A_s"]) * (k / float(p["k_pivot"])) ** (float(p["n_s"]) - 1)
    f = go.Figure(go.Scatter(x=k, y=y, line=dict(color=COLORS[0], width=3), hovertemplate="k=%{x:.3e} Mpc⁻¹<br>𝒫ℛ=%{y:.4e}<extra></extra>"))
    f.add_vline(x=float(p["k_pivot"]), line_dash="dot", line_color=COLORS[1], annotation_text="kₚ", annotation_position="top right")
    f.update_xaxes(type="log", title="k [Mpc⁻¹]")
    f.update_yaxes(type="log", title="𝒫ℛ(k)")
    return set_plot(f, "Primordial curvature spectrum", GRAPH_CAPTIONS["Primordial spectrum"])


def power_fig(result, redshifts=None):
    f = go.Figure(); use = redshifts or list(result["redshifts"])
    for i, z in enumerate(use):
        j = z_index(result["redshifts"], z)
        f.add_trace(go.Scatter(x=result["k"], y=result["P_by_z"][j], name=f"z={float(z):g}", line=dict(color=COLORS[i % len(COLORS)], width=2.7), hovertemplate="k=%{x:.3e} Mpc⁻¹<br>P=%{y:.4e} Mpc³<extra></extra>"))
    f.update_xaxes(type="log", title="k [Mpc⁻¹]")
    f.update_yaxes(type="log", title="P(k,z) [Mpc³]")
    return set_plot(f, "Linear matter power", GRAPH_CAPTIONS["Matter P(k)"])


def delta2_fig(result, redshifts=None):
    f = go.Figure(); use = redshifts or list(result["redshifts"])
    for i, z in enumerate(use):
        j = z_index(result["redshifts"], z)
        y = result["k"] ** 3 * result["P_by_z"][j] / (2 * np.pi**2)
        f.add_trace(go.Scatter(x=result["k"], y=y, name=f"z={float(z):g}", line=dict(color=COLORS[i % len(COLORS)], width=2.6), hovertemplate="k=%{x:.3e} Mpc⁻¹<br>Δ²=%{y:.4e}<extra></extra>"))
    f.update_xaxes(type="log", title="k [Mpc⁻¹]")
    f.update_yaxes(type="log", title="Δ²(k,z)")
    return set_plot(f, "Dimensionless power per ln k", GRAPH_CAPTIONS["Dimensionless Δ²(k)"])


def transfer_fig(result, p):
    k = np.asarray(result["k"])
    shape = np.asarray(result["P"]) / k ** float(p["n_s"])
    norm_mask = k <= max(5 * k[0], 0.003)
    shape /= float(np.median(shape[norm_mask])) if np.any(norm_mask) else shape[0]
    f = go.Figure(go.Scatter(x=k, y=shape, line=dict(color=COLORS[5], width=2.8), hovertemplate="k=%{x:.3e} Mpc⁻¹<br>T² proxy=%{y:.4e}<extra></extra>"))
    f.update_xaxes(type="log", title="k [Mpc⁻¹]")
    f.update_yaxes(type="log", title="normalized P(k)/kⁿˢ")
    return set_plot(f, "Scale-dependent processing shape", GRAPH_CAPTIONS["Processing shape"])


def windows_fig(squared=False):
    y = np.logspace(-3, 2, 1000); f = go.Figure()
    for i, w in enumerate(WINDOWS):
        values = window_squared(y, w) if squared else window_W(y, w)
        f.add_trace(go.Scatter(x=y, y=values, name=w, line=dict(color=COLORS[i], width=2.5), hovertemplate="kR=%{x:.3e}<br>response=%{y:.4f}<extra></extra>"))
    f.update_xaxes(type="log", title="kR")
    f.update_yaxes(title="W²(kR)" if squared else "W(kR)", range=[-0.03, 1.08] if squared else [-0.35, 1.08])
    key = "Window W²" if squared else "Window W"
    return set_plot(f, "Squared smoothing response" if squared else "Smoothing-window response", GRAPH_CAPTIONS[key])


def taylor_fig():
    y = np.logspace(-7, -0.3, 600)
    exact = top_hat_W_exact(y); series = top_hat_W_series(y)
    err = np.abs((series - exact) / np.maximum(np.abs(exact), 1e-30))
    f = go.Figure(go.Scatter(x=y, y=np.maximum(err, 1e-18), line=dict(color=COLORS[1], width=2.6), hovertemplate="kR=%{x:.3e}<br>relative error=%{y:.3e}<extra></extra>"))
    f.add_vline(x=0.1, line_dash="dash", line_color=COLORS[0], annotation_text="series switch")
    f.update_xaxes(type="log", title="kR")
    f.update_yaxes(type="log", title="relative error")
    return set_plot(f, "Small-kR top-hat validation", GRAPH_CAPTIONS["Top-hat series error"])


def sigma_fig(sigma, p, redshifts=None):
    f = go.Figure(); use = redshifts or list(sigma["redshifts"])
    for i, z in enumerate(use):
        j = z_index(sigma["redshifts"], z)
        f.add_trace(go.Scatter(x=sigma["M_h"], y=sigma["sigma_by_z"][j], name=f"z={float(z):g}", line=dict(color=COLORS[i % len(COLORS)], width=2.7), hovertemplate="M=%{x:.3e} h⁻¹M☉<br>σ=%{y:.4f}<extra></extra>"))
    f.add_hline(y=float(p["delta_c"]), line_dash="dot", line_color=COLORS[3], annotation_text="δc", annotation_position="top right")
    f.update_xaxes(type="log", title="M [h⁻¹ M☉]")
    f.update_yaxes(type="log", title="σ(M,z)")
    return set_plot(f, "Mass variance", GRAPH_CAPTIONS["σ(M)"])


def derivative_fig(sigma, p, z=None):
    use_z = float(p["single_z"] if z is None else z)
    j = z_index(sigma["redshifts"], use_z)
    y = np.abs(sigma["dlnsigma_dlnM_by_z"][j])
    f = go.Figure(go.Scatter(x=sigma["M_h"], y=y, line=dict(color=COLORS[4], width=2.7), hovertemplate="M=%{x:.3e} h⁻¹M☉<br>|slope|=%{y:.4f}<extra></extra>"))
    f.update_xaxes(type="log", title="M [h⁻¹ M☉]")
    f.update_yaxes(type="log", title="|d lnσ / d lnM|")
    return set_plot(f, "Logarithmic variance slope", GRAPH_CAPTIONS["σ slope"])


def radius_fig(sigma):
    f = go.Figure(go.Scatter(x=sigma["M_h"], y=sigma["R"], line=dict(color=COLORS[2], width=2.8), hovertemplate="M=%{x:.3e} h⁻¹M☉<br>R=%{y:.4f} Mpc<extra></extra>"))
    f.update_xaxes(type="log", title="M [h⁻¹ M☉]")
    f.update_yaxes(type="log", title="R [Mpc]")
    return set_plot(f, "Mass to top-hat radius", GRAPH_CAPTIONS["Mass–radius map"])


def integrand_fig(result, sigma, p):
    mass = 10 ** float(p["selected_mass_exp"])
    R = float(np.exp(np.interp(np.log(mass), np.log(sigma["M_h"]), np.log(sigma["R"]))))
    j = z_index(result["redshifts"], p["single_z"])
    f = go.Figure()
    for i, w in enumerate(WINDOWS):
        values = sigma_integrand_per_logk(result["k"], result["P_by_z"][j], R, w)
        f.add_trace(go.Scatter(x=result["k"], y=np.maximum(values, 1e-300), name=w, line=dict(color=COLORS[i], width=2.5), hovertemplate="k=%{x:.3e} Mpc⁻¹<br>dσ²/dlnk=%{y:.4e}<extra></extra>"))
    f.update_xaxes(type="log", title="k [Mpc⁻¹]")
    f.update_yaxes(type="log", title="dσ²/dlnk")
    return set_plot(f, f"Variance contribution for 10^{float(p['selected_mass_exp']):g} h⁻¹M☉", GRAPH_CAPTIONS["σ integrand"])


def growth_fig(result, sigma):
    order = np.argsort(result["redshifts"])
    z = np.asarray(result["redshifts"])[order]
    growth = np.asarray(result["growth_class"])[order]
    sigma_sorted = np.asarray(sigma["sigma8_pipeline_by_z"])[order]
    sigma_ratio = sigma_sorted / float(sigma_sorted[z_index(z, 0.0)])
    f = go.Figure()
    f.add_trace(go.Scatter(x=z, y=growth, mode="lines+markers", name="CLASS D(z)", line=dict(color=COLORS[4], width=2.8), hovertemplate="z=%{x:g}<br>D=%{y:.5f}<extra></extra>"))
    f.add_trace(go.Scatter(x=z, y=sigma_ratio, mode="lines+markers", name="σ₈(z)/σ₈(0)", line=dict(color=COLORS[0], width=2.5, dash="dash"), hovertemplate="z=%{x:g}<br>ratio=%{y:.5f}<extra></extra>"))
    f.update_xaxes(type="linear", title="redshift z")
    f.update_yaxes(title="normalized linear growth", range=[0, 1.06])
    return set_plot(f, "Growth consistency check", GRAPH_CAPTIONS["Growth"])


def multiplicity_fig(p, fits=None):
    s = np.logspace(-1.1, 0.8, 500); f = go.Figure(); chosen = fits or [p["fitting"]]
    for i, name in enumerate(chosen):
        y = fitting_values(s, p["delta_c"], name, z=p["single_z"], delta_halo=p.get("delta_halo", 200))
        f.add_trace(go.Scatter(x=s, y=y, name=name, line=dict(color=COLORS[i % len(COLORS)], width=2.5), hovertemplate="σ=%{x:.4f}<br>f(σ)=%{y:.4e}<extra></extra>"))
    f.update_xaxes(type="log", title="σ")
    f.update_yaxes(type="log", title="f(σ)")
    return set_plot(f, "Halo multiplicity functions", GRAPH_CAPTIONS["Multiplicity f(σ)"])


def hmf_fig(run, fits=None, redshifts=None, cumulative=False):
    p = run["params"]
    f = go.Figure(); chosen = fits or [p["fitting"]]; zs = redshifts or [p["single_z"]]; n = 0
    for z in zs:
        for fit in chosen:
            r = hmf_z(run, float(z), fit)
            y = cumulative_hmf(r["M_h"], r["hmf"]) if cumulative else r["hmf"]
            f.add_trace(go.Scatter(x=r["M_h"], y=y, name=f"{fit} · z={float(z):g}", line=dict(color=COLORS[n % len(COLORS)], width=2.6, dash="solid" if n < len(COLORS) else "dash"), hovertemplate="M=%{x:.3e} h⁻¹M☉<br>n=%{y:.4e} h³Mpc⁻³<extra></extra>"))
            n += 1
    f.update_xaxes(type="log", title="M [h⁻¹ M☉]")
    f.update_yaxes(type="log", title="n(>M) [h³ Mpc⁻³]" if cumulative else "dn/dlnM [h³ Mpc⁻³]")
    key = "Cumulative HMF" if cumulative else "HMF"
    return set_plot(f, "Cumulative halo abundance" if cumulative else "Differential halo mass function", GRAPH_CAPTIONS[key])


GRAPH_NAMES = ["Primordial spectrum", "Matter P(k)", "Dimensionless Δ²(k)", "Processing shape", "Window W", "Window W²", "Top-hat series error", "Mass–radius map", "σ(M)", "σ slope", "σ integrand", "Growth", "Multiplicity f(σ)", "HMF", "Cumulative HMF"]

COMPARE_METRICS = [
    "P(k)",
    "Δ²(k)",
    "Growth",
    "σ(M)",
    "σ slope",
    "HMF",
    "Cumulative HMF",
]


def graph_for(name, run, result, sigma, fits, zs):
    p = run["params"]
    return {
        "Primordial spectrum": lambda: primordial_fig(p),
        "Matter P(k)": lambda: power_fig(result, zs),
        "Dimensionless Δ²(k)": lambda: delta2_fig(result, zs),
        "Processing shape": lambda: transfer_fig(result, p),
        "Window W": lambda: windows_fig(False),
        "Window W²": lambda: windows_fig(True),
        "Top-hat series error": taylor_fig,
        "Mass–radius map": lambda: radius_fig(sigma),
        "σ(M)": lambda: sigma_fig(sigma, p, zs),
        "σ slope": lambda: derivative_fig(sigma, p),
        "σ integrand": lambda: integrand_fig(result, sigma, p),
        "Growth": lambda: growth_fig(result, sigma),
        "Multiplicity f(σ)": lambda: multiplicity_fig(p, fits),
        "HMF": lambda: hmf_fig(run, fits, zs),
        "Cumulative HMF": lambda: hmf_fig(run, fits, zs, True),
    }[name]()


def dashboard_view():
    st.markdown('<section class="hero"><div><span>COMPUTATIONAL COSMOLOGY</span><h1>One universe.<br>Every scale exposed.</h1><p>Primordial seeds → AxiCLASS P(k,z) → smoothing → σ(M,z) → halo abundance.</p></div><div class="cosmic-orbit"><i></i><b></b><em></em></div></section>', unsafe_allow_html=True)
    ready = require_run()
    if not ready:
        return
    run, result, sigma = ready; d = result["derived"]; p = run["params"]
    cols = st.columns(5)
    vals = [("h", d["h"], ".5f"), ("Ωm", d["Omega_m"], ".5f"), ("σ₈ CLASS", d["sigma8"], ".5f"), ("z samples", len(result["redshifts"]), "d"), ("k evaluations", len(result["k"]) * len(result["redshifts"]), ",d")]
    for c, (label, val, fmt) in zip(cols, vals):
        c.metric(label, format(val, fmt))
    st.markdown('<div class="section-label">PIPELINE AT A GLANCE</div>', unsafe_allow_html=True)
    g = st.columns(2)
    with g[0]: chart(power_fig(result, [0]), 430, "dash_power")
    with g[1]: chart(sigma_fig(sigma, p, [0]), 430, "dash_sigma")
    g = st.columns(2)
    with g[0]: chart(integrand_fig(result, sigma, p), 430, "dash_integrand")
    with g[1]: chart(hmf_fig(run, [p["fitting"]], [p["single_z"]]), 430, "dash_hmf")


def graph_studio_view():
    st.markdown('<div class="page-head"><span>GRAPH STUDIO</span><h2>Build the analysis canvas.</h2><p>Every graph is tied to the completed run. Use per-chart axis controls for custom paper ranges without changing the underlying arrays.</p></div>', unsafe_allow_html=True)
    ready = require_run()
    if not ready:
        return
    run, result, sigma = ready; p = run["params"]
    a, b, c = st.columns([1.4, 1, 1])
    layout = a.segmented_control("Layout", ["Focus", "2 columns", "3 columns"], default="2 columns")
    fits = b.multiselect("Fits", FITTING_NAMES, default=[p["fitting"]], max_selections=5)
    zs = c.multiselect("Redshifts", list(result["redshifts"]), default=[float(p["single_z"])], max_selections=5)
    defaults = ["Matter P(k)", "Dimensionless Δ²(k)", "σ(M)", "σ integrand", "HMF", "Growth"]
    selected = st.multiselect("Visible graph modules", GRAPH_NAMES, default=defaults)
    ncol = {"Focus": 1, "2 columns": 2, "3 columns": 3}[layout]
    columns = st.columns(ncol)
    for i, name in enumerate(selected):
        with columns[i % ncol]:
            chart(
                graph_for(
                    name,
                    run,
                    result,
                    sigma,
                    fits or [p["fitting"]],
                    zs or [p["single_z"]],
                ),
                470,
                key=f"studio_chart_{GRAPH_NAMES.index(name)}",
                axis_controls=True,
            )

def _saved_pipeline(saved, window=None):
    a = saved["arrays"]; p = saved["params"]; h = float(saved["derived"]["h"])
    red = np.asarray(a.get("redshifts", [0.])); pz = np.asarray(a.get("P_by_z", [a["P"]]))
    if window is None or window == saved.get("window_type", p.get("window_type", "Top-hat")):
        sig = np.asarray(a.get("sigma_by_z", [a["sigma"]]))
        deriv = np.asarray(a.get("dlnsigma_dlnM_by_z", [a["dlnsigma_dlnM"]]))
        R = a["R"]; rho = saved["rho0"]
    else:
        key = f"alt_sigma_{saved['run_id']}_{window}"
        cached = st.session_state.get(key)
        if cached is None:
            grids = [sigma_grid(a["M"], a["k"], row, {"h": h, "Omega_m": p["Omega_m"]}, window, int(p.get("quad_limit", 200))) for row in pz]
            cached = (np.asarray([g["sigma"] for g in grids]), np.asarray([g["dlnsigma_dlnM"] for g in grids]), grids[0]["R"], grids[0]["rho0"])
            st.session_state[key] = cached
        sig, deriv, R, rho = cached
    return {
        "params": p,
        "power_result": {"k": a["k"], "P": a["P"], "P_by_z": pz, "redshifts": red, "growth_class": a.get("growth_class", np.ones_like(red)), "derived": {"h": h}},
        "sigma_result": {"M_h": a["M_h"], "M": a["M"], "R": R, "sigma": sig[0], "sigma_by_z": sig, "dlnsigma_dlnM": deriv[0], "dlnsigma_dlnM_by_z": deriv, "redshifts": red, "rho0": rho},
    }


def transform_curve(x, y, bx, by, mode):
    x = np.asarray(x, float); y = np.asarray(y, float); bx = np.asarray(bx, float); by = np.asarray(by, float)
    if np.all(x > 0) and np.all(bx > 0) and np.all(by > 0):
        base = np.exp(np.interp(np.log(x), np.log(bx), np.log(by)))
    else:
        base = np.interp(x, bx, by)
    base = np.where(np.abs(base) > 1e-300, base, np.nan)
    if mode == "Ratio": return y / base
    if mode == "Fractional difference": return y / base - 1
    if mode == "Percent difference": return 100 * (y / base - 1)
    return y


def _metric_axis(metric):
    return {
        "P(k)": ("k [Mpc⁻¹]", "P(k,z) [Mpc³]"),
        "Δ²(k)": ("k [Mpc⁻¹]", "Δ²(k,z)"),
        "Growth": ("redshift z", "D(z)/D(0)"),
        "σ(M)": ("M [h⁻¹ M☉]", "σ(M,z)"),
        "σ slope": ("M [h⁻¹ M☉]", "|d lnσ/d lnM|"),
        "HMF": ("M [h⁻¹ M☉]", "dn/dlnM [h³ Mpc⁻³]"),
        "Cumulative HMF": ("M [h⁻¹ M☉]", "n(>M) [h³ Mpc⁻³]"),
    }[metric]


def compare_metric(metric, selected, baseline, mode, z, fits, windows):
    f = go.Figure(); curves = []
    for saved in selected:
        relevant_windows = windows if metric in {"σ(M)", "σ slope", "HMF", "Cumulative HMF"} else [""]
        for window in relevant_windows:
            pipe = _saved_pipeline(saved, window or None); a = saved["arrays"]
            relevant_fits = fits if metric in {"HMF", "Cumulative HMF"} else [""]
            for fit in relevant_fits:
                if metric == "P(k)": x, y = a["k"], a["P_by_z"][z_index(a.get("redshifts", [0]), z)]
                elif metric == "Δ²(k)": x = a["k"]; y = x**3 * a["P_by_z"][z_index(a.get("redshifts", [0]), z)] / (2 * np.pi**2)
                elif metric == "Growth": x = a.get("redshifts", [0]); y = a.get("growth_class", np.ones_like(x))
                elif metric == "σ(M)":
                    j = z_index(pipe["sigma_result"]["redshifts"], z); x = pipe["sigma_result"]["M_h"]; y = pipe["sigma_result"]["sigma_by_z"][j]
                elif metric == "σ slope":
                    j = z_index(pipe["sigma_result"]["redshifts"], z); x = pipe["sigma_result"]["M_h"]; y = np.abs(pipe["sigma_result"]["dlnsigma_dlnM_by_z"][j])
                else:
                    r = hmf_z(pipe, z, fit); x = r["M_h"]; y = cumulative_hmf(x, r["hmf"]) if metric == "Cumulative HMF" else r["hmf"]
                curves.append({"saved": saved, "window": window, "fit": fit, "x": np.asarray(x), "y": np.asarray(y)})
    if not curves:
        return f
    base_curves = [curve for curve in curves if curve["saved"]["run_id"] == baseline["run_id"]]
    for i, curve in enumerate(curves[:36]):
        matches = [candidate for candidate in base_curves if candidate["window"] == curve["window"] and candidate["fit"] == curve["fit"]]
        base = matches[0] if matches else base_curves[0]
        yy = transform_curve(curve["x"], curve["y"], base["x"], base["y"], mode)
        saved = curve["saved"]
        suffix = f" · {curve['window']}" if curve["window"] else ""
        suffix += f" · {curve['fit']}" if curve["fit"] else ""
        variant_index = (windows.index(curve["window"]) if curve["window"] in windows else 0) + (fits.index(curve["fit"]) if curve["fit"] in fits else 0)
        dash = ["solid", "dash", "dot", "dashdot"][variant_index % 4]
        f.add_trace(go.Scatter(
            x=curve["x"], y=yy, name=saved["name"] + suffix,
            line=dict(color=saved.get("color", COLORS[i % len(COLORS)]), width=3.2 if saved["run_id"] == baseline["run_id"] else 2.5, dash=dash),
            opacity=1.0 if saved["run_id"] == baseline["run_id"] else 0.9,
            hovertemplate="x=%{x:.4e}<br>y=%{y:.4e}<extra></extra>",
        ))
    x_title, overlay_y = _metric_axis(metric)
    logx = metric != "Growth"
    logy = mode == "Overlay" and metric != "Growth"
    y_title = overlay_y if mode == "Overlay" else mode
    f.update_xaxes(type="log" if logx else "linear", title=x_title)
    f.update_yaxes(type="log" if logy else "linear", title=y_title)
    if mode == "Ratio":
        f.add_hline(y=1, line_color="#8ea1a9", line_dash="dot")
    elif mode in {"Fractional difference", "Percent difference"}:
        f.add_hline(y=0, line_color="#8ea1a9", line_dash="dot")
        vals = _trace_values(f, "y")
        vals = vals[np.isfinite(vals)]
        if vals.size:
            extent = max(float(np.nanpercentile(np.abs(vals), 99)), 1e-4)
            f.update_yaxes(range=[-1.15 * extent, 1.15 * extent])
    caption = f"{mode} at z={z:g}. Each curve is compared with the baseline using the matching window and HMF fit, so unlike modeling choices are not divided by one another. Baseline: {baseline['name']}."
    return set_plot(f, metric, caption)


def compare_view():
    st.markdown(
        '<div class="page-head"><span>COMPARE LAB</span>'
        '<h2>Make subtle physics impossible to miss.</h2>'
        '<p>Overlay is the default. Ratios and residuals use a matching '
        'baseline curve for every window and fitting-function combination.</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    runs = load_all_runs()

    if len(runs) < 2:
        st.info(
            "Run at least two cosmologies. "
            "Try LCDM, run it, then EDE and run again."
        )
        return

    run_map = {
        str(run["run_id"]): run
        for run in runs
    }

    run_ids = list(run_map)

    labels = {
        run_id: get_run_label(run_map[run_id])
        for run_id in run_ids
    }

    default_run_ids = [
        str(run["run_id"])
        for run in runs[-min(3, len(runs)):]
    ]

    _prepare_multiselect_state(
        "compare_run_ids",
        run_ids,
    )

    run_select_kwargs = {}

    if "compare_run_ids" not in st.session_state:
        run_select_kwargs["default"] = default_run_ids

    selected_ids = st.multiselect(
        "Runs",
        run_ids,
        format_func=lambda run_id: labels[run_id],
        key="compare_run_ids",
        **run_select_kwargs,
    )

    selected_ids = [
        run_id
        for run_id in selected_ids
        if run_id in run_map
    ]

    selected = [
        run_map[run_id]
        for run_id in selected_ids
    ]

    if not selected:
        return

    c = st.columns(4)

    preferred_baseline_id = next(
        (
            run_id
            for run_id in selected_ids
            if run_map[run_id].get("is_baseline")
        ),
        selected_ids[0],
    )

    _prepare_scalar_widget_state(
        "compare_baseline_id",
        selected_ids,
    )

    baseline_kwargs = {}

    if "compare_baseline_id" not in st.session_state:
        baseline_kwargs["index"] = selected_ids.index(
            preferred_baseline_id
        )

    baseline_id = c[0].selectbox(
        "Baseline",
        selected_ids,
        format_func=lambda run_id: run_map[run_id]["name"],
        key="compare_baseline_id",
        **baseline_kwargs,
    )

    baseline = run_map[baseline_id]

    mode = c[1].selectbox(
        "Comparison",
        [
            "Overlay",
            "Ratio",
            "Fractional difference",
            "Percent difference",
        ],
        index=0,
        key="compare_mode",
    )

    common = sorted(
        set.intersection(
            *[
                set(
                    float(value)
                    for value in run["arrays"].get(
                        "redshifts",
                        [0.0],
                    )
                )
                for run in selected
            ]
        )
    )

    if not common:
        st.error(
            "The selected runs have no common sampled redshift."
        )
        return

    _prepare_scalar_widget_state(
        "compare_common_redshift",
        common,
    )

    redshift_kwargs = {}

    if "compare_common_redshift" not in st.session_state:
        redshift_kwargs["index"] = 0

    z = c[2].selectbox(
        "Common redshift",
        common,
        key="compare_common_redshift",
        **redshift_kwargs,
    )

    ncol = c[3].segmented_control(
        "Panels",
        ["1", "2", "3"],
        default="2",
        key="compare_panel_count",
    )

    fits = st.multiselect(
        "HMF fits",
        FITTING_NAMES,
        default=[baseline["params"]["fitting"]],
        max_selections=4,
        key="compare_fits",
    )

    windows = st.multiselect(
        "Smoothing windows",
        WINDOWS,
        default=[baseline["params"]["window_type"]],
        max_selections=3,
        key="compare_windows",
    )

    metrics = st.multiselect(
        "Comparison panels",
        COMPARE_METRICS,
        default=["P(k)", "σ(M)", "HMF"],
        key="compare_metrics",
    )

    count = (
        len(selected)
        * max(1, len(windows))
        * max(1, len(fits))
    )

    if count > 36:
        st.warning(
            "This selection creates many curves. "
            "HaloForge displays the first 36 per panel; "
            "narrow a dimension for cleaner interpretation."
        )

    cols = st.columns(int(ncol))

    for display_index, metric in enumerate(metrics):
        with cols[display_index % int(ncol)]:
            chart(
                compare_metric(
                    metric,
                    selected,
                    baseline,
                    mode,
                    float(z),
                    fits or [
                        baseline["params"]["fitting"]
                    ],
                    windows or [
                        baseline["params"]["window_type"]
                    ],
                ),
                500,
                key=(
                    "compare_chart_"
                    f"{COMPARE_METRICS.index(metric)}"
                ),
                axis_controls=True,
            )

    table = []

    for run in selected:
        p = run["params"]

        table.append(
            {
                "run": run["name"],
                "baseline": (
                    run["run_id"]
                    == baseline["run_id"]
                ),
                "H0": p["H0"],
                "Ωm": p["Omega_m"],
                "Ωb": p["Omega_b"],
                "Aₛ": p["A_s"],
                "nₛ": p["n_s"],
                "EDE": p.get("enable_ede"),
                "fEDE": p.get("f_EDE"),
                "log₁₀aᶜ": p.get("log10_a_c"),
                "σ₈": run.get("sigma8"),
            }
        )

    st.dataframe(
        pd.DataFrame(table),
        width="stretch",
        hide_index=True,
    )
    
@st.cache_data(show_spinner=False, max_entries=3)
def shared_fourier_seed(box: float, n: int, seed: int):
    rng = np.random.default_rng(int(seed))
    white = rng.normal(size=(n, n, n))
    modes = np.fft.rfftn(white)
    kx = 2 * np.pi * np.fft.fftfreq(n, d=box / n)
    ky = 2 * np.pi * np.fft.fftfreq(n, d=box / n)
    kz = 2 * np.pi * np.fft.rfftfreq(n, d=box / n)
    kk = np.sqrt(kx[:, None, None] ** 2 + ky[None, :, None] ** 2 + kz[None, None, :] ** 2)
    return modes, kk


def gaussian_field_slice(k, power, box, n, smoothing, modes, kk):
    k = np.asarray(k, float); power = np.asarray(power, float)
    amplitude = np.zeros_like(kk, dtype=float)
    mask = (kk >= k[0]) & (kk <= k[-1]) & (kk > 0)
    interpolated = np.exp(np.interp(np.log(kk[mask]), np.log(k), np.log(np.maximum(power, 1e-300))))
    amplitude[mask] = (n / box) ** 1.5 * np.sqrt(interpolated) * np.exp(-0.5 * (kk[mask] * smoothing) ** 2)
    filtered = modes * amplitude
    filtered[0, 0, 0] = 0
    field = np.fft.irfftn(filtered, s=(n, n, n)).real
    field -= field.mean()
    return field[:, :, n // 2]


def structure_view():
    st.markdown('<div class="page-head"><span>STRUCTURE FIELD</span><h2>Same phases. Honest amplitude differences.</h2><p>A periodic 3D Gaussian linear-density realization is filtered by each run’s computed P(k,z); all panels use the same Fourier seed and a common normalization.</p></div>', unsafe_allow_html=True)
    runs = load_all_runs()
    if not runs:
        st.info("Run a cosmology first.")
        return
    run_map = {r["run_id"]: r for r in runs}
    default_ids = [r["run_id"] for r in runs[-min(3, len(runs)):]]
    with st.form("structure_controls"):
        selected_ids = st.multiselect("Runs", list(run_map), default=default_ids, format_func=lambda rid: run_map[rid]["name"], max_selections=4)
        selected = [run_map[rid] for rid in selected_ids]
        common = sorted(set.intersection(*[set(map(float, r["arrays"].get("redshifts", [0]))) for r in selected])) if selected else []
        c = st.columns(5)
        z = c[0].selectbox("Redshift", common or [0.0])
        box = c[1].slider("Box size [Mpc]", 50, 1000, 300, 10)
        smoothing = c[2].slider("Gaussian smoothing [Mpc]", 0.2, 12.0, 2.0, 0.2)
        n = c[3].select_slider("Grid", options=[64, 96, 128], value=96)
        seed = c[4].number_input("Shared seed", 0, 999999, 271828)
        baseline_id = st.selectbox("Field baseline", selected_ids or [""], format_func=lambda rid: run_map[rid]["name"] if rid in run_map else "No run")
        st.form_submit_button("Render matched fields", type="primary")
    if not selected:
        return
    if not common:
        st.error("The selected runs have no common sampled redshift.")
        return
    baseline = run_map[baseline_id] if baseline_id in run_map else selected[0]

    modes, kk = shared_fourier_seed(float(box), int(n), int(seed))
    fields = []
    for r in selected:
        a = r["arrays"]; j = z_index(a.get("redshifts", [0]), z)
        fields.append(gaussian_field_slice(a["k"], a["P_by_z"][j], float(box), int(n), float(smoothing), modes, kk))
    baseline_index = selected.index(baseline)
    base_rms = max(float(np.std(fields[baseline_index])), 1e-30)
    normalized = [field / base_rms for field in fields]
    all_values = np.concatenate([np.abs(field).ravel() for field in normalized])
    color_extent = max(float(np.percentile(all_values, 99.5)), 1.0)
    coords = np.linspace(0, box, n, endpoint=False)

    cols = st.columns(len(selected))
    for i, (r, field) in enumerate(zip(selected, normalized)):
        fig = go.Figure(go.Heatmap(x=coords, y=coords, z=field, colorscale=[[0, "#07131b"], [0.25, "#123d56"], [0.5, "#c5d2c9"], [0.72, "#e8a349"], [1, "#fff1bf"]], zmin=-color_extent, zmax=color_extent, colorbar=dict(title="δ/σbase", thickness=9)))
        fig.update_xaxes(title="x [Mpc]", showgrid=False)
        fig.update_yaxes(title="y [Mpc]", showgrid=False, scaleanchor="x")
        set_plot(fig, r["name"], "Central slice of the same 3D Fourier realization. Colors are normalized by the baseline RMS, not separately by each run.")
        with cols[i]:
            chart(
                fig,
                480,
                key=f"field_{r['run_id']}",
            )
    differences = [(i, field - normalized[baseline_index]) for i, field in enumerate(normalized) if i != baseline_index]
    if differences:
        st.markdown('<div class="section-label">DIFFERENCE FROM BASELINE</div>', unsafe_allow_html=True)
        dcols = st.columns(min(3, len(differences)))
        diff_extent = max(float(np.percentile(np.concatenate([np.abs(diff).ravel() for _, diff in differences]), 99.5)), 1e-4)
        for panel, (i, diff) in enumerate(differences):
            fig = go.Figure(go.Heatmap(x=coords, y=coords, z=diff, colorscale="RdBu_r", zmid=0, zmin=-diff_extent, zmax=diff_extent, colorbar=dict(title="Δδ/σbase", thickness=9)))
            fig.update_xaxes(title="x [Mpc]", showgrid=False)
            fig.update_yaxes(title="y [Mpc]", showgrid=False, scaleanchor="x")
            set_plot(fig, f"{selected[i]['name']} − {baseline['name']}", "Matched-mode difference. Because phases and normalization are shared, both shape and amplitude changes remain visible.")
            with dcols[panel % len(dcols)]:
                chart(
                    fig,
                    440,
                    key=f"field_diff_{selected[i]['run_id']}",
                )

    left, right = st.columns([1.5, 1])
    with left:
        fig = go.Figure()
        central = n // 2
        for i, field in enumerate(normalized):
            fig.add_trace(go.Scatter(x=coords, y=field[central], name=selected[i]["name"], line=dict(color=selected[i].get("color", COLORS[i]), width=2.5)))
        fig.update_xaxes(title="distance [Mpc]")
        fig.update_yaxes(title="δ/σbaseline")
        set_plot(fig, "Matched central slice", "A one-dimensional cut through the exact same Fourier phases in every selected run.")
        chart(fig, 430, "field_slice", axis_controls=True)
    with right:
        rows = []
        base_flat = fields[baseline_index].ravel()
        for r, field in zip(selected, fields):
            flat = field.ravel()
            rows.append({
                "run": r["name"], "RMS δ": float(np.std(flat)), "RMS / baseline": float(np.std(flat) / base_rms),
                "correlation with baseline": float(np.corrcoef(flat, base_flat)[0, 1]),
                "difference RMS / baseline": float(np.std(flat - base_flat) / base_rms),
            })
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        fundamental = 2 * np.pi / box; nyquist = np.pi * n / box
        st.markdown(f'<div class="lesson-card compact"><span>FOURIER COVERAGE</span><h3>{n}³ periodic grid</h3><p>kfund = {fundamental:.4f} Mpc⁻¹<br>kNyquist = {nyquist:.4f} Mpc⁻¹<br>Gaussian smoothing = {smoothing:.2f} Mpc</p><b>The quantitative table preserves the actual linear-field RMS. The maps are a visualization, not an N-body halo catalogue.</b></div>', unsafe_allow_html=True)


def learn_view():
    st.markdown('<div class="page-head"><span>VISUAL COURSE</span><h2>Learn the pipeline by touching it.</h2><p>One connected journey from initial fluctuations to predicted halo counts.</p></div>', unsafe_allow_html=True)
    ready = require_run()
    steps = ["1 · Seeds", "2 · Processing", "3 · Choose mass", "4 · Smooth", "5 · Variance", "6 · Collapse", "7 · Count halos"]
    step = st.segmented_control("Pipeline step", steps, default=steps[0]); index = steps.index(step)
    st.markdown('<div class="pipeline-track">' + ''.join(f'<div class="{"active" if i == index else "done" if i < index else ""}"><i>{i + 1}</i><span>{s.split(" · ")[1]}</span></div>' for i, s in enumerate(steps)) + '</div>', unsafe_allow_html=True)
    if not ready:
        chart(primordial_fig(params), 430, "learn_pre")
        st.info("The primordial controls can be inspected before a run. Complete an AxiCLASS run to continue through processed matter and halos.")
        return
    run, result, sigma = ready; p = run["params"]
    left, right = st.columns([1.6, 1])
    visuals = [primordial_fig(p), power_fig(result, [0]), radius_fig(sigma), integrand_fig(result, sigma, p), sigma_fig(sigma, p, [p["single_z"]]), multiplicity_fig(p, ["Press-Schechter 1974", "Sheth-Tormen 2001"]), hmf_fig(run, [p["fitting"]], [p["single_z"]])]
    messages = [
        ("Primordial seeds", "Aₛ sets overall strength; nₛ decides how that strength is distributed across scale.", "Amplitude shifts vertically. Tilt pivots around kₚ."),
        ("The universe processes the seeds", "Radiation, matter, baryons, expansion, and EDE reshape the initial spectrum into P(k,z).", "Large k means smaller comoving structure."),
        ("Mass becomes a radius", "At the mean cosmic density, each target mass corresponds to a smoothing radius R.", "A cluster samples a much larger region than a dwarf halo."),
        ("The window selects modes", "W(kR) weights which Fourier scales contribute to the chosen mass.", "The displayed integrand is the contribution to σ² per ln k."),
        ("Variance measures fluctuation strength", "σ(M,z) is the RMS smoothed density contrast and usually falls toward high mass.", "Massive regions are rarer because larger volumes average fluctuations down."),
        ("Collapse maps peaks to halos", "f(σ) encodes a collapse prescription or simulation-calibrated multiplicity relation.", "Fits differ because halo definitions and calibrations differ."),
        ("The HMF predicts abundance", "dn/dlnM combines density, multiplicity, and the σ slope.", "The high-mass tail is exponentially sensitive to small cosmological changes."),
    ]
    with left: chart(visuals[index], 530, key=f"learn_{index}")
    with right:
        h, body, take = messages[index]
        st.markdown(f'<div class="lesson-card"><span>STEP {index + 1} OF 7</span><h3>{h}</h3><p>{body}</p><b>{take}</b></div>', unsafe_allow_html=True)
        if index == 0: st.latex(r"\mathcal P_\mathcal R(k)=A_s(k/k_*)^{n_s-1}")
        elif index == 2: st.latex(r"R(M)=\left(3M/4\pi\rho_{m,0}\right)^{1/3}")
        elif index == 3: st.latex(r"W_{TH}(y)=3(\sin y-y\cos y)/y^3")
        elif index == 4: st.latex(r"\sigma^2(R,z)=\frac1{2\pi^2}\int k^2P(k,z)W^2(kR)dk")
        elif index == 6: st.latex(r"\frac{dn}{d\ln M}=\frac{\rho_{m,0}}M f(\sigma)\left|\frac{d\ln\sigma}{d\ln M}\right|")


def atlas_view():
    st.markdown('<div class="page-head"><span>FIT + WINDOW ATLAS</span><h2>Expose every modeling choice.</h2></div>', unsafe_allow_html=True)
    active = get_active_params()
    view = st.segmented_control("Atlas view", ["Multiplicity fits", "Window functions", "Window impact on σ", "Calibration table"], default="Multiplicity fits")
    if view == "Multiplicity fits":
        chosen = st.multiselect("Visible fits", FITTING_NAMES, default=["Press-Schechter 1974", "Sheth-Tormen 2001", "Tinker 2008", "Watson FOF 2013"], max_selections=8)
        chart(multiplicity_fig(active, chosen), 580, "atlas_fit", axis_controls=True)
    elif view == "Window functions":
        a, b = st.columns(2)
        with a: chart(windows_fig(False), 450, "atlas_w")
        with b: chart(windows_fig(True), 450, "atlas_w2")
        chart(taylor_fig(), 400, "atlas_taylor")
    elif view == "Window impact on σ":
        ready = require_run()
        if ready:
            run, result, sigma = ready; p = run["params"]; f = go.Figure()
            for i, w in enumerate(WINDOWS):
                g = sigma_grid(sigma["M"], result["k"], result["P"], {"h": result["derived"]["h"], "Omega_m": p["Omega_m"]}, w, int(p.get("quad_limit", 200)))
                f.add_trace(go.Scatter(x=sigma["M_h"], y=g["sigma"], name=w, line=dict(color=COLORS[i], width=2.7)))
            f.update_xaxes(type="log", title="M [h⁻¹ M☉]")
            f.update_yaxes(type="log", title="σ(M)")
            set_plot(f, "Same P(k), different smoothing window", "This isolates the modeling choice W(kR) while holding the completed matter spectrum fixed.")
            chart(f, 540, "atlas_sigma", axis_controls=True)
    else:
        st.dataframe(pd.DataFrame([{"fit": n, "published range": FIT_METADATA[n][0], "redshift": FIT_METADATA[n][1], "calibration": FIT_METADATA[n][2]} for n in FITTING_NAMES]), width="stretch", hide_index=True)
        st.warning("A fit is publication-ready only when its halo definition, overdensity, cosmology, redshift, and calibrated range match your analysis.")


def workspace_zip():
    out = BytesIO()
    with ZipFile(out, "w", ZIP_DEFLATED) as zf:
        for folder in (run_storage.RUN_DIR, run_storage.EXPORT_DIR, run_storage.STATE_DIR):
            if folder.exists():
                for path in folder.rglob("*"):
                    if path.is_file() and path.name != ".gitkeep":
                        zf.write(path, path.relative_to(run_storage.DATA_ROOT).as_posix())
    return out.getvalue()


def restore_workspace(payload):
    count = 0; run_storage.DATA_ROOT.mkdir(parents=True, exist_ok=True)
    with ZipFile(BytesIO(payload)) as zf:
        for info in zf.infolist():
            path = Path(info.filename)
            if info.is_dir() or path.is_absolute() or ".." in path.parts or not path.parts or path.parts[0] not in {"saved_runs", "exports", "state"}:
                continue
            target = run_storage.DATA_ROOT / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(zf.read(info)); count += 1
    return count


def runs_view():
    st.markdown(
        '<div class="page-head"><span>RUN VAULT</span>'
        '<h2>Every completed universe is already saved.</h2>'
        '<p>Runs are written atomically to the host project’s data folder '
        'through Docker Compose, so refreshes, browser closes, app restarts, '
        'and container rebuilds do not erase them.</p></div>',
        unsafe_allow_html=True,
    )

    st.code(
        str(run_storage.DATA_ROOT),
        language="text",
    )

    upload = st.file_uploader(
        "Restore workspace",
        type=["zip"],
    )

    if upload and st.button("Restore now"):
        try:
            count = restore_workspace(
                upload.getvalue()
            )

            restored = load_all_runs()

            if restored:
                load_run_into_session(
                    restored[-1]["run_id"]
                )
                clear_widgets()

            st.success(
                f"Restored {count} files"
            )

            st.rerun()

        except Exception as exc:
            st.error(str(exc))

    runs = load_all_runs()

    if not runs:
        st.info("No saved runs yet.")
        return

    st.download_button(
        "Download complete workspace",
        workspace_zip(),
        "haloforge_workspace.zip",
        "application/zip",
        type="primary",
    )

    run_map = {
        str(run["run_id"]): run
        for run in runs
    }

    run_ids = list(run_map)

    preferred_run_id = next(
        (
            run_id
            for run_id in run_ids
            if run_map[run_id].get("is_baseline")
        ),
        run_ids[-1],
    )

    _prepare_scalar_widget_state(
        "run_vault_selected_id",
        run_ids,
    )

    chosen_kwargs = {}

    if "run_vault_selected_id" not in st.session_state:
        chosen_kwargs["index"] = run_ids.index(
            preferred_run_id
        )

    chosen_id = st.selectbox(
        "Selected run",
        run_ids,
        format_func=lambda run_id: get_run_label(
            run_map[run_id]
        ),
        key="run_vault_selected_id",
        **chosen_kwargs,
    )

    chosen = run_map[chosen_id]
    arrays = chosen["arrays"]

    rename_key = f"rename_{chosen_id}"

    rename_value = st.text_input(
        "Rename selected run",
        value=chosen["name"],
        key=rename_key,
    )

    actions = st.columns(4)

    if actions[0].button(
        "Load",
        width="stretch",
    ):
        if load_run_into_session(chosen_id):
            clear_widgets()
            st.rerun()

    if actions[1].button(
        "Rename",
        width="stretch",
    ):
        rename_run(
            chosen_id,
            rename_value,
        )
        st.rerun()

    if actions[2].button(
        "Duplicate",
        width="stretch",
    ):
        duplicate_run(chosen_id)
        st.rerun()

    if actions[3].button(
        "Set baseline",
        width="stretch",
    ):
        set_baseline(chosen_id)
        st.rerun()

    downloads = st.columns(5)

    downloads[0].download_button(
        "P(k,z) CSV",
        pd.DataFrame(
            {
                "k_Mpc^-1": arrays["k"],
                **{
                    f"P_z{z:g}_Mpc3": (
                        arrays["P_by_z"][i]
                    )
                    for i, z in enumerate(
                        arrays.get(
                            "redshifts",
                            [0],
                        )
                    )
                },
            }
        ).to_csv(index=False),
        "power_spectra.csv",
        "text/csv",
        width="stretch",
    )

    downloads[1].download_button(
        "σ(M,z) CSV",
        pd.DataFrame(
            {
                "M_hinv_Msun": arrays["M_h"],
                "M_Msun": arrays["M"],
                "R_Mpc": arrays["R"],
                **{
                    f"sigma_z{z:g}": (
                        arrays["sigma_by_z"][i]
                    )
                    for i, z in enumerate(
                        arrays.get(
                            "redshifts",
                            [0],
                        )
                    )
                },
            }
        ).to_csv(index=False),
        "sigma_mass.csv",
        "text/csv",
        width="stretch",
    )

    downloads[2].download_button(
        "Parameters",
        json.dumps(
            chosen["params"],
            indent=2,
        ),
        "parameters.json",
        "application/json",
        width="stretch",
    )

    downloads[3].download_button(
        "CLASS settings",
        json.dumps(
            chosen.get(
                "class_settings",
                {},
            ),
            indent=2,
        ),
        "class_settings.json",
        "application/json",
        width="stretch",
    )

    export_zip = Path(
        chosen.get(
            "exports",
            {},
        ).get(
            "exports.zip",
            "",
        )
    )

    if export_zip.is_file():
        downloads[4].download_button(
            "Run bundle",
            export_zip.read_bytes(),
            export_zip.name,
            "application/zip",
            width="stretch",
        )

    st.dataframe(
        pd.DataFrame(
            [
                {
                    "name": run["name"],
                    "baseline": run.get(
                        "is_baseline"
                    ),
                    "backend": run.get(
                        "class_status"
                    ),
                    "created": run.get(
                        "created_at"
                    ),
                    "EDE": run["params"].get(
                        "enable_ede"
                    ),
                    "fEDE": run["params"].get(
                        "f_EDE"
                    ),
                    "σ₈": run.get("sigma8"),
                }
                for run in runs
            ]
        ),
        width="stretch",
        hide_index=True,
    )

    if st.button(
        "Delete selected run",
        type="secondary",
    ):
        deleted_was_loaded = (
            st.session_state.get(
                "loaded_run_id"
            )
            == chosen_id
        )

        deleted_was_baseline = bool(
            chosen.get("is_baseline")
        )

        delete_run(chosen_id)

        st.session_state.pop(
            "run_vault_selected_id",
            None,
        )

        st.session_state.pop(
            rename_key,
            None,
        )

        remaining = load_all_runs()

        if (
            remaining
            and deleted_was_baseline
            and not any(
                run.get("is_baseline")
                for run in remaining
            )
        ):
            set_baseline(
                remaining[0]["run_id"]
            )
            remaining = load_all_runs()

        if deleted_was_loaded:
            if remaining:
                load_run_into_session(
                    remaining[-1]["run_id"]
                )
                clear_widgets()
            else:
                clear_loaded_run()

        st.rerun()


def diagnostics_view():
    st.markdown('<div class="page-head"><span>VALIDATION</span><h2>Inspect every assumption.</h2></div>', unsafe_allow_html=True)
    diag = environment_diagnostics(); ok = diag["classy_imports"]
    st.markdown(f'<div class="diagnostic {"pass" if ok else "fail"}"><i></i><b>{"classy is importable" if ok else "classy unavailable"}</b><span>{diag.get("classy_path") or diag.get("classy_error")}</span></div>', unsafe_allow_html=True)
    st.info("Every CLASS solve runs in an isolated child process. A native AxiCLASS failure can end that worker, but it cannot take down the Streamlit app or overwrite the last completed run.")
    if st.button("Run real CLASS smoke test", type="primary"):
        with st.spinner("Running minimal ΛCDM solve in an isolated worker…"):
            smoke = tiny_class_smoke_test(get_params())
        (st.success if smoke["ok"] else st.error)(smoke["message"])
    ready = require_run()
    if ready:
        run, result, sigma = ready; rows = []
        for i, z in enumerate(result["redshifts"]):
            d = float(result["growth_class"][i])
            p_growth = float(np.sqrt(np.interp(0.01, result["k"], result["P_by_z"][i]) / np.interp(0.01, result["k"], result["P_by_z"][0])))
            rows.append({"z": z, "σ8 pipeline": sigma["sigma8_pipeline_by_z"][i], "D CLASS": d, "D from P(k=.01)": p_growth, "fractional difference": p_growth / d - 1})
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        chart(growth_fig(result, sigma), 440, "diag_growth")
        coverage = sigma.get("coverage")
        if coverage:
            st.json(coverage)
    a, b = st.columns(2)
    with a:
        st.subheader("Runtime"); st.json(diag)
    with b:
        st.subheader("Next exact CLASS settings"); st.code(json.dumps(build_class_settings(get_params()), indent=2), language="json")


section = sidebar()
{
    "Dashboard": dashboard_view,
    "Graph studio": graph_studio_view,
    "Compare lab": compare_view,
    "Structure field": structure_view,
    "Learn the pipeline": learn_view,
    "Fit + window atlas": atlas_view,
    "Runs + export": runs_view,
    "Diagnostics": diagnostics_view,
}[section]()
