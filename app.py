"""HaloForge — stable interactive AxiCLASS structure laboratory."""

from __future__ import annotations

import json
import os
from html import escape
from copy import deepcopy
from io import BytesIO
from pathlib import Path
from time import perf_counter
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config.defaults import DEFAULT_PARAMS
from config.ranges import CONTROL_RANGES
from engine.class_runner import (
    ClassRuntimeError,
    build_class_settings,
    compute_matter_power,
    environment_diagnostics,
    tiny_class_smoke_test,
)
from engine.cosmology import derived_quantities
from engine.comparison import compare_at_point, transform_curve
from engine.onboarding import (
    guided_parameter_pair,
    CLUSTER_MASS_HINV_MSUN,
    REVEAL_STAGES,
    committed_prediction,
    halo_abundance_change,
    reveal_stage,
)
from engine.fitting_functions import FITTING_NAMES, FIT_METADATA, fitting_values
from engine.contracts import MASS_DEFINITIONS, fit_contract
from engine.hmf import cumulative_hmf, hmf_z
from engine.redshift import redshift_index
from engine.sigma import sigma_grid, sigma_integrand_per_logk
from engine.windows import top_hat_W_exact, top_hat_W_series, window_W, window_squared
from engine.explanations import explain_change
from engine.assurance import assurance_report
from engine.validity import scientific_validity_record
from engine.figure_transcript import chart_transcript
from engine.inspection import inspect_k_range, inspect_mass_point
from engine.sensitivity import controlled_sensitivity, smallest_saved_counterfactual
from engine.benchmark import (
    CANONICAL_VALIDATION_CASES,
    CANONICAL_CASES_VERSION,
    assess_canonical_case,
    canonical_case,
    canonical_case_rows,
    internal_sigma8_benchmark,
)
from engine.saved_run import pipeline_from_saved_run
from engine.performance import PERFORMANCE_BENCHMARK_VERSION, profile_core_pipeline
from engine.figure_recipes import RECIPES, apply_figure_recipe, apply_chart_theme
from engine.navigation import command_index, search_commands
from engine.uncertainty import uncertainty_inventory
from engine.experiment_design import PLANS, design_experiment
from engine.plot_insights import (
    largest_deviation_point,
    peak_point,
    reference_crossings,
    turning_points,
    validity_boundaries,
)
from engine.hover_context import hover_context, with_hover_context
from engine.chart_selection import selected_x_bounds
from engine.failure_taxonomy import classify_failure
from engine.fingerprint import cosmic_fingerprint
from engine.structure import (
    gaussian_field_slice as _gaussian_field_slice,
    shared_fourier_seed as _shared_fourier_seed,
)
from content.modules import (
    MODULES,
    guided_experiment_for_module,
    get_module,
    instructor_guide,
    lecture_outline,
    lecture_slides,
    notebook_template,
    student_handout,
    teaching_bundle,
)
from content.lab_sections import (
    instructor_section_bundle,
    local_lab_section,
    student_section_bundle,
)
from content.concepts import CONCEPTS, concept_by_label
from content.limitations import LIMITATIONS_VERSION, limitations_rows
from content.skepticism import EXERCISES, get_exercise
import state.run_storage as run_storage
from state.campaign_storage import (
    campaign_export_bundle,
    completed_member_timings,
    list_campaign_ids,
    load_campaign,
    save_campaign,
)
from state.catalogue_storage import save_fof_catalogue
from state.run_storage import (
    delete_run,
    duplicate_run,
    generate_run_exports,
    share_card_markdown,
    get_run_label,
    load_all_runs,
    rename_run,
    restore_deleted_run,
    save_draft_params,
    set_baseline,
    update_run_metadata,
)
from state.notebook import (
    annotations_for_chart,
    attached_artifact_rows,
    lineage_rows,
    normalize_notebook_entry,
    parameter_diff,
)
from state.audit import audit_status
from state.storage_policy import (
    UnsafeHostedStorage,
    privacy_statement,
    require_safe_persistent_storage,
)
from state.bundle import (
    BundleValidationError,
    WORKSPACE_MANIFEST,
    import_workspace,
    plan_workspace_import,
    sha256_bytes,
)
from state.observability import (
    clear_local_diagnostics,
    local_diagnostics_enabled,
    local_diagnostics_summary,
    record_local_diagnostic,
    set_local_diagnostics_enabled,
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
from engine.evolution import evolution_redshifts
from engine.evolution_studio import (
    calculate_evolution_frames,
    build_evolution_figure,
    evolution_contact_sheet,
    evolution_frame_summary_table,
)
from engine.evolution_video import VIDEO_FORMATS, render_evolution_video
from engine.scientific_status import (
    CALCULATED_LINEAR_THEORY,
    STATUS_LABELS,
    weakest_status,
)
from engine.campaign import (
    cartesian_campaign,
    estimate_campaign_resources,
    latin_hypercube_campaign,
    sobol_campaign,
)
from engine.campaign_orchestrator import (
    campaign_parallel_coordinates,
    campaign_response_figure,
    campaign_to_dataframe,
    compute_campaign_sensitivities,
    campaign_counts,
    cancel_pending_members,
    create_campaign,
    execute_campaign_step,
    pause_campaign,
    resume_campaign,
    retry_failed_members,
)
from engine.nbody_setup import (
    compute_box_resolution,
)
from engine.gadget4_adapter import (
    run_installation_doctor,
    generate_config_sh,
    generate_gadget4_parameter_file,
    generate_output_times_file,
    GADGET4_VERSION,
    GADGET4_PINNED_COMMIT,
    GADGET4_CITATION,
)
from engine.ic_generator import generate_zeldovich_particles
from engine.halo_catalogue import (
    find_fof_halos,
    catalogue_to_dataframe,
    render_3d_halo_view,
)
from engine.gadget_snapshot import load_gadget4_dm_snapshot
from engine.simulation_manifest import (
    load_and_validate_snapshot_manifest,
    manifest_path_for_snapshot,
    write_snapshot_manifest,
)
from engine.hmf_nbody_comparison import (
    compare_catalogue_to_analytic_hmf,
    render_hmf_comparison_plot,
)

APP_ROOT = Path(__file__).resolve().parent
st.set_page_config(
    page_title="HaloForge",
    page_icon="◉",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(
    f"<style>{(APP_ROOT / 'assets/custom.css').read_text(encoding='utf-8')}</style>",
    unsafe_allow_html=True,
)
try:
    require_safe_persistent_storage()
except UnsafeHostedStorage as exc:
    st.error(str(exc))
    st.info("No runs or drafts have been opened or written on this deployment.")
    st.stop()
init_session()
params = get_params()
if "hf_local_diagnostics_enabled" not in st.session_state:
    st.session_state["hf_local_diagnostics_enabled"] = local_diagnostics_enabled(
        run_storage.DATA_ROOT
    )


def show_failure(exc: BaseException, *, preserve_note: str | None = None) -> None:
    """Render an actionable failure without suppressing its technical context."""
    failure = classify_failure(exc)
    st.error(f"{failure.title}. {failure.summary}")
    st.info("Next step: " + failure.remedy)
    if preserve_note:
        st.caption(preserve_note)
    with st.expander("Technical detail"):
        st.code(failure.technical_detail, language="text")
    record_local_diagnostic(
        run_storage.DATA_ROOT,
        bool(st.session_state.get("hf_local_diagnostics_enabled", False)),
        "calculation_failed",
    )


COLORS = [
    "#35d7e5",
    "#ffb454",
    "#a78bfa",
    "#ff718b",
    "#45d49d",
    "#67a7ff",
    "#f5df70",
    "#d879ff",
]
WINDOWS = ["Top-hat", "Gaussian", "Sharp-k"]

PARAM_INFO = {
    "A_s": (
        "Primordial amplitude Aₛ",
        "Sets the overall normalization of primordial curvature fluctuations.",
        "Increasing Aₛ shifts the entire primordial spectrum upward at every k; nₛ controls the tilt.",
        "amplitude",
    ),
    "n_s": (
        "Spectral tilt nₛ",
        "Controls the relative primordial power on scales above and below the pivot kₚ.",
        "Changing nₛ rotates the spectrum around kₚ: larger nₛ gives relatively more high-k power.",
        "tilt",
    ),
    "k_pivot": (
        "Pivot scale kₚ",
        "The reference wavenumber where the amplitude Aₛ is defined.",
        "It is an anchor for the power-law parameterization, not a new physical feature in P(k).",
        "pivot",
    ),
    "H0": (
        "Expansion rate H₀",
        "The present expansion rate and the source of h = H₀/100.",
        "Changing H₀ changes physical densities and characteristic scales; the guide shows faster versus slower expansion, not a fake P(k) line.",
        "expansion",
    ),
    "Omega_m": (
        "Matter density Ωₘ",
        "Fraction of today's critical density in total matter.",
        "It changes equality, growth, the mean matter density, the M↔R mapping, and halo abundance.",
        "matter",
    ),
    "Omega_b": (
        "Baryon density Ωᵦ",
        "The baryonic part of matter; Ωcdm is computed as Ωₘ−Ωᵦ.",
        "It changes baryon loading and the scale-dependent oscillatory structure in the processed matter spectrum.",
        "baryon",
    ),
    "Omega_k": (
        "Curvature Ωₖ",
        "Contribution of spatial curvature to the background geometry.",
        "Zero is spatially flat; positive and negative values change geometry, distances, and growth.",
        "curvature",
    ),
    "N_eff": (
        "Massless relativistic species N_ur",
        "Passed directly to CLASS/AxiCLASS as N_ur and used for the radiation-density estimate.",
        "This is not a full massive-neutrino-sector model. More massless radiation delays equality and changes early-time processing.",
        "radiation",
    ),
    "Tcmb": (
        "CMB temperature Tcmb",
        "Sets today's photon temperature and therefore the photon energy density.",
        "A higher Tcmb raises radiation density and shifts early expansion and equality.",
        "temperature",
    ),
    "tau_reio": (
        "Optical depth τ",
        "Integrated probability that CMB photons rescattered after reionization.",
        "It is central for CMB observables; its direct influence on the linear matter spectrum is limited in this app.",
        "reionization",
    ),
    "f_EDE": (
        "EDE peak fraction fEDE",
        "Maximum temporary early-dark-energy fraction near the critical epoch.",
        "A larger value raises the height of the early expansion pulse; it does not act like dark energy today.",
        "ede_amp",
    ),
    "log10_a_c": (
        "EDE epoch log₁₀aᶜ",
        "Locates when the axion-like field becomes dynamical.",
        "More negative values move the temporary EDE pulse earlier in cosmic history.",
        "ede_epoch",
    ),
    "k_min": (
        "Minimum k",
        "Largest Fourier scale retained in the sampled spectrum.",
        "Raising kmin discards large-scale modes and can bias very large smoothing radii.",
        "kmin",
    ),
    "k_max": (
        "Maximum k",
        "Smallest Fourier scale retained in the sampled spectrum.",
        "Low-mass halos need sufficiently high kmax for converged σ(M).",
        "kmax",
    ),
    "k_points": (
        "k samples",
        "Number of logarithmically spaced AxiCLASS P(k) samples.",
        "More points resolve shape and stabilize fixed-grid integration, but require more backend evaluations.",
        "samples",
    ),
    "quad_limit": (
        "Integration batch size",
        "Safe vectorization batch used by the fixed log-k Simpson integrator.",
        "It controls memory batching only; the complete sampled CLASS k grid is always integrated.",
        "accuracy",
    ),
    "delta_c": (
        "Collapse threshold δc",
        "Linear overdensity threshold used by multiplicity models.",
        "A higher threshold makes collapse rarer and suppresses the high-mass tail.",
        "threshold",
    ),
    "mass_min_exp": (
        "Minimum halo mass",
        "Lower log₁₀ mass bound in h⁻¹M☉.",
        "Smaller masses correspond to smaller smoothing radii and greater sensitivity to high k.",
        "mass_range",
    ),
    "mass_max_exp": (
        "Maximum halo mass",
        "Upper log₁₀ mass bound in h⁻¹M☉.",
        "Larger masses probe rare peaks and the exponentially falling HMF tail.",
        "mass_range",
    ),
    "mass_points": (
        "Mass samples",
        "Number of logarithmic M samples used for σ(M), its derivative, and the HMF.",
        "A denser grid stabilizes numerical derivatives and curve detail.",
        "samples",
    ),
    "selected_mass_exp": (
        "Inspection mass",
        "Mass highlighted in the σ integrand and learning views.",
        "The marker moves across the mass range and reveals which k modes contribute to that scale.",
        "inspect_mass",
    ),
    "single_z": (
        "Analysis redshift z",
        "Epoch used by focused σ and HMF diagnostics.",
        "At larger z, linear growth is smaller and massive halos are much rarer.",
        "redshift",
    ),
}

VISUALS = {
    "amplitude": """<svg viewBox="0 0 300 105" aria-label="A_s shifts the primordial spectrum vertically"><path class="axis" d="M28 12V84H282"/><path class="guide" d="M38 38C102 44 184 54 272 64"/><path class="accent" d="M38 22C102 28 184 38 272 48"/><path class="arrow" d="M238 61V49M233 54l5-5 5 5"/><text x="41" y="18">higher Aₛ</text><text x="41" y="52">lower Aₛ</text><text x="244" y="99">log k</text><text x="4" y="17">𝒫ℛ</text></svg>""",
    "tilt": """<svg viewBox="0 0 300 105" aria-label="n_s pivots the primordial spectrum"><path class="axis" d="M28 12V84H282"/><path class="guide" d="M38 25L272 66"/><path class="accent" d="M38 66L272 28"/><line class="marker" x1="155" y1="12" x2="155" y2="84"/><circle class="dot" cx="155" cy="46" r="4"/><text x="162" y="18">pivot kₚ</text><text x="205" y="27">larger nₛ</text><text x="205" y="74">smaller nₛ</text></svg>""",
    "pivot": """<svg viewBox="0 0 300 105"><path class="axis" d="M28 12V84H282"/><path class="accent" d="M38 28C105 33 188 46 272 65"/><line class="marker" x1="130" y1="12" x2="130" y2="84"/><line class="marker amber" x1="205" y1="12" x2="205" y2="84"/><circle class="dot" cx="130" cy="40" r="4"/><circle class="dot amber-fill" cx="205" cy="51" r="4"/><text x="100" y="98">different anchors, same law</text></svg>""",
    "expansion": """<svg viewBox="0 0 300 105" aria-label="H0 changes present expansion rate"><circle class="universe slow" cx="82" cy="52" r="20"/><circle class="universe fast" cx="215" cy="52" r="20"/><path class="arrow" d="M82 18V5M77 10l5-5 5 5M82 86v13M77 94l5 5 5-5M48 52H35M40 47l-5 5 5 5M116 52h13M124 47l5 5-5 5"/><path class="arrow amber" d="M215 12V2M210 7l5-5 5 5M215 92v11M210 98l5 5 5-5M175 52h-15M165 47l-5 5 5 5M255 52h15M265 47l5 5-5 5"/><text x="50" y="100">lower H₀</text><text x="187" y="100">higher H₀</text></svg>""",
    "matter": """<svg viewBox="0 0 300 105"><path class="axis" d="M28 12V84H282"/><path class="guide" d="M38 78C65 38 92 20 123 25C168 32 211 56 272 75"/><path class="accent" d="M38 79C79 31 115 18 153 28C195 39 229 59 272 73"/><line class="marker" x1="123" y1="18" x2="123" y2="84"/><line class="marker amber" x1="153" y1="18" x2="153" y2="84"/><text x="68" y="99">equality/turnover shifts</text></svg>""",
    "baryon": """<svg viewBox="0 0 300 105"><path class="axis" d="M28 12V84H282"/><path class="guide" d="M38 64C65 44 81 64 106 48S146 62 169 48S211 59 236 48S260 52 272 48"/><path class="accent" d="M38 58C63 31 82 69 106 39S147 67 170 39S213 65 237 39S261 51 272 39"/><text x="77" y="97">baryon loading changes wiggles</text></svg>""",
    "curvature": """<svg viewBox="0 0 300 105"><path class="guide" d="M30 72Q82 22 134 72"/><path class="accent" d="M164 72Q216 112 268 72"/><path class="axis" d="M30 52H134M164 52H268"/><text x="55" y="93">closed-like</text><text x="197" y="93">open-like</text><text x="126" y="16">Ωₖ=0 is flat</text></svg>""",
    "radiation": """<svg viewBox="0 0 300 105"><path class="axis" d="M25 55H280"/><circle class="dot" cx="88" cy="55" r="5"/><circle class="dot amber-fill" cx="154" cy="55" r="5"/><path class="arrow" d="M88 35H154M148 30l6 5-6 5"/><text x="39" y="26">less radiation</text><text x="153" y="26">more radiation</text><text x="63" y="82">equality</text><text x="132" y="82">later equality</text></svg>""",
    "temperature": """<svg viewBox="0 0 300 105"><path class="guide" d="M28 42C46 16 64 68 82 42S118 16 136 42"/><path class="accent" d="M164 42C181 8 198 76 215 42S249 8 266 42"/><circle class="dot" cx="82" cy="76" r="8"/><circle class="dot amber-fill" cx="215" cy="76" r="13"/><text x="34" y="99">lower photon density</text><text x="169" y="99">higher photon density</text></svg>""",
    "reionization": """<svg viewBox="0 0 300 105"><path class="accent" d="M20 52H278"/><circle class="dot" cx="105" cy="52" r="6"/><circle class="dot amber-fill" cx="190" cy="52" r="6"/><path class="guide" d="M105 52L72 23M190 52L224 24"/><text x="52" y="94">larger τ → more rescattering</text></svg>""",
    "ede_amp": """<svg viewBox="0 0 300 105"><path class="axis" d="M28 12V84H282"/><path class="guide" d="M35 80C92 80 102 76 120 43S147 80 276 80"/><path class="accent" d="M35 80C92 80 101 75 120 16S148 80 276 80"/><text x="82" y="99">temporary early pulse</text><text x="136" y="21">larger fEDE</text></svg>""",
    "ede_epoch": """<svg viewBox="0 0 300 105"><path class="axis" d="M25 82H282"/><path class="guide" d="M35 80C58 80 64 72 78 23S101 80 130 80"/><path class="accent" d="M155 80C178 80 184 72 198 23S221 80 274 80"/><path class="arrow" d="M92 13H185M179 8l6 5-6 5"/><text x="45" y="99">earlier</text><text x="225" y="99">later</text></svg>""",
    "kmin": """<svg viewBox="0 0 300 105"><path class="axis" d="M28 12V84H282"/><path class="accent" d="M35 70C80 35 128 25 178 39S240 65 275 72"/><rect class="shade" x="30" y="12" width="70" height="72"/><line class="marker" x1="100" y1="12" x2="100" y2="84"/><text x="36" y="99">discarded largest scales</text></svg>""",
    "kmax": """<svg viewBox="0 0 300 105"><path class="axis" d="M28 12V84H282"/><path class="accent" d="M35 70C80 35 128 25 178 39S240 65 275 72"/><rect class="shade" x="220" y="12" width="58" height="72"/><line class="marker" x1="220" y1="12" x2="220" y2="84"/><text x="119" y="99">discarded smallest scales</text></svg>""",
    "samples": """<svg viewBox="0 0 300 105"><path class="guide" d="M25 76C82 22 158 24 278 66"/><g class="sparse"><circle cx="25" cy="76" r="4"/><circle cx="90" cy="31" r="4"/><circle cx="176" cy="31" r="4"/><circle cx="278" cy="66" r="4"/></g><g class="dense"><circle cx="48" cy="56" r="2"/><circle cx="70" cy="41" r="2"/><circle cx="112" cy="26" r="2"/><circle cx="140" cy="25" r="2"/><circle cx="205" cy="39" r="2"/><circle cx="244" cy="54" r="2"/></g><text x="65" y="99">denser sampling follows the same curve</text></svg>""",
    "accuracy": """<svg viewBox="0 0 300 105"><path class="axis" d="M28 12V84H282"/><path class="accent" d="M34 78C72 42 108 26 150 35S225 71 276 74"/><path class="bars" d="M45 78V70M65 78V57M85 78V45M105 78V36M125 78V33M145 78V35M165 78V42M185 78V51M205 78V61M225 78V68M245 78V72"/><text x="56" y="99">complete log-k grid, safe batches</text></svg>""",
    "threshold": """<svg viewBox="0 0 300 105"><path class="axis" d="M25 82H282"/><path class="accent" d="M28 82C77 82 89 18 150 18S225 82 278 82"/><line class="marker" x1="214" y1="14" x2="214" y2="82"/><path class="tail" d="M214 62C233 75 255 81 278 82V82H214Z"/><text x="201" y="11">δc</text><text x="221" y="99">collapse tail</text></svg>""",
    "mass_range": """<svg viewBox="0 0 300 105"><path class="axis" d="M28 55H282"/><line class="marker" x1="75" y1="25" x2="75" y2="82"/><line class="marker amber" x1="238" y1="25" x2="238" y2="82"/><path class="arrow" d="M75 34H238M81 29l-6 5 6 5M232 29l6 5-6 5"/><text x="39" y="99">dwarf</text><text x="135" y="99">galaxy/group</text><text x="235" y="99">cluster</text></svg>""",
    "inspect_mass": """<svg viewBox="0 0 300 105"><path class="axis" d="M28 55H282"/><line class="marker moving" x1="158" y1="18" x2="158" y2="82"/><circle class="dot" cx="158" cy="55" r="6"/><text x="87" y="99">selected M chooses smoothing R</text></svg>""",
    "redshift": """<svg viewBox="0 0 300 105"><path class="axis" d="M28 12V84H282"/><path class="accent" d="M35 28C105 34 185 48 275 70"/><path class="guide" d="M35 48C105 53 185 65 275 78"/><text x="40" y="23">today: more growth</text><text x="151" y="96">higher z: less growth</text></svg>""",
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
    "Cumulative HMF": "Number density between M and the largest sampled mass. Positive intervals use exact integration of a power-law interpolant; intervals touching zero use linear integration in ln M.",
}

GRAPH_LIMITS = {
    "Primordial spectrum": "This is the input parameterization, not a direct observation of initial conditions.",
    "Matter P(k)": "Linear P(k) is not a nonlinear matter field or a galaxy survey prediction.",
    "Dimensionless Δ²(k)": "A large contribution per log interval does not by itself establish numerical convergence.",
    "Processing shape": "The normalized shape proxy is explanatory; it is not an independently fitted transfer function.",
    "Window W": "The window is a mathematical weighting kernel, not a literal boundary around a halo.",
    "Window W²": "Positive W² contributions do not make a finite sampled k range complete.",
    "Top-hat series error": "Agreement of two algebraic evaluations checks numerical stability only near y=0.",
    "Mass–radius map": "The mapping assumes the displayed mean density and a top-hat convention.",
    "σ(M)": "σ(M) is a linear-theory variance, not a measured halo catalogue.",
    "σ slope": "Numerical derivatives can be sensitive to mass sampling and endpoint behavior.",
    "σ integrand": "The displayed band only covers modes included in the computed CLASS grid.",
    "Growth": "Agreement of these two curves does not independently validate the solver or physical model.",
    "Multiplicity f(σ)": "Different fits encode different assumptions and calibration domains.",
    "HMF": "An HMF fit is empirical; dotted or unsupported regions are not publication-ready.",
    "Cumulative HMF": "The integral stops at the largest sampled mass; it is not an integral to infinity.",
}


def set_plot(fig: go.Figure, title: str, caption: str) -> go.Figure:
    fig.update_layout(
        title=title,
        meta={
            "caption": caption,
            "mislead": GRAPH_LIMITS.get(
                title,
                "Inspect the stated assumptions, sampled domain, and numerical diagnostics before treating this visual as evidence.",
            ),
        },
    )
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
    data_lo = (
        float(np.nanmin(current[current > 0]))
        if axis_type == "log" and np.any(current > 0)
        else float(np.nanmin(current))
    )
    data_hi = float(np.nanmax(current))
    lo = data_lo if lo is None else lo
    hi = data_hi if hi is None else hi
    if hi <= lo or (axis_type == "log" and lo <= 0):
        return
    target = (
        [float(np.log10(lo)), float(np.log10(hi))] if axis_type == "log" else [lo, hi]
    )
    (fig.update_xaxes if axis == "x" else fig.update_yaxes)(
        range=target, autorange=False
    )


def chart(
    fig: go.Figure,
    height: int = 430,
    key: str | None = None,
    axis_controls: bool = False,
) -> None:
    title = str(fig.layout.title.text or "")
    caption = ""
    mislead = ""
    if isinstance(fig.layout.meta, dict):
        caption = str(fig.layout.meta.get("caption", ""))
        mislead = str(fig.layout.meta.get("mislead", ""))
    cursor_context = hover_context(title)
    for trace in fig.data:
        trace.hovertemplate = with_hover_context(
            getattr(trace, "hovertemplate", None), cursor_context
        )
    current_run_id = st.session_state.get("current_run_id")
    if current_run_id:
        saved = run_storage.load_run(str(current_run_id))
        if saved:
            for annotation in annotations_for_chart(saved.get("notebook"), title):
                fig.add_vrect(
                    x0=annotation["x_start"],
                    x1=annotation["x_end"],
                    fillcolor="#ffb454",
                    opacity=0.16,
                    line_width=1,
                    line_color="#ffb454",
                    annotation_text=str(annotation["text"]),
                    annotation_position="top left",
                )
    show_legend = (
        sum(1 for trace in fig.data if getattr(trace, "showlegend", True) is not False)
        > 1
    )
    # Keep the legend outside the data rectangle. Long immutable run names are
    # common in comparison figures; the former small margin let them compete
    # with the x-axis at narrow analysis widths.
    bottom = 142 if show_legend else 62
    fig.update_layout(
        height=height,
        margin=dict(l=70, r=28, t=84, b=bottom),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#091218",
        font=dict(family="IBM Plex Sans, sans-serif", color="#c9d3d9", size=12),
        title=dict(
            text=title,
            x=0.02,
            xanchor="left",
            y=0.97,
            yanchor="top",
            font=dict(size=17 if len(title) < 54 else 14, color="#f5f1e8"),
        ),
        hovermode="closest",
        hoverlabel=dict(bgcolor="#111d23", bordercolor="#4d626b", font_color="#f5f7f4"),
        legend=dict(
            bgcolor="rgba(9,18,24,.92)",
            bordercolor="#2b3d45",
            borderwidth=1,
            orientation="h",
            y=-0.34,
            yanchor="top",
            x=0,
            xanchor="left",
            font=dict(size=9),
            tracegroupgap=4,
        ),
        showlegend=show_legend,
    )
    fig.update_xaxes(
        gridcolor="#1c2a31",
        zerolinecolor="#3b4a52",
        showspikes=True,
        spikemode="across",
        spikesnap="cursor",
        spikecolor="#74878f",
        automargin=True,
        exponentformat="power",
    )
    fig.update_yaxes(
        gridcolor="#1c2a31",
        zerolinecolor="#3b4a52",
        showspikes=True,
        spikemode="across",
        spikesnap="cursor",
        spikecolor="#74878f",
        automargin=True,
        exponentformat="power",
    )

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

    recipe_key = f"{key or title}_figure_recipe"
    with st.popover("Figure recipe"):
        recipe = st.selectbox("Presentation", RECIPES, key=recipe_key)
        st.caption(
            {
                "Interactive": "Keep Plotly controls and the current app presentation.",
                "Paper draft": "High-contrast white layout with readable publication-style typography. Verify caption, citations, fit validity, and external validation before publication.",
                "Lecture": "Large labels for projector use.",
                "Student explanation": "Cursor-aligned hover for narrated reading.",
                "Share card": "Square composition for an honest, caveat-carrying card.",
            }[recipe]
        )
    apply_chart_theme(fig, st.session_state.get("hf_theme", "Dark"))
    apply_figure_recipe(fig, recipe)

    if axis_controls and key:
        with st.popover("Configure axes"):
            st.caption(f"Adjust scale and range for **{title}**")
            mode = st.selectbox(
                "Range mode",
                ["Best default", "Full data", "Custom"],
                key=f"{key}_range_mode",
            )
            axes = st.columns(2)
            with axes[0]:
                xtype = st.selectbox(
                    "x scale", ["Keep", "Linear", "Log"], key=f"{key}_xtype"
                )
            with axes[1]:
                ytype = st.selectbox(
                    "y scale", ["Keep", "Linear", "Log"], key=f"{key}_ytype"
                )
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

    chart_kwargs = {
        "width": "stretch",
        "key": key,
        "config": {
            "displaylogo": False,
            "responsive": True,
            "scrollZoom": True,
            "modeBarButtonsToAdd": ["drawline", "eraseshape"],
            "toImageButtonOptions": {
                "format": "svg" if recipe == "Paper draft" else "png",
                "scale": 2,
            },
        },
    }
    if key:
        chart_kwargs.update({"on_select": "rerun", "selection_mode": ("box", "lasso")})
    plot_event = st.plotly_chart(fig, **chart_kwargs)
    brushed_bounds = selected_x_bounds(plot_event) if key else None
    if brushed_bounds and current_run_id:
        low, high = brushed_bounds
        st.caption(
            f"Brushed x-range: {low:.5g} to {high:.5g}. Add an explicit note to preserve it with this run."
        )
        selection_note = st.text_input(
            "Note for brushed region",
            placeholder="Example: Verify sampled high-k coverage here",
            key=f"{key}_brushed_note",
        )
        if st.button(
            "Save brushed region to Notebook",
            key=f"{key}_save_brush",
            disabled=not selection_note.strip(),
        ):
            saved = run_storage.load_run(str(current_run_id))
            if saved:
                notebook = normalize_notebook_entry(saved.get("notebook"))
                notebook["annotations"] = notebook["annotations"] + [
                    {
                        "text": selection_note.strip(),
                        "chart_title": title,
                        "x_start": low,
                        "x_end": high,
                        "created_at": pd.Timestamp.utcnow().isoformat(),
                    }
                ]
                saved["notebook"] = notebook
                update_run_metadata(saved)
                generate_run_exports(saved)
                st.success("Brushed region saved with this run and its export bundle.")
    if recipe == "Paper draft":
        st.caption(
            "The chart toolbar’s image action exports SVG in this recipe. Verify the resulting figure, citation, caption, font availability, and scientific validity before publication."
        )
    if caption:
        st.caption(caption)
    if mislead:
        with st.expander("Interpretation limits"):
            st.caption(mislead)
    transcript = chart_transcript(fig)
    with st.expander("Data table & download"):
        st.caption(
            "Plotted samples for inspection, reuse, keyboard access, and CSV export. Use the run bundle for complete provenance and arrays."
        )
        if transcript.empty:
            st.info("This chart has no pointwise data transcript.")
        else:
            st.dataframe(transcript, width="stretch", hide_index=True, height=250)
            st.download_button(
                "Download chart transcript (CSV)",
                transcript.to_csv(index=False),
                f"{title or 'haloforge_chart'}_transcript.csv",
                "text/csv",
                key=f"{key or title}_transcript",
            )


def info_label(key: str, title: str | None = None) -> None:
    heading, meaning, downstream, kind = PARAM_INFO[key]
    # Native disclosure is keyboard and touch operable; the former hover-only
    # card hid teaching content from non-pointer users.
    with st.expander(f"Physics guide — {title or heading}", expanded=False):
        st.caption("PHYSICS GUIDE · schematic, not an evolving simulation")
        st.markdown(f"#### {heading}")
        st.markdown(VISUALS[kind], unsafe_allow_html=True)
        st.caption(meaning)
        st.caption(f"**Downstream:** {downstream}")


def slider(key: str, title: str | None = None) -> None:
    spec = CONTROL_RANGES[key]
    info_label(key, title)
    value = float(params.get(key, spec["default"]))
    integral = isinstance(spec["default"], int) and value.is_integer()
    cast = int if integral else float
    kwargs = dict(
        label=title or key,
        min_value=cast(min(spec["min"], value)),
        max_value=cast(max(spec["max"], value)),
        value=cast(value),
        step=cast(spec["step"]),
        label_visibility="collapsed",
        key=f"hf_{key}",
    )
    if "format" in spec:
        kwargs["format"] = spec["format"]
    params[key] = st.slider(**kwargs)


def clear_widgets() -> None:
    for key in list(st.session_state):
        if key.startswith("hf_"):
            del st.session_state[key]


def apply_accessibility_preferences() -> None:
    """Apply local display choices without relying on system or network state."""
    theme = st.session_state.get("hf_theme", "Dark")
    reduced_motion = bool(st.session_state.get("hf_reduced_motion", False))
    styles = []
    if theme == "Light":
        styles.append("""
        html:root{--ink:#f7f8f6;--panel:#ffffff;--panel2:#f0f3f2;--line:#b7c2c3;--paper:#132126;--muted:#42545a;--cyan:#006f7a;--amber:#8a4b00;--rose:#ae1742;--green:#087543}
        [data-testid="stAppViewContainer"]{background-image:none!important;background:#f7f8f6!important}
        [data-testid="stSidebar"]{background:#edf2f1!important;border-right:1px solid #c9d5d6!important}
        [data-testid="stHeader"]{background:rgba(247,248,246,.9)!important}
        .hero,.lesson-card,.empty,div[data-testid="stMetric"],div[data-testid="stExpander"]{background:#fff!important;border:1px solid #c9d5d6!important}
        .hero h1,.page-head h2,.empty b{color:#132126!important}
        [data-testid="stAlert"]{background:#eef5f4!important;border:1px solid #95b8b8!important;color:#132126!important}
        [data-testid="stAlert"] *{color:#132126!important}
        .js-plotly-plot{filter:none}
        [data-testid="stDataFrame"],[data-testid="stTable"]{background:#ffffff!important;color:#132126!important;border-color:#b7c2c3!important}
        [data-testid="stDataFrame"] *{color:#132126!important}
        [data-testid="stPopoverBody"]{background:#ffffff!important;color:#132126!important;border:1px solid #b7c2c3!important;box-shadow:0 10px 30px rgba(0,0,0,0.15)!important}
        [data-testid="stPopoverBody"] label,[data-testid="stPopoverBody"] p,[data-testid="stPopoverBody"] span{color:#132126!important}
        [data-baseweb="popover"],[data-baseweb="menu"],[role="listbox"]{background:#ffffff!important;color:#132126!important;border:1px solid #b7c2c3!important}
        [data-baseweb="popover"] *{color:#132126}
        [data-baseweb="menu"] li{color:#132126!important}
        [data-baseweb="menu"] li:hover,[data-baseweb="menu"] li[aria-selected="true"]{background-color:#006f7a!important;color:#ffffff!important}
        [data-baseweb="menu"] li:hover *,[data-baseweb="menu"] li[aria-selected="true"] *{color:#ffffff!important}
        [data-baseweb="select"]>div,[data-baseweb="input"],[data-baseweb="base-input"],textarea{background-color:#ffffff!important;color:#132126!important;border-color:#b7c2c3!important}
        button:not([role="tab"]){background-color:#ffffff!important;color:#132126!important;border:1px solid #b7c2c3!important;box-shadow:0 1px 2px rgba(0,0,0,0.05)}
        button:not([role="tab"]):hover{background-color:#f2f7f7!important;border-color:#006f7a!important;color:#006f7a!important}
        .stButton>button[kind="primary"],.stDownloadButton>button[kind="primary"]{background:var(--cyan)!important;color:#ffffff!important;border:none!important}
        .stButton>button[kind="primary"] p,.stDownloadButton>button[kind="primary"] p{color:#ffffff!important}
        [data-testid="stSegmentedControl"]{background:#e3ebea!important;border:1px solid #b7c2c3!important;border-radius:8px!important;margin-bottom:0.75rem!important}
        [data-testid="stSegmentedControl"] button{background:transparent!important;border:none!important;box-shadow:none!important;color:#132126!important}
        [data-testid="stSegmentedControl"] button[aria-checked="true"]{background:#ffffff!important;color:#006f7a!important;font-weight:600!important;box-shadow:0 2px 5px rgba(0,0,0,0.1)!important}
        """)
    elif theme == "High contrast":
        styles.append("""
        html:root{--ink:#000;--panel:#000;--panel2:#000;--line:#fff;--paper:#fff;--muted:#fff;--cyan:#00ffff;--amber:#ffff00;--rose:#ff75a6;--green:#5cff94}
        [data-testid="stAppViewContainer"]{background-image:none!important}.hero,.lesson-card,.empty,div[data-testid="stMetric"],div[data-testid="stExpander"]{background:#000!important;border-color:#fff!important}
        .hero h1,.page-head h2,.empty b{color:#fff!important}.stButton>button{border-color:#fff!important}.stButton>button[kind="primary"]{background:#00ffff!important;color:#000!important}
        [data-testid="stDataFrame"],[data-testid="stTable"]{background:#000000!important;border-color:#ffffff!important}
        [data-testid="stDataFrame"] *{color:#ffffff!important}
        [data-testid="stPopoverBody"]{background:#000000!important;border:1px solid #ffffff!important}
        [data-baseweb="popover"],[data-baseweb="menu"]{background:#000000!important;color:#ffffff!important}
        """)
    if reduced_motion:
        styles.append(
            """*,*::before,*::after{animation-duration:.001ms!important;animation-iteration-count:1!important;scroll-behavior:auto!important;transition-duration:.001ms!important}"""
        )
    st.markdown(f"<style>{''.join(styles)}</style>", unsafe_allow_html=True)


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
    if name in ("LCDM", "ΛCDM"):
        params["enable_ede"] = False
        st.session_state["run_name_draft"] = "Baseline ΛCDM"
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
    # Views can request a parameter stage after sidebar widgets have already
    # existed in a prior render. Apply it before rebuilding any control so a
    # stale widget value cannot silently overwrite the requested case.
    pending_stage = st.session_state.pop("pending_parameter_stage", None)
    if isinstance(pending_stage, dict):
        overrides = pending_stage.get("overrides")
        if isinstance(overrides, dict):
            clear_widgets()
            params.update(overrides)
            st.session_state["run_name_draft"] = str(pending_stage.get("name", ""))
            st.session_state["hf_research_question"] = str(
                pending_stage.get("question", "")
            )
            st.session_state["hf_hypothesis"] = str(pending_stage.get("hypothesis", ""))
            st.session_state["hf_primary_mode"] = "Research"
            st.session_state["hf_research_workspace"] = "Benchmark lab"
            save_draft_params(params)
    pending_navigation = st.session_state.pop("hf_command_navigation", None)
    if isinstance(pending_navigation, dict):
        primary_mode = pending_navigation.get("primary_mode")
        if primary_mode in {"Explore", "Compare", "Research"}:
            st.session_state["hf_primary_mode"] = primary_mode
        workspace = pending_navigation.get("workspace")
        if isinstance(workspace, str):
            if primary_mode == "Compare":
                st.session_state["hf_compare_workspace"] = workspace
            elif primary_mode == "Research":
                st.session_state["hf_research_workspace"] = workspace
    command_run_id = st.session_state.pop("hf_command_load_run_id", None)
    if command_run_id:
        # A command result contains only a durable run ID.  The established
        # session loader performs integrity checks before arrays become active.
        load_run_into_session(str(command_run_id))
    with st.sidebar:
        # A guided outcome may choose a next route.  Apply this before the
        # widgets are instantiated so Streamlit navigates on the next rerun.
        onboarding_route = st.session_state.pop("onboarding_route", None)
        if onboarding_route == "Compare lab":
            st.session_state["hf_primary_mode"] = "Compare"
            st.session_state["hf_compare_workspace"] = onboarding_route
        elif onboarding_route in {"Learn the pipeline", "Diagnostics"}:
            st.session_state["hf_primary_mode"] = "Research"
            st.session_state["hf_research_workspace"] = onboarding_route
        st.markdown(
            '<div class="brand"><i></i><div><b>HALOFORGE</b><small>AXICLASS STRUCTURE LAB</small></div></div>',
            unsafe_allow_html=True,
        )
        recovery_issues = st.session_state.get("draft_recovery_issues", [])
        if recovery_issues:
            st.warning(
                "The saved draft could not be restored. Default controls or the last valid run were loaded. Recovery left the draft file unchanged."
            )
            with st.expander("Draft recovery details"):
                for issue in recovery_issues:
                    st.write(issue)
                if st.button("Dismiss recovery notice"):
                    st.session_state["draft_recovery_issues"] = []
                    st.rerun()
        current_theme = st.session_state.get("hf_theme", "Dark")
        theme_map = {
            "Dark": "🌙 Dark",
            "Light": "☀️ Light",
            "High contrast": "👁️ Contrast",
        }
        rev_theme_map = {v: k for k, v in theme_map.items()}
        current_idx = (
            ["Dark", "Light", "High contrast"].index(current_theme)
            if current_theme in ["Dark", "Light", "High contrast"]
            else 0
        )
        selected_label = st.radio(
            "Color mode",
            ["🌙 Dark", "☀️ Light", "👁️ Contrast"],
            index=current_idx,
            key="hf_theme_radio",
            horizontal=True,
            label_visibility="collapsed",
        )
        selected_theme = rev_theme_map.get(selected_label, "Dark")
        if selected_theme != current_theme:
            st.session_state["hf_theme"] = selected_theme
            st.rerun()

        with st.popover("Display & motion preferences"):
            st.toggle(
                "Reduce motion",
                key="hf_reduced_motion",
                help="Stops decorative and explanatory animation. Every visual remains available as a static view.",
            )
            enabled = st.toggle(
                "Record local operational diagnostics",
                key="hf_local_diagnostics_enabled",
                help="Off by default. Records only local event categories and optional durations—never parameters, run IDs, notes, IP addresses, or network telemetry.",
            )
            if enabled != local_diagnostics_enabled(run_storage.DATA_ROOT):
                set_local_diagnostics_enabled(run_storage.DATA_ROOT, enabled)
            if enabled:
                diagnostic_summary = local_diagnostics_summary(run_storage.DATA_ROOT)
                st.caption(
                    f"Local-only diagnostics: {diagnostic_summary['event_count']} retained events. No network transmission."
                )
                if st.button(
                    "Delete local diagnostic history", key="clear_local_diagnostics"
                ):
                    clear_local_diagnostics(run_storage.DATA_ROOT)
                    st.success("Local diagnostic history deleted.")
        apply_accessibility_preferences()

        def open_command(command: dict) -> None:
            """Apply a palette command at a widget callback boundary."""
            run_id = command.get("run_id")
            if run_id:
                # Defer the session hydration to the next render, before
                # sidebar widgets are constructed.
                st.session_state["hf_command_load_run_id"] = str(run_id)
            # The target is applied at the start of the next script run,
            # before the keyed Streamlit navigation widgets exist.  Writing
            # their keys inside this callback can leave the selector's visible
            # value behind the rendered workspace.
            st.session_state["hf_command_navigation"] = {
                "primary_mode": command["primary_mode"],
                "workspace": command.get("workspace"),
            }
            st.session_state["hf_command_query"] = ""

        with st.expander("Go to…", expanded=False):
            st.caption(
                "Search runs, figures, concepts, equations, experiments, and exports. Tab to a result and press Enter to open it."
            )
            command_query = st.text_input(
                "Search HaloForge",
                key="hf_command_query",
                placeholder="Try ‘halo mass function’ or a run name",
            )
            command_records = command_index(concepts=CONCEPTS, runs=load_all_runs())
            command_results = search_commands(command_query, command_records, limit=6)
            for command in command_results:
                st.button(
                    command["title"],
                    key="hf_command_" + command["identifier"],
                    help=command["detail"],
                    width="stretch",
                    on_click=open_command,
                    args=(command,),
                )
                st.caption(command["detail"])
        mode = st.radio(
            "Primary mode",
            ["Explore", "Compare", "Research"],
            key="hf_primary_mode",
            horizontal=True,
        )
        if mode == "Explore":
            st.caption(
                "Explore guided experiments, compare saved runs, or open the full research workspace."
            )
            return "Explore"
        if mode == "Compare":
            section = st.radio(
                "Compare tools",
                ["Compare lab", "Evolution studio", "Sensitivity explorer"],
                key="hf_compare_workspace",
                label_visibility="collapsed",
            )
            st.caption(
                "Design a controlled comparison, inspect evolution, or explore parameter sensitivity."
            )
        else:
            research_options = [
                "Dashboard",
                "Graph studio",
                "Structure field",
                "Fit + window atlas",
                "Design experiment",
                "Benchmark lab",
                "Performance lab",
                "Convergence lab",
                "Evolution studio",
                "Campaign lab",
                "Simulation lab",
                "Teach",
                "Learn the pipeline",
                "Notebook",
                "Known limitations",
                "Runs + export",
                "Diagnostics",
            ]
            workspace_key = "hf_research_workspace"
            current_sec = st.session_state.get(workspace_key)
            # A fresh Research visit needs Dashboard as its conventional
            # default. A pending quick action has already supplied a valid
            # state value, where passing an index would instead compete with
            # the keyed widget's selection.
            if current_sec not in research_options:
                st.session_state.pop(workspace_key, None)
                workspace_index = 0
            else:
                workspace_index = None
            section = st.selectbox(
                "Research workspace",
                research_options,
                index=workspace_index,
                format_func=lambda s: {
                    "Dashboard": "🔬 Core · Dashboard",
                    "Graph studio": "🔬 Core · Graph Studio",
                    "Structure field": "🔬 Core · Structure Field 3D",
                    "Fit + window atlas": "🔬 Core · Fit + Window Atlas",
                    "Design experiment": "🧪 Labs · One-Change Planner",
                    "Benchmark lab": "🧪 Labs · Benchmark Lab",
                    "Performance lab": "🧪 Labs · Performance Lab",
                    "Convergence lab": "🧪 Labs · Convergence Lab",
                    "Evolution studio": "🌌 HPC · Evolution Studio (z=20→0)",
                    "Campaign lab": "🌌 HPC · Campaign Orchestrator",
                    "Simulation lab": "🌌 HPC · GADGET-4 Simulation Lab",
                    "Teach": "📚 Learn · Guided Pedagogy",
                    "Learn the pipeline": "📚 Learn · Pipeline Tour",
                    "Notebook": "📚 Learn · Research Notebook",
                    "Known limitations": "📚 Learn · Known Limitations",
                    "Runs + export": "💾 System · Saved Runs & Export",
                    "Diagnostics": "💾 System · Diagnostics & Health",
                }.get(s, s),
                key="hf_research_workspace",
            )

        SHOW_COSMOLOGY_FORM_PAGES = {
            "Dashboard",
            "Design experiment",
            "Benchmark lab",
            "Performance lab",
            "Convergence lab",
            "Graph studio",
            "Structure field",
            "Fit + window atlas",
        }
        if section not in SHOW_COSMOLOGY_FORM_PAGES:
            st.markdown(
                f'<div class="preset-label">WORKSPACE CONTEXT</div>'
                f'<div class="mini-readout"><span>Active</span><b>{section}</b></div>',
                unsafe_allow_html=True,
            )
            st.caption(
                "Linear controls are in Dashboard, Design experiment, and Benchmark lab."
            )
            return section

        st.markdown(
            '<div class="preset-label">QUICK UNIVERSES</div>', unsafe_allow_html=True
        )
        pcols = st.columns(2)
        for i, preset in enumerate(["ΛCDM", "EDE", "High amplitude", "Blue tilt"]):
            if pcols[i % 2].button(preset, key=f"preset_{i}", width="stretch"):
                apply_preset(preset)
                st.rerun()
        st.caption(
                "Presets load stable starting values. Nothing expensive runs until you press Calculate & save run."
        )

        with st.form("cosmology_controls", clear_on_submit=False):
            run_name = st.text_input(
                "Run name",
                value=st.session_state.get("run_name_draft", ""),
                key="hf_run_name",
                placeholder="Leave blank for an automatic name",
            )
            with st.expander("Primordial perturbations (Aₛ, nₛ)", expanded=False):
                slider("A_s")
                slider("n_s")
                slider("k_pivot")
            with st.expander("Background cosmology (H₀, Ωₘ...)", expanded=True):
                slider("H0")
                slider("Omega_m")
                slider("Omega_b")
                slider("Omega_k")
                slider("N_eff")
                slider("Tcmb")
                slider("tau_reio")
                d = derived_quantities(params)
                st.markdown(
                    f'<div class="mini-readout"><span>h</span><b>{d["h"]:.4f}</b><span>Ωcdm</span><b>{d["Omega_cdm"]:.4f}</b><span>Ωr</span><b>{d["Omega_r"]:.2e}</b></div>',
                    unsafe_allow_html=True,
                )
            ede_active = bool(params.get("enable_ede", True))
            with st.expander("Axion early dark energy (EDE)", expanded=ede_active):
                params["enable_ede"] = st.toggle(
                    "Enable EDE",
                    value=ede_active,
                    key="hf_enable_ede",
                )
                slider("f_EDE")
                slider("log10_a_c")
                params["n_EDE"] = st.select_slider(
                    "Potential index n",
                    options=sorted({2, 3, 4, 5, 6, int(params.get("n_EDE", 3))}),
                    value=int(params.get("n_EDE", 3)),
                    key="hf_n_EDE",
                )
                params["scf_parameters"] = st.text_input(
                    "Initial field θᵢ, θ̇ᵢ",
                    value=str(params.get("scf_parameters", "2.806,0.0")),
                    key="hf_scf_parameters",
                )
                d = derived_quantities(params)
                st.caption(f"aᶜ = {d['a_c']:.3e}  ·  zᶜ = {d['z_c']:.1f}")
            with st.expander("Halo mass function & fits", expanded=False):
                params["window_type"] = st.selectbox(
                    "Primary window",
                    WINDOWS,
                    index=WINDOWS.index(params.get("window_type", "Top-hat")),
                    key="hf_window_type",
                )
                if params["window_type"] != "Top-hat":
                    st.info(
                        "Variance exploration: masses label equivalent top-hat radii. This window does not produce calibrated halo counts; HMF panels and exports will be unavailable."
                    )
                params["fitting"] = st.selectbox(
                    "Primary HMF fit",
                    FITTING_NAMES,
                    index=FITTING_NAMES.index(params.get("fitting", FITTING_NAMES[1])),
                    format_func=lambda f: {
                        "Sheth-Tormen 1999": "⭐ Sheth-Tormen (1999) [Standard Ref]",
                        "Tinker 2008": "⭐ Tinker (2008) [Calibrated SO]",
                        "Press-Schechter 1974": "Press-Schechter (1974) [Analytic Top-Hat]",
                        "Watson SO 2013": "Watson (2013) [Spherical Overdensity]",
                        "Watson FO 2013": "Watson (2013) [Friends-of-Friends]",
                    }.get(f, f),
                    key="hf_fitting",
                )
                contract = fit_contract(params["fitting"])
                required_definition = contract.mass_definition
                params["mass_definition"] = required_definition
                st.caption(
                    f"Halo mass definition · {MASS_DEFINITIONS[required_definition]}. This fit fixes the definition; HaloForge never silently converts halo masses."
                )
                slider("delta_c")
                slider("mass_min_exp")
                slider("mass_max_exp")
                slider("mass_points")
                slider("selected_mass_exp")
                slider("single_z")
                params["delta_halo"] = st.number_input(
                    "Halo overdensity Δ relative to mean matter density",
                    min(75.1, float(params.get("delta_halo", 200.0))),
                    max(3200.0, float(params.get("delta_halo", 200.0))),
                    float(params.get("delta_halo", 200.0)),
                    1.0,
                    key="hf_delta_halo",
                )
                if params["fitting"] == "Tinker 2008":
                    st.caption(
                        "Tinker uses 200 ≤ Δmean ≤ 3200. A, a, b, and c all vary with Δ. Its redshift calibration extends to z=2.5; evaluation beyond it is exploratory."
                    )
            with st.expander("CLASS + integration numerics", expanded=False):
                slider("k_min")
                slider("k_max")
                slider("k_points")
                slider("quad_limit", "Integration batch size")
                st.caption(
                    "Top-hat and Gaussian variance use Simpson integration in ln k. Sharp-k integrates to the exact cutoff, k = 1/R."
                )
            with st.expander("Research intent", expanded=False):
                st.caption(
                    "Optional, but saved with this experiment and its reproducibility bundle."
                )
                research_question = st.text_area(
                    "Research question",
                    key="hf_research_question",
                    placeholder="What am I trying to learn?",
                )
                hypothesis = st.text_area(
                    "Hypothesis before running",
                    key="hf_hypothesis",
                    placeholder="What do I predict, and why?",
                )
                prediction = st.text_input(
                    "Prediction",
                    key="hf_prediction",
                    placeholder="For example: fewer massive halos",
                )
            selected_z = st.multiselect(
                "Redshifts sent to CLASS",
                sorted({0, 0.5, 1, 2, 5, 10, 100, *params.get("z_values", [])}),
                default=params.get("z_values", [0, 0.5, 1, 2, 5, 10, 100]),
                key="hf_z_values",
            )
            params["z_values"] = sorted(
                set(float(z) for z in [0, *selected_z, float(params["single_z"])])
            )
            submitted = st.form_submit_button(
                "Calculate & save run", type="primary", width="stretch"
            )

        reset_col, status_col = st.columns([1, 1.7])
        if reset_col.button("Reset controls", width="stretch"):
            reset_params()
            clear_widgets()
            st.rerun()
        status_col.caption(
            "Slider edits are staged in the browser and cannot trigger calculations by themselves."
        )

        if submitted:
            calculation_started = perf_counter()
            try:
                with st.spinner(
                    "Running isolated AxiCLASS, integrating σ(M,z), and saving atomically…"
                ):
                    result = run_new_cosmology(
                        run_name,
                        notebook={
                            "research_question": research_question,
                            "hypothesis": hypothesis,
                            "prediction": prediction,
                            "parent_run_id": st.session_state.get("current_run_id"),
                        },
                    )
                st.session_state["run_flash"] = (
                    f"Saved {result['saved_run']['name']} locally."
                )
                record_local_diagnostic(
                    run_storage.DATA_ROOT,
                    bool(st.session_state.get("hf_local_diagnostics_enabled", False)),
                    "calculation_completed",
                    duration_seconds=perf_counter() - calculation_started,
                )
                clear_widgets()
                st.rerun()
            except (ClassRuntimeError, ValueError) as exc:
                show_failure(
                    exc,
                    preserve_note="The previous completed run is still loaded and its files were not changed.",
                )
            except Exception as exc:
                show_failure(
                    exc,
                    preserve_note="The previous completed run is still loaded and its files were not changed.",
                )

        if st.session_state.get("run_flash"):
            st.success(st.session_state.pop("run_flash"))
        if slow_parameters_changed():
            st.warning(
                "Draft controls differ from the completed run. Graphs remain tied to the saved run until you run the new settings."
            )
        result = get_matter_power_result()
        if result:
            source = "cache" if result.get("from_cache") else "computed"
            active_name = st.session_state.get("last_auto_saved_run", {}).get(
                "name", "restored run"
            )
            st.markdown(
                f'<div class="runtime-ok"><i></i>{result["class_status"]} · {source} · {len(result["redshifts"])} z<br><span>{active_name}</span></div>',
                unsafe_allow_html=True,
            )
    return section


def require_run():
    result = get_matter_power_result()
    sigma = get_sigma_result()
    run = current_pipeline_run()
    if result is None or sigma is None or run is None:
        st.markdown(
            '<div class="empty"><div class="empty-orbit"><i></i></div><b>Forge the first universe</b><span>Choose a preset or tune the staged controls, then press Calculate & save run. Completed runs persist in the local data vault.</span></div>',
            unsafe_allow_html=True,
        )
        return None
    return run, result, sigma


GUIDED_EXPERIMENTS = {
    "Early expansion": {
        "name": "Guided EDE experiment",
        "params": {"enable_ede": True, "f_EDE": 0.12, "log10_a_c": -3.5},
        "question": "How does early dark energy change cluster abundance?",
        "description": "Add early dark energy near matter–radiation equality. Keep the other cosmological parameters fixed.",
        "prediction": "If early expansion briefly speeds up, what do you expect for very massive halos?",
        "choices": ["More massive halos", "Fewer massive halos", "I am not sure yet"],
        "expected": "Expect a causal chain from the early background to P(k), σ(M), then model-dependent halo counts.",
        "caveat": "A halo mass-function fit is an empirical calibration, not a direct observation.",
    },
    "More small-scale power": {
        "name": "Guided blue-tilt experiment",
        "params": {"enable_ede": False, "n_s": 0.99},
        "question": "Can a small tilt change matter more for small halos than for clusters?",
        "description": "Raise nₛ while holding the primordial amplitude and pivot fixed.",
        "prediction": "Which scales should respond most strongly to a bluer primordial tilt?",
        "choices": [
            "Lower-mass halo scales",
            "All halo masses equally",
            "I am not sure yet",
        ],
        "expected": "Expect relative high-k power to change first, then lower-mass σ(M) more strongly.",
        "caveat": "The magnitude of any halo-count response depends on the selected fit and its calibration domain.",
    },
    "Why kmax matters": {
        "name": "Guided kmax-coverage experiment",
        "params": {"enable_ede": False, "k_max": 300.0},
        "question": "Can a numerical Fourier cutoff alter a low-mass conclusion without changing the cosmology?",
        "description": "Extend the sampled k range while leaving physical cosmological parameters fixed.",
        "prediction": "Is this change physical, numerical, or both?",
        "choices": ["Numerical coverage", "A different cosmology", "I am not sure yet"],
        "expected": "Expect the selected low-mass contribution band and endpoint diagnostic to be the key evidence.",
        "caveat": "A wider sampled range is not by itself a full convergence proof.",
    },
}


def _choose_onboarding_universe(experiment: dict) -> None:
    """Stage one safe, meaningful experiment without exposing the control panel."""
    baseline, candidate = guided_parameter_pair(DEFAULT_PARAMS, experiment["params"])
    params.clear()
    params.update(candidate)
    st.session_state["onboarding_baseline_params"] = baseline
    st.session_state["run_name_draft"] = experiment["name"]
    st.session_state.pop("onboarding_finished_run", None)
    st.session_state.pop("onboarding_outcome", None)
    save_draft_params(params)


def _guided_cluster_outcome(
    baseline_run: dict, candidate_run: dict, fitting: str
) -> dict:
    """Compute the headline result from the exact two completed guided runs."""
    baseline = hmf_z(baseline_run, 0.0, fitting)
    candidate = hmf_z(candidate_run, 0.0, fitting)
    outcome = halo_abundance_change(
        candidate["M_h"],
        candidate["hmf"],
        baseline["M_h"],
        baseline["hmf"],
        CLUSTER_MASS_HINV_MSUN,
    )
    outcome["fitting"] = fitting
    outcome["candidate_validity"] = candidate.get("validity", {})
    outcome["baseline_validity"] = baseline.get("validity", {})
    return outcome


def causal_reveal_visual(stage: int, experiment_name: str = "Early expansion") -> None:
    """Render the guided causal chain as optional motion with a static reading order.

    The labels and arrows intentionally carry the explanation without relying on
    animation.  CSS honors both the app's Reduce motion setting and the operating
    system preference, leaving the same causal map visible in either mode.
    """
    input_label = {
        "Early expansion": "Early dark energy",
        "More small-scale power": "Primordial tilt",
        "Why kmax matters": "Fourier coverage",
    }[experiment_name]
    labels = (
        ("CHANGED INPUT", input_label, "The controlled change"),
        ("MATTER POWER", "P(k) is processed", "First calculated consequence"),
        ("SMOOTHING", "σ(M) gathers modes", "Scale becomes halo mass"),
        ("HALO ABUNDANCE", "dn/dlnM responds", "Fit-dependent prediction"),
    )
    active = max(0, min(int(stage), len(labels) - 1))
    nodes = "".join(
        f'<div class="causal-node {"active" if index == active else ""} {"seen" if index < active else ""}">'
        f"<span>{escape(kicker)}</span><b>{escape(title)}</b><small>{escape(detail)}</small></div>"
        for index, (kicker, title, detail) in enumerate(labels)
    )
    st.markdown(
        '<div class="causal-reveal" role="img" '
        f'aria-label="Causal chain, currently focused on {labels[active][1]}. '
        'Early condition leads to matter power, then smoothing, then a fit-dependent halo abundance prediction.">'
        f'<div class="causal-stream stage-{active}"><i></i><i></i><i></i></div>{nodes}</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Motion is illustrative, not evidence. The labeled chain remains the complete explanation when motion is reduced or unavailable."
    )


def _reset_guided_experiment() -> None:
    for key in (
        "onboarding_ready",
        "onboarding_finished_run",
        "onboarding_outcome",
        "onboarding_baseline_params",
        "onboarding_reveal_stage",
    ):
        st.session_state.pop(key, None)


def explore_view():
    requested_experiment = st.session_state.pop("teaching_experiment_request", None)
    experiment_names = list(GUIDED_EXPERIMENTS)
    # A teaching module supplies a requested experiment before this widget is
    # created.  Using its index—not a late session-state write—keeps the
    # visible selector and staged lesson in sync.
    if requested_experiment in GUIDED_EXPERIMENTS:
        st.session_state.pop("guided_experiment", None)
        experiment_index = experiment_names.index(requested_experiment)
    else:
        experiment_index = 0
    experiment_name = st.selectbox(
        "Curated experiment",
        experiment_names,
        index=experiment_index,
        key="guided_experiment",
        on_change=_reset_guided_experiment,
    )
    experiment = GUIDED_EXPERIMENTS[experiment_name]
    launched_module = st.session_state.pop("teaching_launch_notice", None)
    if launched_module:
        st.success(
            f"Teaching module ready: {launched_module}. Start with a prediction, then run this controlled comparison."
        )
    st.markdown(
        '<div class="page-head"><span>GUIDED EXPERIMENT</span>'
        f"<h2>{experiment['question']}</h2>"
        "<p>Predict the effect, then compare matter power, mass variance, and modelled halo abundance.</p></div>",
        unsafe_allow_html=True,
    )
    st.info(
        "Each experiment compares two calculations. Explanations below introduce the physics and its limits."
    )
    prediction = st.radio(
        "Before calculating: " + experiment["prediction"],
        experiment["choices"],
        index=None,
        key=f"onboarding_prediction_{experiment_name}",
        horizontal=True,
    )
    prediction_ready = prediction in experiment["choices"]
    if not prediction_ready:
        st.caption("Choose a prediction to set up the comparison.")
    left, right = st.columns([1.25, 1])
    with left:
        st.markdown("#### Make one change")
        st.markdown(experiment["description"])
        if st.button(
            "Set up this experiment",
            type="primary",
            key="onboarding_setup",
            disabled=not prediction_ready,
        ):
            _choose_onboarding_universe(experiment)
            st.session_state["onboarding_experiment"] = experiment_name
            st.session_state["onboarding_ready"] = True
            st.rerun()
    with right:
        st.markdown("#### What you will inspect")
        st.markdown("""1. The direct input changes.
2. Matter power responds.
3. Smoothing maps the response to mass.
4. The halo-count prediction responds—subject to fit validity.""")
        st.caption(experiment["expected"])

    if st.session_state.get("onboarding_ready"):
        st.success(
            "Prediction recorded. Your one change is staged; it has not run yet."
        )
        if st.button(
            "Calculate the guided universe", type="primary", key="onboarding_run"
        ):
            candidate_params = deepcopy(params)
            try:
                prediction = committed_prediction(prediction, experiment["choices"])
                with st.spinner(
                    "Calculating the ΛCDM baseline and candidate with AxiCLASS…"
                ):
                    params.clear()
                    params.update(
                        deepcopy(st.session_state["onboarding_baseline_params"])
                    )
                    save_draft_params(params)
                    baseline_result = run_new_cosmology(
                        "Guided Planck-like baseline",
                        notebook={
                            "research_question": "Matched baseline for "
                            + experiment["question"],
                            "hypothesis": "Reference cosmology before the guided change.",
                            "prediction": "Reference curve for a bounded comparison.",
                            "caveats": [
                                "This is a matched local reference, not an external validation case."
                            ],
                        },
                    )
                    baseline_run = current_pipeline_run()
                    params.clear()
                    params.update(candidate_params)
                    save_draft_params(params)
                    result = run_new_cosmology(
                        experiment["name"],
                        notebook={
                            "research_question": experiment["question"],
                            "hypothesis": prediction,
                            "prediction": prediction,
                            "parent_run_id": baseline_result["saved_run"]["run_id"],
                            "caveats": [experiment["caveat"]],
                        },
                    )
                    candidate_run = current_pipeline_run()
                    outcome = _guided_cluster_outcome(
                        baseline_run, candidate_run, candidate_params["fitting"]
                    )
                st.session_state["onboarding_finished_run"] = result["saved_run"][
                    "run_id"
                ]
                st.session_state["onboarding_outcome"] = outcome
                st.session_state["onboarding_baseline_id"] = baseline_result[
                    "saved_run"
                ]["run_id"]
                st.session_state["onboarding_ready"] = False
                st.rerun()
            except (ClassRuntimeError, ValueError) as exc:
                params.clear()
                params.update(candidate_params)
                save_draft_params(params)
                show_failure(
                    exc,
                    preserve_note="Nothing was overwritten. You can change course or investigate the diagnostic details.",
                )
            except Exception as exc:
                params.clear()
                params.update(candidate_params)
                save_draft_params(params)
                show_failure(
                    exc,
                    preserve_note="Nothing was overwritten. Your guided candidate remains staged, and the prior completed run is preserved.",
                )

    if st.session_state.get("onboarding_finished_run"):
        st.markdown("### The reveal")
        outcome = st.session_state.get("onboarding_outcome")
        if outcome:
            st.markdown(
                f"Modelled abundance at **10¹⁴ h⁻¹ M☉**, z = 0: "
                f"**{outcome['percent_difference']:+.2f}%** relative to the matched ΛCDM baseline."
            )
            st.caption(
                f"{outcome['candidate_method']} versus {outcome['baseline_method']} · "
                f"{outcome['observable']} · fit: {outcome['fitting']}. {outcome['scope_limit']}"
            )
        else:
            st.info(
                "The guided run is saved, but the matched cluster comparison was unavailable. Inspect the causal chain and HMF validity before drawing an abundance conclusion."
            )
        reveal_index = st.select_slider(
            "Reveal the causal chain — focus this control and use Left/Right Arrow keys.",
            options=list(range(len(REVEAL_STAGES))),
            value=0,
            format_func=reveal_stage,
            key="onboarding_reveal_stage",
        )
        reveal_stage(reveal_index)
        pipeline = current_pipeline_run()
        causal_steps = explain_change(
            st.session_state["onboarding_baseline_params"], params
        )
        causal_reveal_visual(reveal_index, experiment_name)
        if reveal_index == 0:
            st.markdown(
                '<div class="lesson-card compact"><span>EARLY CONDITION</span><h3>One early-universe change</h3><p>The guided candidate differs from the matched baseline only in the selected experiment settings. HaloForge keeps the rest of the staged baseline settings fixed so the following plots can be read as a controlled comparison.</p><b>Start with the physical condition—not with a graph.</b></div>',
                unsafe_allow_html=True,
            )
        elif pipeline is None:
            st.info(
                "The saved guided run could not be restored for this stage. Its notebook and export bundle remain available."
            )
        elif reveal_index == 1:
            st.caption(
                "The first plotted consequence: linear matter power at z = 0. This is a calculated spectrum, not a halo catalogue."
            )
            chart(
                power_fig(pipeline["power_result"], [0]), 470, "onboarding_reveal_power"
            )
        elif reveal_index == 2:
            st.caption(
                "The smoothing step maps matter power to the RMS fluctuation strength at each halo mass. The selected mass remains synchronized across the calculation."
            )
            chart(
                sigma_fig(pipeline["sigma_result"], pipeline["params"], [0]),
                470,
                "onboarding_reveal_sigma",
            )
        else:
            st.caption(
                "The last step applies the selected halo mass-function relation. Read its calibration and cosmology-support limits with the curve."
            )
            chart(
                hmf_fig(pipeline, [pipeline["params"]["fitting"]], [0]),
                470,
                "onboarding_reveal_hmf",
            )
        with st.expander("Why this stage follows from the changed input"):
            if not causal_steps:
                st.info(
                    "No supported causal step is available for the current guided parameters. Inspect the exact parameter diff and diagnostics instead of inferring a story."
                )
            for step in causal_steps:
                st.markdown(f"""**Changed:** {step.changed}.

**Directly affected:** {step.direct_equation}.

**Then:** {step.downstream}.

**Robust statement:** {step.robust}.

**What could make this interpretation wrong:** {step.caveat}.

**Try to falsify it:** {step.falsifier}.""")
        routes = st.columns(3)
        routes[0].button(
            "Understand: open the learning path",
            key="onboarding_understand",
            on_click=lambda: st.session_state.update(
                {"onboarding_route": "Learn the pipeline"}
            ),
        )
        routes[1].button(
            "Compare: use a baseline",
            key="onboarding_compare",
            on_click=lambda: st.session_state.update(
                {
                    "onboarding_route": "Compare lab",
                    "compare_run_ids": [
                        st.session_state["onboarding_baseline_id"],
                        st.session_state["onboarding_finished_run"],
                    ],
                    "compare_baseline_id": st.session_state["onboarding_baseline_id"],
                    "compare_mode": "Percent difference",
                    "compare_panel_count": "1",
                }
            ),
        )
        routes[2].button(
            "Investigate: inspect assumptions",
            key="onboarding_investigate",
            on_click=lambda: st.session_state.update(
                {"onboarding_route": "Diagnostics"}
            ),
        )
        st.caption(
            "Choose a route to continue immediately. The matched baseline and guided candidate are preserved as linked local runs in the Notebook."
        )

    st.markdown("### Conceptual zoom")
    concept = st.selectbox(
        "Choose a concept", [item.label for item in CONCEPTS], key="concept_zoom_term"
    )
    concept_detail = concept_by_label(concept)
    zoom = st.tabs(["Intuition", "Course", "Research", "Implementation"])
    zoom[0].write(concept_detail.intuition)
    zoom[1].write(concept_detail.course)
    zoom[2].write(concept_detail.research)
    zoom[3].write(concept_detail.implementation)


def notebook_view():
    st.markdown(
        '<div class="page-head"><span>EXPERIMENT NOTEBOOK</span><h2>Every run is a question, a prediction, and an auditable branch.</h2><p>Notes are saved locally with the run and included in its reproducibility bundle.</p></div>',
        unsafe_allow_html=True,
    )
    runs = load_all_runs()
    if not runs:
        st.info(
            "Your notebook begins when you calculate an experiment. Explore provides a guided first entry."
        )
        return
    rows = lineage_rows(runs)
    children = {row["run_id"]: [] for row in rows}
    roots = []
    for row in rows:
        if row["parent_run_id"] and row["parent_run_id"] in children:
            children[row["parent_run_id"]].append(row)
        else:
            roots.append(row)

    def tree(row, prefix=""):
        lines = [f"{prefix}{row['name']}"]
        for child in children[row["run_id"]]:
            lines.extend(tree(child, prefix + "└── "))
        return lines

    st.code("\n".join(line for root in roots for line in tree(root)), language="text")
    run_map = {run["run_id"]: run for run in runs}
    selected_id = st.selectbox(
        "Experiment",
        list(run_map),
        format_func=lambda rid: run_map[rid]["name"],
        key="notebook_selected_run",
    )
    selected = run_map[selected_id]
    entry = normalize_notebook_entry(selected.get("notebook"))
    available_follow_ups = [run_id for run_id in run_map if run_id != selected_id]
    linked_follow_ups = [
        run_id
        for run_id in entry["follow_up_run_ids"]
        if run_id in run_map and run_id != selected_id
    ]
    missing_follow_ups = [
        run_id for run_id in entry["follow_up_run_ids"] if run_id not in run_map
    ]
    if linked_follow_ups:
        st.caption("Linked follow-up experiments")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "experiment": run_map[run_id]["name"],
                        "run_id": run_id,
                        "relationship": "follow-up evidence",
                    }
                    for run_id in linked_follow_ups
                ]
            ),
            width="stretch",
            hide_index=True,
        )
    if missing_follow_ups:
        st.warning(
            "This notebook references follow-up runs that are not available locally: "
            + ", ".join(missing_follow_ups)
            + ". Their IDs are preserved in the saved record until you edit these links."
        )
    baseline = next((run for run in runs if run.get("is_baseline")), None)
    if baseline and baseline["run_id"] != selected_id:
        diff = parameter_diff(baseline["params"], selected["params"])
        st.caption("Parameter difference from named baseline")
        st.dataframe(pd.DataFrame(diff), width="stretch", hide_index=True)
    with st.form("notebook_entry"):
        question = st.text_area("Research question", value=entry["research_question"])
        hypothesis = st.text_area(
            "Hypothesis before running", value=entry["hypothesis"]
        )
        prediction = st.text_input("Prediction", value=entry["prediction"])
        conclusion = st.text_area("Conclusion", value=entry["conclusion"])
        caveats = st.text_area(
            "Caveats — one per line", value="\n".join(entry["caveats"])
        )
        citations = st.text_area(
            "Citations — one per line", value="\n".join(entry["citations"])
        )
        follow_up_ids = st.multiselect(
            "Linked follow-up experiments",
            available_follow_ups,
            default=linked_follow_ups,
            format_func=lambda run_id: run_map[run_id]["name"],
            help="Link convergence checks, fit comparisons, or other evidence without changing either run's parameters or parent branch.",
        )
        annotate_region = st.checkbox("Attach a note to a chart region")
        annotation = st.text_input(
            "Annotation note",
            placeholder="Example: Check the high-mass tail above 10¹⁴ h⁻¹ M☉",
        )
        chart_title = st.selectbox(
            "Chart",
            [
                "Linear matter power spectrum",
                "Mass variance",
                "Differential halo mass function",
                "Cumulative halo abundance",
                "Variance contribution",
                "Growth consistency check",
            ],
            disabled=not annotate_region,
        )
        region = st.columns(2)
        x_start = region[0].number_input(
            "Region start x", value=0.0, format="%.5g", disabled=not annotate_region
        )
        x_end = region[1].number_input(
            "Region end x", value=0.0, format="%.5g", disabled=not annotate_region
        )
        if st.form_submit_button("Save notebook entry", type="primary"):
            new_annotations = entry["annotations"] + (
                [
                    {
                        "text": annotation,
                        "chart_title": chart_title,
                        "x_start": x_start,
                        "x_end": x_end,
                        "created_at": pd.Timestamp.utcnow().isoformat(),
                    }
                ]
                if annotate_region and annotation.strip() and x_end > x_start
                else []
            )
            if annotate_region and annotation.strip() and x_end <= x_start:
                st.warning(
                    "The chart region was not added because its end must be greater than its start."
                )
            selected["notebook"] = normalize_notebook_entry(
                {
                    **entry,
                    "research_question": question,
                    "hypothesis": hypothesis,
                    "prediction": prediction,
                    "conclusion": conclusion,
                    "caveats": caveats.splitlines(),
                    "citations": citations.splitlines(),
                    "follow_up_run_ids": follow_up_ids,
                    "annotations": new_annotations,
                }
            )
            update_run_metadata(selected)
            generate_run_exports(selected)
            st.success("Notebook entry and export bundle updated.")
    if entry["annotations"]:
        st.caption("Saved chart-region annotations")
        st.dataframe(
            pd.DataFrame(entry["annotations"]), width="stretch", hide_index=True
        )
    st.caption(
        f"Reproducibility hash: {selected.get('reproducibility_hash', 'unavailable')}"
    )
    integrity = selected.get("integrity_status", {})
    if integrity:
        state = integrity.get("state", "unverified_legacy")
        (st.success if state == "verified" else st.warning)(
            f"Saved-artifact integrity: {state.replace('_', ' ')}. "
            + (
                "The stored arrays and declared calculation identity match."
                if state == "verified"
                else "This run has incomplete or failed integrity evidence; do not treat it as a verified reproduction."
            )
        )
    audit = audit_status(selected)
    with st.expander("Local audit trail"):
        st.caption(audit["scope"])
        if audit["state"] == "verified":
            st.success(f"Local event chain verified ({audit['events']} events).")
        else:
            st.warning(
                "Local event chain is unavailable or malformed; do not treat it as an authenticated audit record."
            )
        st.dataframe(
            pd.DataFrame(selected.get("audit_trail", [])),
            width="stretch",
            hide_index=True,
        )
    artifacts = attached_artifact_rows(selected)
    with st.expander("Attached export artifacts"):
        st.caption(
            "These local files are generated for this exact saved run. Missing files remain visible; an attachment does not establish independent scientific validity."
        )
        if artifacts:
            st.dataframe(pd.DataFrame(artifacts), width="stretch", hide_index=True)
            st.download_button(
                "Download artifact attachment manifest",
                json.dumps(artifacts, indent=2),
                f"{selected['name']}_artifact_manifest.json",
                "application/json",
            )
        else:
            st.info("No export artifacts are attached to this run yet.")
    st.markdown("### Share card")
    fingerprint = cosmic_fingerprint(
        selected, baseline if baseline and baseline["run_id"] != selected_id else None
    )
    if fingerprint["has_baseline"] and fingerprint["signals"]:
        signal_rows = [
            row
            for row in fingerprint["signals"]
            if row["fractional_change"] is not None
        ]
        if signal_rows:
            card_fig = go.Figure(
                go.Bar(
                    x=[100 * row["fractional_change"] for row in signal_rows],
                    y=[row["label"] for row in signal_rows],
                    orientation="h",
                    marker_color=[
                        "#35d7e5" if row["fractional_change"] >= 0 else "#ff718b"
                        for row in signal_rows
                    ],
                    hovertemplate="stored fractional change=%{x:+.2f}%<extra></extra>",
                )
            )
            card_fig.update_xaxes(
                title="stored fractional change [%]",
                zeroline=True,
                zerolinecolor="#8ea1a9",
            )
            card_fig.update_yaxes(
                categoryorder="array",
                categoryarray=[row["label"] for row in signal_rows],
            )
            chart(
                set_plot(
                    card_fig,
                    "Cosmic fingerprint",
                    "A compact, data-derived summary of stored linear baseline/candidate differences.",
                ),
                320,
                "cosmic_fingerprint",
            )
    st.caption(fingerprint["scope_limit"])
    card = share_card_markdown(
        selected, baseline if baseline and baseline["run_id"] != selected_id else None
    )
    st.markdown(card)
    st.download_button(
        "Download share card",
        card,
        f"{selected['name']}_share_card.md",
        "text/markdown",
    )


def z_index(redshifts, z):
    return redshift_index(redshifts, z)


def primordial_fig(p):
    k = np.logspace(-5, 1, 500)
    y = float(p["A_s"]) * (k / float(p["k_pivot"])) ** (float(p["n_s"]) - 1)
    f = go.Figure(
        go.Scatter(
            x=k,
            y=y,
            line=dict(color=COLORS[0], width=3),
            hovertemplate="k=%{x:.3e} Mpc⁻¹<br>𝒫ℛ=%{y:.4e}<extra></extra>",
        )
    )
    f.add_vline(
        x=float(p["k_pivot"]),
        line_dash="dot",
        line_color=COLORS[1],
        annotation_text="kₚ",
        annotation_position="top right",
    )
    f.update_xaxes(type="log", title="k [Mpc⁻¹]")
    f.update_yaxes(type="log", title="𝒫ℛ(k)")
    return set_plot(
        f, "Primordial curvature spectrum", GRAPH_CAPTIONS["Primordial spectrum"]
    )


def power_fig(result, redshifts=None):
    f = go.Figure()
    use = redshifts or list(result["redshifts"])
    for i, z in enumerate(use):
        j = z_index(result["redshifts"], z)
        f.add_trace(
            go.Scatter(
                x=result["k"],
                y=result["P_by_z"][j],
                name=f"z={float(z):g}",
                line=dict(color=COLORS[i % len(COLORS)], width=2.7),
                hovertemplate="k=%{x:.3e} Mpc⁻¹<br>P=%{y:.4e} Mpc³<extra></extra>",
            )
        )
    f.update_xaxes(type="log", title="k [Mpc⁻¹]")
    f.update_yaxes(type="log", title="P(k,z) [Mpc³]")
    return set_plot(f, "Linear matter power", GRAPH_CAPTIONS["Matter P(k)"])


def delta2_fig(result, redshifts=None):
    f = go.Figure()
    use = redshifts or list(result["redshifts"])
    for i, z in enumerate(use):
        j = z_index(result["redshifts"], z)
        y = result["k"] ** 3 * result["P_by_z"][j] / (2 * np.pi**2)
        f.add_trace(
            go.Scatter(
                x=result["k"],
                y=y,
                name=f"z={float(z):g}",
                line=dict(color=COLORS[i % len(COLORS)], width=2.6),
                hovertemplate="k=%{x:.3e} Mpc⁻¹<br>Δ²=%{y:.4e}<extra></extra>",
            )
        )
    f.update_xaxes(type="log", title="k [Mpc⁻¹]")
    f.update_yaxes(type="log", title="Δ²(k,z)")
    return set_plot(
        f, "Dimensionless power per ln k", GRAPH_CAPTIONS["Dimensionless Δ²(k)"]
    )


def transfer_fig(result, p):
    k = np.asarray(result["k"])
    shape = np.asarray(result["P"]) / k ** float(p["n_s"])
    norm_mask = k <= max(5 * k[0], 0.003)
    shape /= float(np.median(shape[norm_mask])) if np.any(norm_mask) else shape[0]
    f = go.Figure(
        go.Scatter(
            x=k,
            y=shape,
            line=dict(color=COLORS[5], width=2.8),
            hovertemplate="k=%{x:.3e} Mpc⁻¹<br>T² proxy=%{y:.4e}<extra></extra>",
        )
    )
    f.update_xaxes(type="log", title="k [Mpc⁻¹]")
    f.update_yaxes(type="log", title="normalized P(k)/kⁿˢ")
    return set_plot(
        f, "Scale-dependent processing shape", GRAPH_CAPTIONS["Processing shape"]
    )


def windows_fig(squared=False):
    y = np.logspace(-3, 2, 1000)
    f = go.Figure()
    for i, w in enumerate(WINDOWS):
        values = window_squared(y, w) if squared else window_W(y, w)
        f.add_trace(
            go.Scatter(
                x=y,
                y=values,
                name=w,
                line=dict(color=COLORS[i], width=2.5),
                hovertemplate="kR=%{x:.3e}<br>response=%{y:.4f}<extra></extra>",
            )
        )
    f.update_xaxes(type="log", title="kR")
    f.update_yaxes(
        title="W²(kR)" if squared else "W(kR)",
        range=[-0.03, 1.08] if squared else [-0.35, 1.08],
    )
    key = "Window W²" if squared else "Window W"
    return set_plot(
        f,
        "Squared smoothing response" if squared else "Smoothing-window response",
        GRAPH_CAPTIONS[key],
    )


def taylor_fig():
    y = np.logspace(-7, -0.3, 600)
    exact = top_hat_W_exact(y)
    series = top_hat_W_series(y)
    err = np.abs((series - exact) / np.maximum(np.abs(exact), 1e-30))
    f = go.Figure(
        go.Scatter(
            x=y,
            y=np.maximum(err, 1e-18),
            line=dict(color=COLORS[1], width=2.6),
            hovertemplate="kR=%{x:.3e}<br>relative error=%{y:.3e}<extra></extra>",
        )
    )
    f.add_vline(
        x=0.1, line_dash="dash", line_color=COLORS[0], annotation_text="series switch"
    )
    f.update_xaxes(type="log", title="kR")
    f.update_yaxes(type="log", title="relative error")
    return set_plot(
        f, "Small-kR top-hat validation", GRAPH_CAPTIONS["Top-hat series error"]
    )


def sigma_fig(sigma, p, redshifts=None):
    f = go.Figure()
    use = redshifts or list(sigma["redshifts"])
    for i, z in enumerate(use):
        j = z_index(sigma["redshifts"], z)
        f.add_trace(
            go.Scatter(
                x=sigma["M_h"],
                y=sigma["sigma_by_z"][j],
                name=f"z={float(z):g}",
                line=dict(color=COLORS[i % len(COLORS)], width=2.7),
                hovertemplate="M=%{x:.3e} h⁻¹M☉<br>σ=%{y:.4f}<extra></extra>",
            )
        )
    f.add_hline(
        y=float(p["delta_c"]),
        line_dash="dot",
        line_color=COLORS[3],
        annotation_text="δc",
        annotation_position="top right",
    )
    f.add_vline(
        x=10 ** float(p["selected_mass_exp"]),
        line_dash="dash",
        line_color=COLORS[1],
        annotation_text="selected M",
        annotation_position="top left",
    )
    f.update_xaxes(type="log", title="M [h⁻¹ M☉]")
    f.update_yaxes(type="log", title="σ(M,z)")
    return set_plot(f, "Mass variance", GRAPH_CAPTIONS["σ(M)"])


def derivative_fig(sigma, p, z=None):
    use_z = float(p["single_z"] if z is None else z)
    j = z_index(sigma["redshifts"], use_z)
    y = np.abs(sigma["dlnsigma_dlnM_by_z"][j])
    f = go.Figure(
        go.Scatter(
            x=sigma["M_h"],
            y=y,
            line=dict(color=COLORS[4], width=2.7),
            hovertemplate="M=%{x:.3e} h⁻¹M☉<br>|slope|=%{y:.4f}<extra></extra>",
        )
    )
    f.update_xaxes(type="log", title="M [h⁻¹ M☉]")
    f.update_yaxes(type="log", title="|d lnσ / d lnM|")
    return set_plot(f, "Logarithmic variance slope", GRAPH_CAPTIONS["σ slope"])


def radius_fig(sigma, p=None):
    f = go.Figure(
        go.Scatter(
            x=sigma["M_h"],
            y=sigma["R"],
            line=dict(color=COLORS[2], width=2.8),
            hovertemplate="M=%{x:.3e} h⁻¹M☉<br>R=%{y:.4f} Mpc<extra></extra>",
        )
    )
    if p is not None:
        f.add_vline(
            x=10 ** float(p["selected_mass_exp"]),
            line_dash="dash",
            line_color=COLORS[1],
            annotation_text="selected M",
            annotation_position="top left",
        )
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
        f.add_trace(
            go.Scatter(
                x=result["k"],
                y=np.maximum(values, 1e-300),
                name=w,
                line=dict(color=COLORS[i], width=2.5),
                hovertemplate="k=%{x:.3e} Mpc⁻¹<br>dσ²/dlnk=%{y:.4e}<extra></extra>",
            )
        )
        if w == "Top-hat":
            peak = peak_point(result["k"], values)
            if peak:
                f.add_vline(
                    x=peak["x"],
                    line_dash="dot",
                    line_color=COLORS[i],
                    annotation_text="peak sampled contribution",
                    annotation_position="top right",
                )
    f.update_xaxes(type="log", title="k [Mpc⁻¹]")
    f.update_yaxes(type="log", title="dσ²/dlnk")
    return set_plot(
        f,
        f"Variance contribution for 10^{float(p['selected_mass_exp']):g} h⁻¹M☉",
        GRAPH_CAPTIONS["σ integrand"],
    )


def growth_fig(result, sigma):
    order = np.argsort(result["redshifts"])
    z = np.asarray(result["redshifts"])[order]
    growth = np.asarray(result["growth_class"])[order]
    sigma_sorted = np.asarray(sigma["sigma8_pipeline_by_z"])[order]
    sigma_ratio = sigma_sorted / float(sigma_sorted[z_index(z, 0.0)])
    f = go.Figure()
    f.add_trace(
        go.Scatter(
            x=z,
            y=growth,
            mode="lines+markers",
            name="CLASS D(z)",
            line=dict(color=COLORS[4], width=2.8),
            hovertemplate="z=%{x:g}<br>D=%{y:.5f}<extra></extra>",
        )
    )
    f.add_trace(
        go.Scatter(
            x=z,
            y=sigma_ratio,
            mode="lines+markers",
            name="σ₈(z)/σ₈(0)",
            line=dict(color=COLORS[0], width=2.5, dash="dash"),
            hovertemplate="z=%{x:g}<br>ratio=%{y:.5f}<extra></extra>",
        )
    )
    f.update_xaxes(type="linear", title="redshift z")
    f.update_yaxes(title="normalized linear growth", range=[0, 1.06])
    return set_plot(f, "Growth consistency check", GRAPH_CAPTIONS["Growth"])


def multiplicity_fig(p, fits=None):
    s = np.logspace(-1.1, 0.8, 500)
    f = go.Figure()
    chosen = fits or [p["fitting"]]
    for i, name in enumerate(chosen):
        try:
            y = fitting_values(
                s,
                p["delta_c"],
                name,
                z=p["single_z"],
                delta_halo=p.get("delta_halo", 200),
                neff=-2.0,
                omega_m_z=0.3,
            )
        except (ValueError, FloatingPointError) as exc:
            st.warning(f"{name} was not plotted: {exc}")
            continue
        f.add_trace(
            go.Scatter(
                x=s,
                y=y,
                name=(
                    name + " (n_eff = −2 reference)"
                    if name == "Reed 2007"
                    else name + " (Ωm(z) = 0.3 reference)"
                    if name == "Watson SO 2013"
                    else name
                ),
                line=dict(color=COLORS[i % len(COLORS)], width=2.5),
                hovertemplate="σ=%{x:.4f}<br>f(σ)=%{y:.4e}<extra></extra>",
            )
        )
        for point in turning_points(s, y)[:2]:
            f.add_trace(
                go.Scatter(
                    x=[point["x"]],
                    y=[point["y"]],
                    mode="markers",
                    showlegend=False,
                    marker=dict(
                        color=COLORS[i % len(COLORS)], size=8, symbol="diamond-open"
                    ),
                    hovertemplate=f"{point['kind']}<br>σ=%{{x:.4f}}<br>f(σ)=%{{y:.4e}}<extra></extra>",
                )
            )
    f.update_xaxes(type="log", title="σ")
    f.update_yaxes(type="log", title="f(σ)")
    return set_plot(
        f, "Halo multiplicity functions", GRAPH_CAPTIONS["Multiplicity f(σ)"]
    )


def hmf_fig(run, fits=None, redshifts=None, cumulative=False):
    p = run["params"]
    f = go.Figure()
    chosen = fits or [p["fitting"]]
    zs = redshifts or [p["single_z"]]
    n = 0
    for z in zs:
        for fit in chosen:
            try:
                r = hmf_z(run, float(z), fit)
            except (ValueError, FloatingPointError) as exc:
                st.warning(f"{fit} at z={float(z):g} was not plotted: {exc}")
                continue
            y = cumulative_hmf(r["M_h"], r["hmf"]) if cumulative else r["hmf"]
            color = COLORS[n % len(COLORS)]
            name = f"{fit} · z={float(z):g}"
            valid = np.asarray(r["validity"]["calibrated_mask"], dtype=bool)
            if cumulative:
                valid = np.logical_and.accumulate(valid[::-1])[::-1]
            valid_y = np.where(valid, y, np.nan)
            f.add_trace(
                go.Scatter(
                    x=r["M_h"],
                    y=valid_y,
                    name=name,
                    line=dict(color=color, width=2.6),
                    hovertemplate="M=%{x:.3e} h⁻¹M☉<br>n=%{y:.4e} h³Mpc⁻³<extra></extra>",
                )
            )
            if not np.all(valid):
                f.add_trace(
                    go.Scatter(
                        x=r["M_h"],
                        y=np.where(valid, np.nan, y),
                        name=name + " · outside calibration",
                        line=dict(color=color, width=2.2, dash="dot"),
                        hovertemplate="M=%{x:.3e} h⁻¹M☉<br>n=%{y:.4e} h³Mpc⁻³<br>status=outside calibration<extra></extra>",
                    )
                )
                st.warning(f"{name}: " + "; ".join(r["validity"]["reasons"]))
                for boundary in validity_boundaries(r["M_h"], valid):
                    f.add_vline(
                        x=boundary,
                        line_dash="dash",
                        line_color=color,
                        opacity=0.55,
                        annotation_text="fit boundary",
                        annotation_position="top right",
                    )
            n += 1
    f.update_xaxes(type="log", title="M [h⁻¹ M☉]")
    f.add_vline(
        x=10 ** float(p["selected_mass_exp"]),
        line_dash="dash",
        line_color=COLORS[1],
        annotation_text="selected M",
        annotation_position="top left",
    )
    f.update_yaxes(
        type="log",
        title="n(M < m < Mmax) [h³ Mpc⁻³]" if cumulative else "dn/dlnM [h³ Mpc⁻³]",
    )
    key = "Cumulative HMF" if cumulative else "HMF"
    caption = (
        GRAPH_CAPTIONS[key]
        + " Solid segments are inside the stated calibration checks; dotted segments are numerical extrapolations and are not publication-ready."
    )
    return set_plot(
        f,
        "Cumulative halo abundance"
        if cumulative
        else "Differential halo mass function",
        caption,
    )


GRAPH_NAMES = [
    "Primordial spectrum",
    "Matter P(k)",
    "Dimensionless Δ²(k)",
    "Processing shape",
    "Window W",
    "Window W²",
    "Top-hat series error",
    "Mass–radius map",
    "σ(M)",
    "σ slope",
    "σ integrand",
    "Growth",
    "Multiplicity f(σ)",
    "HMF",
    "Cumulative HMF",
]

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
        "Mass–radius map": lambda: radius_fig(sigma, p),
        "σ(M)": lambda: sigma_fig(sigma, p, zs),
        "σ slope": lambda: derivative_fig(sigma, p),
        "σ integrand": lambda: integrand_fig(result, sigma, p),
        "Growth": lambda: growth_fig(result, sigma),
        "Multiplicity f(σ)": lambda: multiplicity_fig(p, fits),
        "HMF": lambda: hmf_fig(run, fits, zs),
        "Cumulative HMF": lambda: hmf_fig(run, fits, zs, True),
    }[name]()


def dashboard_view():
    st.markdown(
        '<section class="hero"><div><span>COMPUTATIONAL COSMOLOGY</span><h1>One universe.<br>Every scale exposed.</h1><p>Primordial seeds → AxiCLASS P(k,z) → smoothing → σ(M,z) → halo abundance.</p></div><div class="cosmic-orbit"><i></i><b></b><em></em></div></section>',
        unsafe_allow_html=True,
    )
    ready = require_run()
    if not ready:
        return
    run, result, sigma = ready
    d = result["derived"]
    p = run["params"]
    cols = st.columns(5)
    vals = [
        ("h", d["h"], ".5f"),
        ("Ωm", d["Omega_m"], ".5f"),
        ("σ₈ CLASS", d["sigma8"], ".5f"),
        ("z samples", len(result["redshifts"]), "d"),
        ("k evaluations", len(result["k"]) * len(result["redshifts"]), ",d"),
    ]
    for c, (label, val, fmt) in zip(cols, vals):
        c.metric(label, format(val, fmt))
    hmf_validity = None
    try:
        hmf_validity = hmf_z(run, float(p["single_z"]), p["fitting"]).get("validity")
    except (ValueError, FloatingPointError):
        pass
    assurance = assurance_report(run, hmf_validity)
    stored = run_storage.load_run(st.session_state.get("current_run_id", ""))
    benchmark = None
    if stored:
        benchmark = stored.get("benchmarks", {}).get("haloforge-internal-sigma8-v1")
    validity = scientific_validity_record(run, hmf_validity, benchmark)
    validity_label = {
        "not_computed": "Not calculated",
        "computed_needs_scientific_review": "Calculated — review numerical evidence",
        "computed_with_unresolved_numerical_evidence": "Calculated — numerical evidence still unresolved",
        "publication_ready": "Publication evidence recorded",
    }.get(validity["overall_state"], "Scientific status needs review")
    st.caption(
        f"Scientific record: {validity_label}. Detailed claims and the versioned record are below."
    )
    with st.expander(
        "Scientific assurance — six different claims, not one confidence score",
        expanded=False,
    ):
        st.caption(
            "A calculation can be precise without being converged, calibrated, physically appropriate, or publication-ready."
        )
        st.dataframe(pd.DataFrame(assurance), width="stretch", hide_index=True)
    with st.expander("Uncertainty and validity inventory"):
        st.caption(
            "These sources are intentionally separate. A measured integration discrepancy does not quantify theory uncertainty; fit-range support does not prove cosmology support."
        )
        st.dataframe(
            pd.DataFrame(uncertainty_inventory(run, hmf_validity, benchmark)),
            width="stretch",
            hide_index=True,
        )
    try:
        point = inspect_mass_point(
            run, 10 ** float(p["selected_mass_exp"]), float(p["single_z"]), hmf_validity
        )
        with st.expander("Inspect selected halo scale", expanded=False):
            st.caption(
                f"Nearest saved mass to {point['requested_mass_hinv_msun']:.4g} h⁻¹ M☉. "
                "Values below describe that saved sample; no mass interpolation is applied."
            )
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "mass [h⁻¹ M☉]": point["mass_hinv_msun"],
                            "physical mass [M☉]": point["mass_msun"],
                            "radius [Mpc]": point["radius_mpc"],
                            "redshift": point["redshift"],
                            "σ(M)": point["sigma"],
                            "peak height ν": point["nu"],
                            "central sampled k band [Mpc⁻¹]": f"{point['k_10_mpc_inv']:.3g}–{point['k_90_mpc_inv']:.3g}",
                            "passes implemented fit checks": "yes"
                            if point["fit_checks_pass_at_mass"]
                            else (
                                "no"
                                if point["fit_checks_pass_at_mass"] is False
                                else "not checked"
                            ),
                        }
                    ]
                ),
                width="stretch",
                hide_index=True,
                column_config={
                    "mass [h⁻¹ M☉]": st.column_config.NumberColumn(format="%.4e"),
                    "physical mass [M☉]": st.column_config.NumberColumn(format="%.4e"),
                    "radius [Mpc]": st.column_config.NumberColumn(format="%.4g"),
                    "redshift": st.column_config.NumberColumn(format="%.4g"),
                    "σ(M)": st.column_config.NumberColumn(format="%.5g"),
                    "peak height ν": st.column_config.NumberColumn(format="%.5g"),
                },
            )
            st.caption(point["scope_limit"])
    except (ValueError, FloatingPointError, KeyError):
        st.info("The selected mass cannot yet be inspected for this completed run.")
    with st.expander("Select a Fourier range → which halo masses draw on it?"):
        log_min, log_max = (
            float(np.log10(result["k"][0])),
            float(np.log10(result["k"][-1])),
        )
        selected_log_range = st.slider(
            "Selected sampled k interval [log₁₀ Mpc⁻¹]",
            min_value=log_min,
            max_value=log_max,
            value=(
                log_min + 0.2 * (log_max - log_min),
                log_min + 0.8 * (log_max - log_min),
            ),
            step=max((log_max - log_min) / 200, 0.001),
            key="selected_k_contribution_range",
        )
        try:
            range_report = inspect_k_range(
                run,
                10 ** selected_log_range[0],
                10 ** selected_log_range[1],
                float(p["single_z"]),
            )
            st.caption(range_report["scope_limit"])
            st.caption(
                f"Actual sampled interval: {range_report['actual_sampled_k_start_mpc_inv']:.3g}–{range_report['actual_sampled_k_end_mpc_inv']:.3g} Mpc⁻¹"
            )
            st.dataframe(
                pd.DataFrame(range_report["rows"]), width="stretch", hide_index=True
            )
            range_fig = go.Figure(
                go.Bar(
                    x=[row["mass_hinv_msun"] for row in range_report["rows"]],
                    y=[
                        row["sigma2_fraction_from_selected_k"]
                        for row in range_report["rows"]
                    ],
                    marker_color=COLORS[1],
                    hovertemplate="M=%{x:.3e} h⁻¹M☉<br>selected-band fraction=%{y:.2%}<extra></extra>",
                )
            )
            range_fig.update_xaxes(type="log", title="M [h⁻¹ M☉]")
            range_fig.update_yaxes(title="fraction of sampled σ² from selected k")
            chart(
                set_plot(
                    range_fig,
                    "Selected Fourier contribution by mass",
                    "Higher bars identify the stored mass scales for which the selected sampled k interval contributes a larger fraction of σ².",
                ),
                300,
                "selected_k_mass_map",
            )
        except ValueError as exc:
            st.info(str(exc))
    st.markdown(
        '<div class="section-label">PIPELINE AT A GLANCE</div>', unsafe_allow_html=True
    )
    g = st.columns(2)
    with g[0]:
        chart(power_fig(result, [0]), 430, "dash_power")
    with g[1]:
        chart(sigma_fig(sigma, p, [0]), 430, "dash_sigma")
    g = st.columns(2)
    with g[0]:
        chart(integrand_fig(result, sigma, p), 430, "dash_integrand")
    with g[1]:
        chart(hmf_fig(run, [p["fitting"]], [p["single_z"]]), 430, "dash_hmf")


def graph_studio_view():
    st.markdown(
        '<div class="page-head"><span>GRAPH STUDIO</span><h2>Build the analysis canvas.</h2><p>Every graph is tied to the completed run. Use per-chart axis controls for custom paper ranges without changing the underlying arrays.</p></div>',
        unsafe_allow_html=True,
    )
    ready = require_run()
    if not ready:
        return
    run, result, sigma = ready
    p = run["params"]
    a, b, c = st.columns([1.4, 1, 1])
    layout = a.segmented_control(
        "Layout", ["Focus", "2 columns", "3 columns"], default="2 columns"
    )
    fits = b.multiselect(
        "Fits", FITTING_NAMES, default=[p["fitting"]], max_selections=5
    )
    zs = c.multiselect(
        "Redshifts",
        list(result["redshifts"]),
        default=[float(p["single_z"])],
        max_selections=5,
    )
    defaults = [
        "Matter P(k)",
        "Dimensionless Δ²(k)",
        "σ(M)",
        "σ integrand",
        "HMF",
        "Growth",
    ]
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
    a = saved["arrays"]
    p = saved["params"]
    h = float(saved["derived"]["h"])
    red = np.asarray(a.get("redshifts", [0.0]))
    pz = np.asarray(a.get("P_by_z", [a["P"]]))
    if window is None or window == saved.get(
        "window_type", p.get("window_type", "Top-hat")
    ):
        sig = np.asarray(a.get("sigma_by_z", [a["sigma"]]))
        deriv = np.asarray(a.get("dlnsigma_dlnM_by_z", [a["dlnsigma_dlnM"]]))
        R = a["R"]
        rho = saved["rho0"]
    else:
        key = f"alt_sigma_{saved['run_id']}_{window}"
        cached = st.session_state.get(key)
        if cached is None:
            grids = [
                sigma_grid(
                    a["M"],
                    a["k"],
                    row,
                    {"h": h, "Omega_m": p["Omega_m"]},
                    window,
                    int(p.get("quad_limit", 200)),
                )
                for row in pz
            ]
            cached = (
                np.asarray([g["sigma"] for g in grids]),
                np.asarray([g["dlnsigma_dlnM"] for g in grids]),
                grids[0]["R"],
                grids[0]["rho0"],
            )
            st.session_state[key] = cached
        sig, deriv, R, rho = cached
    return {
        "params": p,
        "power_result": {
            "k": a["k"],
            "P": a["P"],
            "P_by_z": pz,
            "redshifts": red,
            "growth_class": a.get("growth_class", np.ones_like(red)),
            "background_omega_m_by_z": a.get("background_omega_m_by_z", np.asarray([])),
            "derived": {"h": h},
        },
        "sigma_result": {
            "M_h": a["M_h"],
            "M": a["M"],
            "R": R,
            "sigma": sig[0],
            "sigma_by_z": sig,
            "dlnsigma_dlnM": deriv[0],
            "dlnsigma_dlnM_by_z": deriv,
            "redshifts": red,
            "rho0": rho,
            "window_type": window or saved.get("window_type", p.get("window_type")),
        },
    }


def _metric_axis(metric):
    return {
        "P(k)": ("k [Mpc⁻¹]", "P(k,z) [Mpc³]"),
        "Δ²(k)": ("k [Mpc⁻¹]", "Δ²(k,z)"),
        "Growth": ("redshift z", "D(z)/D(0)"),
        "σ(M)": ("M [h⁻¹ M☉]", "σ(M,z)"),
        "σ slope": ("M [h⁻¹ M☉]", "|d lnσ/d lnM|"),
        "HMF": ("M [h⁻¹ M☉]", "dn/dlnM [h³ Mpc⁻³]"),
        "Cumulative HMF": ("M [h⁻¹ M☉]", "n(M < m < Mmax) [h³ Mpc⁻³]"),
    }[metric]


def compare_metric(metric, selected, baseline, mode, z, fits, windows):
    f = go.Figure()
    curves = []
    largest = None
    crossings = []
    for saved in selected:
        relevant_windows = (
            windows if metric in {"σ(M)", "σ slope", "HMF", "Cumulative HMF"} else [""]
        )
        for window in relevant_windows:
            pipe = _saved_pipeline(saved, window or None)
            a = saved["arrays"]
            relevant_fits = fits if metric in {"HMF", "Cumulative HMF"} else [""]
            for fit in relevant_fits:
                calibrated = None
                if metric == "P(k)":
                    x, y = a["k"], a["P_by_z"][z_index(a.get("redshifts", [0]), z)]
                elif metric == "Δ²(k)":
                    x = a["k"]
                    y = (
                        x**3
                        * a["P_by_z"][z_index(a.get("redshifts", [0]), z)]
                        / (2 * np.pi**2)
                    )
                elif metric == "Growth":
                    x = a.get("redshifts", [0])
                    y = a.get("growth_class", np.ones_like(x))
                elif metric == "σ(M)":
                    j = z_index(pipe["sigma_result"]["redshifts"], z)
                    x = pipe["sigma_result"]["M_h"]
                    y = pipe["sigma_result"]["sigma_by_z"][j]
                elif metric == "σ slope":
                    j = z_index(pipe["sigma_result"]["redshifts"], z)
                    x = pipe["sigma_result"]["M_h"]
                    y = np.abs(pipe["sigma_result"]["dlnsigma_dlnM_by_z"][j])
                else:
                    try:
                        r = hmf_z(pipe, z, fit)
                    except (ValueError, FloatingPointError) as exc:
                        st.warning(f"{saved['name']} · {fit}: {exc}")
                        continue
                    x = r["M_h"]
                    y = (
                        cumulative_hmf(x, r["hmf"])
                        if metric == "Cumulative HMF"
                        else r["hmf"]
                    )
                    calibrated = np.asarray(
                        r["validity"]["calibrated_mask"], dtype=bool
                    )
                    if metric == "Cumulative HMF":
                        calibrated = np.logical_and.accumulate(calibrated[::-1])[::-1]
                curves.append(
                    {
                        "saved": saved,
                        "window": window,
                        "fit": fit,
                        "x": np.asarray(x),
                        "y": np.asarray(y),
                        "calibrated": calibrated,
                    }
                )
    if not curves:
        return f
    base_curves = [
        curve for curve in curves if curve["saved"]["run_id"] == baseline["run_id"]
    ]
    for i, curve in enumerate(curves[:36]):
        matches = [
            candidate
            for candidate in base_curves
            if candidate["window"] == curve["window"]
            and candidate["fit"] == curve["fit"]
        ]
        if not matches and mode != "Overlay":
            st.warning(
                f"No valid matching baseline for {curve['saved']['name']} · {curve['fit']}; comparison omitted."
            )
            continue
        base = matches[0] if matches else curve
        yy = transform_curve(curve["x"], curve["y"], base["x"], base["y"], mode)
        saved = curve["saved"]
        suffix = f" · {curve['window']}" if curve["window"] else ""
        suffix += f" · {curve['fit']}" if curve["fit"] else ""
        variant_index = (
            windows.index(curve["window"]) if curve["window"] in windows else 0
        ) + (fits.index(curve["fit"]) if curve["fit"] in fits else 0)
        dash = ["solid", "dash", "dot", "dashdot"][variant_index % 4]
        calibrated = curve["calibrated"]
        if calibrated is not None:
            calibrated = calibrated.copy()
            if mode != "Overlay":
                calibrated &= (
                    np.interp(
                        curve["x"],
                        base["x"],
                        base["calibrated"].astype(float),
                        left=0,
                        right=0,
                    )
                    == 1
                )
        f.add_trace(
            go.Scatter(
                x=curve["x"],
                y=yy if calibrated is None else np.where(calibrated, yy, np.nan),
                name=saved["name"] + suffix,
                line=dict(
                    color=saved.get("color", COLORS[i % len(COLORS)]),
                    width=3.2 if saved["run_id"] == baseline["run_id"] else 2.5,
                    dash=dash,
                ),
                opacity=1.0 if saved["run_id"] == baseline["run_id"] else 0.9,
                hovertemplate="x=%{x:.4e}<br>y=%{y:.4e}<extra></extra>",
            )
        )
        if calibrated is not None and not np.all(calibrated):
            f.add_trace(
                go.Scatter(
                    x=curve["x"],
                    y=np.where(calibrated, np.nan, yy),
                    name=saved["name"] + suffix + " · outside calibration",
                    line=dict(
                        color=saved.get("color", COLORS[i % len(COLORS)]),
                        width=2.5,
                        dash="dot",
                    ),
                    hovertemplate="x=%{x:.4e}<br>y=%{y:.4e}<br>outside calibration<extra></extra>",
                )
            )
        if mode != "Overlay" and saved["run_id"] != baseline["run_id"]:
            point = largest_deviation_point(
                curve["x"], yy, reference=1.0 if mode == "Ratio" else 0.0
            )
            if point and (
                largest is None
                or point["absolute_deviation"] > largest["point"]["absolute_deviation"]
            ):
                largest = {"point": point, "name": saved["name"]}
            crossings.extend(
                {**crossing, "name": saved["name"]}
                for crossing in reference_crossings(
                    curve["x"], yy, reference=1.0 if mode == "Ratio" else 0.0
                )
            )
    x_title, overlay_y = _metric_axis(metric)
    logx = metric != "Growth"
    logy = mode == "Overlay" and metric != "Growth"
    y_title = overlay_y if mode == "Overlay" else mode
    f.update_xaxes(type="log" if logx else "linear", title=x_title)
    f.update_yaxes(type="log" if logy else "linear", title=y_title)
    if mode == "Ratio":
        f.add_hline(y=1, line_color="#8ea1a9", line_dash="dot")
    elif mode in {
        "Fractional difference",
        "Percent difference",
        "Residual",
        "Standardized residual",
    }:
        f.add_hline(y=0, line_color="#8ea1a9", line_dash="dot")
        vals = _trace_values(f, "y")
        vals = vals[np.isfinite(vals)]
        if vals.size:
            extent = max(float(np.max(np.abs(vals))), 1e-4)
            f.update_yaxes(range=[-1.15 * extent, 1.15 * extent])
    if largest:
        point = largest["point"]
        f.add_annotation(
            x=np.log10(point["x"]) if logx else point["x"],
            y=point["value"],
            text=f"largest sampled deviation<br>{largest['name']}",
            showarrow=True,
            arrowhead=2,
            font=dict(size=10),
            bgcolor="rgba(9, 20, 28, .82)",
        )
    for crossing in crossings[:4]:
        if crossing["kind"] == "observed reference crossing":
            f.add_trace(
                go.Scatter(
                    x=[crossing["x"]],
                    y=[crossing["y"]],
                    mode="markers",
                    showlegend=False,
                    marker=dict(color="#ffb454", size=8, symbol="x"),
                    hovertemplate=f"observed reference crossing<br>{crossing['name']}<br>x=%{{x:.4e}}<extra></extra>",
                )
            )
        else:
            f.add_vrect(
                x0=crossing["x_lower"],
                x1=crossing["x_upper"],
                fillcolor="#ffb454",
                opacity=0.12,
                line_width=0,
                annotation_text="sampled crossing bracket",
                annotation_position="top left",
            )
    caption = f"{mode} · z = {z:g} · reference: {baseline['name']}. Gaps indicate undefined comparisons."
    if metric in {"HMF", "Cumulative HMF"}:
        caption += " Dotted segments include points outside fit calibration."
    if metric == "Cumulative HMF":
        caption += (
            " Counts end at each run's Mmax; match upper mass limits before comparing."
        )
    return set_plot(f, metric, caption)


def _point_comparison_curve(
    saved, metric: str, z: float, window: str, fit: str
) -> tuple[np.ndarray, np.ndarray, str]:
    """Return one saved observable for a bounded, exact-point comparison."""
    arrays = saved["arrays"]
    if metric == "P(k)":
        return (
            np.asarray(arrays["k"]),
            np.asarray(arrays["P_by_z"])[z_index(arrays.get("redshifts", [0]), z)],
            "Mpc³",
        )
    if metric == "Δ²(k)":
        x = np.asarray(arrays["k"])
        power = np.asarray(arrays["P_by_z"])[z_index(arrays.get("redshifts", [0]), z)]
        return x, x**3 * power / (2 * np.pi**2), "dimensionless"
    if metric == "Growth":
        return (
            np.asarray(arrays.get("redshifts", [0])),
            np.asarray(arrays.get("growth_class", [1])),
            "dimensionless",
        )
    pipeline = _saved_pipeline(saved, window)
    sigma_result = pipeline["sigma_result"]
    if metric == "σ(M)":
        index = z_index(sigma_result["redshifts"], z)
        return (
            np.asarray(sigma_result["M_h"]),
            np.asarray(sigma_result["sigma_by_z"])[index],
            "dimensionless",
        )
    if metric == "σ slope":
        index = z_index(sigma_result["redshifts"], z)
        return (
            np.asarray(sigma_result["M_h"]),
            np.abs(np.asarray(sigma_result["dlnsigma_dlnM_by_z"])[index]),
            "dimensionless",
        )
    hmf = hmf_z(pipeline, z, fit)
    values = (
        cumulative_hmf(hmf["M_h"], hmf["hmf"])
        if metric == "Cumulative HMF"
        else hmf["hmf"]
    )
    return np.asarray(hmf["M_h"]), np.asarray(values), "h³ Mpc⁻³"


def compare_view():
    st.markdown(
        '<div class="page-head"><span>COMPARE LAB</span>'
        "<h2>Compare cosmologies</h2>"
        "<p>Overlay is the default. Ratios and residuals use a matching "
        "baseline curve for every window and fitting-function combination.</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    runs = analysis_runs()

    if len(runs) < 2:
        st.info(
            "Run at least two cosmologies. Try LCDM, run it, then EDE and run again."
        )
        return

    run_map = {str(run["run_id"]): run for run in runs}

    run_ids = list(run_map)

    labels = {run_id: get_run_label(run_map[run_id]) for run_id in run_ids}

    default_run_ids = [str(run["run_id"]) for run in runs[-min(3, len(runs)) :]]

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

    selected_ids = [run_id for run_id in selected_ids if run_id in run_map]

    selected = [run_map[run_id] for run_id in selected_ids]

    if not selected:
        return

    c = st.columns(4)

    preferred_baseline_id = next(
        (run_id for run_id in selected_ids if run_map[run_id].get("is_baseline")),
        selected_ids[0],
    )

    _prepare_scalar_widget_state(
        "compare_baseline_id",
        selected_ids,
    )

    baseline_kwargs = {}

    if "compare_baseline_id" not in st.session_state:
        baseline_kwargs["index"] = selected_ids.index(preferred_baseline_id)

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
            "Residual",
            "Standardized residual",
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
        st.error("The selected runs have no common sampled redshift.")
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

    panel_kwargs = {} if "compare_panel_count" in st.session_state else {"default": "2"}
    ncol = c[3].segmented_control(
        "Panels",
        ["1", "2", "3"],
        **panel_kwargs,
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

    count = len(selected) * max(1, len(windows)) * max(1, len(fits))

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
                    fits or [baseline["params"]["fitting"]],
                    windows or [baseline["params"]["window_type"]],
                ),
                500,
                key=(f"compare_chart_{COMPARE_METRICS.index(metric)}"),
                axis_controls=True,
            )

    candidates = [run for run in selected if run["run_id"] != baseline["run_id"]]
    if candidates:
        with st.expander("Compare at a physical point", expanded=False):
            st.caption(
                "A bounded numerical comparison of stored curves. Ratios are unit-free; the displayed values retain their observable units. This is not an uncertainty interval or a calibration claim."
            )
            point_columns = st.columns(4)
            point_metric = point_columns[0].selectbox(
                "Observable", metrics or COMPARE_METRICS, key="point_compare_metric"
            )
            candidate_ids = [str(run["run_id"]) for run in candidates]
            _prepare_scalar_widget_state("point_compare_candidate_id", candidate_ids)
            point_candidate_id = point_columns[1].selectbox(
                "Candidate",
                candidate_ids,
                format_func=lambda run_id: run_map[run_id]["name"],
                key="point_compare_candidate_id",
            )
            point_candidate = run_map[point_candidate_id]
            point_window = windows[0] if windows else baseline["params"]["window_type"]
            point_fit = fits[0] if fits else baseline["params"]["fitting"]
            if point_metric in {"σ(M)", "σ slope", "HMF", "Cumulative HMF"}:
                point_window = point_columns[2].selectbox(
                    "Smoothing window", windows or WINDOWS, key="point_compare_window"
                )
            else:
                point_columns[2].caption(
                    "Smoothing window is not used for this observable."
                )
            if point_metric in {"HMF", "Cumulative HMF"}:
                point_fit = point_columns[3].selectbox(
                    "HMF fit", fits or FITTING_NAMES, key="point_compare_fit"
                )
            else:
                point_columns[3].caption("HMF fit is not used for this observable.")
            try:
                candidate_x, candidate_y, value_unit = _point_comparison_curve(
                    point_candidate, point_metric, float(z), point_window, point_fit
                )
                baseline_x, baseline_y, baseline_unit = _point_comparison_curve(
                    baseline, point_metric, float(z), point_window, point_fit
                )
                if baseline_unit != value_unit:
                    raise ValueError(
                        "Candidate and baseline use incompatible observable units."
                    )
                lower = max(float(np.min(candidate_x)), float(np.min(baseline_x)))
                upper = min(float(np.max(candidate_x)), float(np.max(baseline_x)))
                if lower >= upper:
                    raise ValueError(
                        "The candidate and baseline have no overlapping stored coordinate range."
                    )
                default_point = (
                    float(np.sqrt(lower * upper))
                    if lower > 0 and upper > 0
                    else (lower + upper) / 2
                )
                requested_point = st.number_input(
                    f"Coordinate ({_metric_axis(point_metric)[0]})",
                    min_value=lower,
                    max_value=upper,
                    value=default_point,
                    format="%.6g",
                    key="point_compare_coordinate",
                )
                comparison = compare_at_point(
                    candidate_x, candidate_y, baseline_x, baseline_y, requested_point
                )
                st.dataframe(
                    pd.DataFrame(
                        [
                            {
                                "observable": point_metric,
                                "coordinate": comparison["point"],
                                "candidate": point_candidate["name"],
                                "candidate value": comparison["candidate_value"],
                                "baseline": baseline["name"],
                                "baseline value": comparison["baseline_value"],
                                "value unit": value_unit,
                                "candidate / baseline": comparison["ratio"],
                                "difference (%)": comparison["percent_difference"],
                            }
                        ]
                    ),
                    width="stretch",
                    hide_index=True,
                )
                st.caption(
                    f"Candidate value: {comparison['candidate_method']}. Baseline value: {comparison['baseline_method']}. {comparison['scope_limit']}"
                )
                uncertainty_rows = [
                    row
                    for row in uncertainty_inventory(point_candidate)
                    if row["source"]
                    in {
                        "Numerical integration",
                        "k-range truncation",
                        "HMF fit calibration",
                        "Cosmology support",
                        "Theory/model uncertainty",
                    }
                ]
                st.caption("Uncertainty context for the candidate run")
                st.dataframe(
                    pd.DataFrame(uncertainty_rows), width="stretch", hide_index=True
                )
            except (ValueError, FloatingPointError) as exc:
                st.info(f"This point comparison is unavailable: {exc}")

    table = []

    for run in selected:
        p = run["params"]

        table.append(
            {
                "run": run["name"],
                "baseline": (run["run_id"] == baseline["run_id"]),
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
    with st.expander("Explain the differences, including what could mislead you"):
        candidates = [run for run in selected if run["run_id"] != baseline["run_id"]]
        if not candidates:
            st.caption(
                "Select a candidate run to generate a parameter-difference explanation."
            )
        for candidate in candidates:
            st.markdown(f"#### {candidate['name']} versus {baseline['name']}")
            steps = explain_change(baseline["params"], candidate["params"])
            if not steps:
                st.caption(
                    "No currently supported causal parameter difference was found. The runs may differ only in settings that need a dedicated analysis."
                )
            for step in steps:
                st.markdown(
                    f"**{step.changed.capitalize()}** directly changes `{step.direct_equation}`. The first expected signal is **{step.first_signal}**; downstream this can affect {step.downstream}. **Robust:** {step.robust}. **Model/numerical limit:** {step.caveat}. **Falsification check:** {step.falsifier}."
                )


def analysis_runs() -> list[dict]:
    runs = load_all_runs()
    excluded = [
        run
        for run in runs
        if not run.get("arrays")
        or run.get("integrity_status", {}).get("state") == "invalid"
    ]
    if excluded:
        st.warning(
            f"{len(excluded)} saved run(s) excluded because their data are missing or failed integrity checks. Inspect them in Runs + export."
        )
    excluded_ids = {run["run_id"] for run in excluded}
    return [run for run in runs if run["run_id"] not in excluded_ids]


def sensitivity_view():
    st.markdown(
        '<div class="page-head"><span>SENSITIVITY EXPLORER</span><h2>Measure one controlled change at a time.</h2><p>Use saved experiments as explicit finite-difference evidence. HaloForge excludes confounded runs instead of pretending they isolate one parameter.</p></div>',
        unsafe_allow_html=True,
    )
    runs = analysis_runs()
    if len(runs) < 2:
        st.info(
            "Save a baseline and at least one run that changes exactly one numeric parameter. The Notebook can keep their lineage and hypotheses together."
        )
        return
    run_map = {run["run_id"]: run for run in runs}
    default_baseline = next(
        (run["run_id"] for run in runs if run.get("is_baseline")), runs[0]["run_id"]
    )
    baseline_id = st.selectbox(
        "Baseline experiment",
        list(run_map),
        index=list(run_map).index(default_baseline),
        format_func=lambda rid: run_map[rid]["name"],
        key="sensitivity_baseline",
    )
    baseline = run_map[baseline_id]
    numeric_parameters = [
        key
        for key, value in baseline["params"].items()
        if isinstance(value, (int, float))
        and not isinstance(value, bool)
        and np.isfinite(value)
        and key not in {"mode", "z_presets_selected"}
    ]
    columns = st.columns(4)
    parameter = columns[0].selectbox(
        "One changed parameter", numeric_parameters, key="sensitivity_parameter"
    )
    observable = columns[1].selectbox(
        "Observable", ["σ(M)", "σ₈"], key="sensitivity_observable"
    )
    redshifts = [float(value) for value in baseline["arrays"].get("redshifts", [0.0])]
    redshift = columns[2].selectbox("Redshift", redshifts, key="sensitivity_redshift")
    selected_mass = 10 ** float(baseline["params"].get("selected_mass_exp", 12))
    mass = columns[3].number_input(
        "Mass [h⁻¹ M☉]",
        min_value=1e5,
        value=float(selected_mass),
        format="%.3e",
        disabled=observable != "σ(M)",
        key="sensitivity_mass",
    )
    try:
        report = controlled_sensitivity(
            baseline,
            [run for run in runs if run["run_id"] != baseline_id],
            parameter,
            observable,
            float(mass) if observable == "σ(M)" else None,
            float(redshift),
        )
    except ValueError as exc:
        st.warning(str(exc))
        return

    st.caption(report["scope_limit"])
    if not report["rows"]:
        st.warning(
            "No saved runs isolate this parameter at the selected redshift. Create a one-at-a-time experiment or choose a different baseline."
        )
    else:
        frame = pd.DataFrame(report["rows"])
        st.dataframe(frame, width="stretch", hide_index=True)
        figure = go.Figure()
        figure.add_trace(
            go.Scatter(
                x=[report["baseline_value"]],
                y=[report["baseline_observable"]],
                mode="markers",
                marker=dict(color=COLORS[1], size=12, symbol="diamond"),
                name="baseline",
                hovertemplate=f"{parameter}=%{{x:.5g}}<br>{observable}=%{{y:.5g}}<extra></extra>",
            )
        )
        figure.add_trace(
            go.Scatter(
                x=frame["parameter_value"],
                y=frame["observable_value"],
                mode="lines+markers",
                line=dict(color=COLORS[0], width=2.5),
                marker=dict(size=9),
                name="controlled candidates",
                hovertemplate=f"{parameter}=%{{x:.5g}}<br>{observable}=%{{y:.5g}}<extra></extra>",
            )
        )
        figure.update_xaxes(title=parameter)
        figure.update_yaxes(title=observable)
        chart(
            set_plot(
                figure,
                "Controlled finite sensitivity",
                "A finite-difference response from saved controlled runs. It is not a posterior distribution or a surrogate-model prediction.",
            ),
            430,
            "sensitivity_chart",
        )
        with st.expander("Counterfactual from saved evidence", expanded=False):
            st.caption(
                "Find the smallest already-saved, one-parameter perturbation that reaches an observable target. No unrun setting is interpolated or predicted."
            )
            counterfactual_columns = st.columns(2)
            target_percent = counterfactual_columns[0].number_input(
                "Target observable change (%)",
                min_value=0.1,
                max_value=1000.0,
                value=10.0,
                step=0.5,
                key="counterfactual_target_percent",
            )
            direction = counterfactual_columns[1].selectbox(
                "Target direction",
                ["either", "increase", "decrease"],
                key="counterfactual_direction",
            )
            counterfactual = smallest_saved_counterfactual(
                report, float(target_percent) / 100, direction
            )
            st.caption(counterfactual["scope_limit"])
            if counterfactual["status"] == "saved_match":
                row = counterfactual["selected"]
                parameter_change_label = (
                    f"{row['parameter_fractional_change']:.2%}"
                    if row["parameter_fractional_change"] is not None
                    else f"{row['parameter_absolute_change']:+.5g} in absolute units"
                )
                st.success(
                    f"Saved match: {row['run']} changes {parameter} by {parameter_change_label} "
                    f"and changes {observable} by {row['observable_fractional_change']:.2%}."
                )
            else:
                st.info(
                    "No saved controlled experiment reaches this target. Create another deliberately controlled run; HaloForge will not estimate an unrun threshold."
                )
    if report["excluded"]:
        with st.expander(f"Excluded comparisons ({len(report['excluded'])})"):
            st.dataframe(
                pd.DataFrame(report["excluded"]), width="stretch", hide_index=True
            )


def experiment_design_view():
    st.markdown(
        '<div class="page-head"><span>DESIGN AN EXPERIMENT</span><h2>Start from a scientific question, not a random slider.</h2><p>Each plan makes one interpretable change, states what remains fixed, and names the evidence that could weaken the interpretation.</p></div>',
        unsafe_allow_html=True,
    )
    baseline_runs = [r for r in run_storage.load_all_runs() if r.get("is_baseline")]
    starting_choice = st.radio(
        "Start this plan from",
        ["Active staged parameters", "Named baseline", "Canonical Planck ΛCDM preset"],
        horizontal=True,
        key="design_plan_starting_choice",
    )
    selected_baseline_name = None
    if starting_choice == "Named baseline":
        if baseline_runs:
            baseline_run = st.selectbox(
                "Baseline run",
                baseline_runs,
                format_func=lambda r: r.get("name", "Untitled"),
                key="design_selected_baseline",
            )
            base_params = deepcopy(baseline_run.get("params", DEFAULT_PARAMS))
            selected_baseline_name = baseline_run.get("name")
            start_source = "named_baseline"
        else:
            st.info(
                "No saved baseline run found. Using canonical Planck ΛCDM baseline."
            )
            base_params = deepcopy(DEFAULT_PARAMS)
            start_source = "canonical_preset"
    elif starting_choice == "Canonical Planck ΛCDM preset":
        base_params = deepcopy(DEFAULT_PARAMS)
        start_source = "canonical_preset"
    else:
        base_params = deepcopy(get_params())
        start_source = "active"

    goal = st.selectbox("Scientific goal", list(PLANS), key="design_goal")
    plan = design_experiment(
        goal,
        base_params,
        starting_source=start_source,
        baseline_name=selected_baseline_name,
    )
    st.caption("Starting point: " + plan["starting_point"])
    st.markdown(f"### {plan['question']}")
    ede_retention_requires_confirmation = bool(
        start_source == "active"
        and base_params.get("enable_ede")
        and goal == "Tilt and low-mass structure"
    )
    retain_ede_confirmed = True
    if ede_retention_requires_confirmation:
        st.warning(
            "Your active parameters include Early Dark Energy. This candidate would be **EDE + tilt**, not ΛCDM + tilt. "
            "Choose the canonical ΛCDM starting point for a clean tilt experiment."
        )
        retain_ede_confirmed = st.checkbox(
            "I intentionally want an EDE + tilt experiment and will interpret it as a two-physics scenario.",
            key="design_confirm_ede_tilt",
        )
    columns = st.columns(2)
    with columns[0]:
        st.markdown("#### Candidate parameter difference")
        st.dataframe(
            pd.DataFrame(
                [
                    {"parameter": key, **values}
                    for key, values in plan["parameter_diff"].items()
                ]
            ),
            width="stretch",
            hide_index=True,
        )
        st.markdown("#### Hold fixed")
        st.markdown("\n".join(f"- {item}" for item in plan["hold_fixed"]))
    with columns[1]:
        st.markdown("#### Predict before calculating")
        st.info(plan["prediction"])
        st.markdown("#### Inspect in this order")
        st.markdown("\n".join(f"1. {item}" for item in plan["inspect"]))
    st.warning("Caveat: " + plan["caveat"])
    st.caption(plan["scope_limit"])
    with st.expander("Complete candidate cosmology", expanded=False):
        st.caption("The full staged parameter set, including values inherited from the selected starting source.")
        st.dataframe(
            pd.DataFrame(
                [
                    {"parameter": key, "value": value}
                    for key, value in sorted(
                        {**base_params, **plan["candidate_parameters"]}.items()
                    )
                ]
            ),
            width="stretch",
            hide_index=True,
        )
    if st.button(
        "Stage this controlled candidate",
        type="primary",
        disabled=not retain_ede_confirmed,
    ):
        st.session_state["params"] = deepcopy(base_params)
        st.session_state["params"].update(plan["candidate_parameters"])
        params = st.session_state["params"]
        st.session_state["run_name_draft"] = "Planned — " + goal
        st.session_state["hf_research_question"] = plan["question"]
        st.session_state["hf_hypothesis"] = plan["prediction"]
        save_draft_params(params)
        st.success(
            "Candidate staged locally. Open the controls in this Research mode to review, record a prediction, then calculate deliberately."
        )


def benchmark_view():
    st.markdown(
        '<div class="page-head"><span>BENCHMARK LAB</span><h2>Check numerical agreement</h2><p>Compare saved σ₈ values with adaptive integration of the same sampled power spectrum.</p></div>',
        unsafe_allow_html=True,
    )
    ready = require_run()
    if not ready:
        return
    run, _result, _sigma = ready
    with st.expander("Canonical validation-case matrix", expanded=False):
        st.caption(
            "These reproducible settings cover the declared regimes. They are not frozen external reference outputs, so a configured case remains distinct from independently validated agreement."
        )
        case_map = {case.identifier: case for case in CANONICAL_VALIDATION_CASES}
        case_id = st.selectbox(
            "Validation case",
            list(case_map),
            format_func=lambda identifier: case_map[identifier].title,
            key="canonical_validation_case",
        )
        selected_case = canonical_case(case_id)
        selected_assessment = assess_canonical_case(run["params"], selected_case)
        st.markdown(
            f"**Target:** {selected_case.target}\n\n**Evidence to record:** {selected_case.expected_evidence}\n\n**Boundary:** {selected_case.boundary}"
        )
        if st.button("Stage this validation case", key="stage_validation_case"):
            st.session_state["pending_parameter_stage"] = {
                "overrides": selected_case.parameter_overrides,
                "name": "Validation — " + selected_case.title,
                "question": selected_case.target,
                "hypothesis": selected_case.expected_evidence,
            }
            st.rerun()
        st.caption(
            "Selected run: "
            + (
                "configuration matches this case."
                if selected_assessment["configured"]
                else "configuration does not yet match this case."
            )
            + " "
            + selected_assessment["reference_status"]
            + "."
        )
        matrix = pd.DataFrame(canonical_case_rows(run["params"]))
        st.dataframe(
            matrix[["title", "configured", "target", "reference_status"]],
            width="stretch",
            hide_index=True,
        )
        st.caption("Registry version: " + CANONICAL_CASES_VERSION)
    st.warning(
        "This check tests integration of the saved spectrum. External solver agreement and empirical HMF calibration require separate evidence."
    )
    run_key = st.session_state.get("current_run_id", "active")
    result_key = f"internal_benchmark_{run_key}"
    if st.button("Benchmark this run", type="primary"):
        benchmark_started = perf_counter()
        try:
            with st.spinner(
                "Reintegrating σ₈ with adaptive quadrature at each sampled redshift…"
            ):
                report = internal_sigma8_benchmark(run)
                st.session_state[result_key] = report
                stored = run_storage.load_run(run_key)
                if stored is not None:
                    stored.setdefault("benchmarks", {})[report["benchmark_version"]] = (
                        report
                    )
                    scientific_run = pipeline_from_saved_run(stored)
                    try:
                        hmf_validity = hmf_z(
                            scientific_run,
                            float(stored["params"].get("single_z", 0.0)),
                            stored["params"]["fitting"],
                        ).get("validity")
                    except (ValueError, FloatingPointError):
                        hmf_validity = None
                    stored["scientific_validity"] = scientific_validity_record(
                        scientific_run, hmf_validity, report
                    )
                    update_run_metadata(stored)
                    generate_run_exports(stored)
                record_local_diagnostic(
                    run_storage.DATA_ROOT,
                    bool(st.session_state.get("hf_local_diagnostics_enabled", False)),
                    "benchmark_completed",
                    duration_seconds=perf_counter() - benchmark_started,
                )
        except (ValueError, FloatingPointError) as exc:
            show_failure(exc)
    report = st.session_state.get(result_key)
    if report is None:
        stored = run_storage.load_run(run_key)
        if stored is not None:
            report = stored.get("benchmarks", {}).get("haloforge-internal-sigma8-v1")
    if not report:
        st.info(
            "Run the benchmark to record exact discrepancies by redshift. Failed agreement remains visible in the result table."
        )
        return
    st.caption(
        f"Reference: {report['benchmark_version']} · tolerance: {report['relative_tolerance']:.1e} · elapsed: {report['elapsed_seconds']:.2f} s"
    )
    table = pd.DataFrame(report["rows"])
    st.dataframe(table, width="stretch", hide_index=True)
    if (table["status"] == "fail").any():
        st.error(
            "At least one benchmark row failed. Treat the discrepancy as a finding; inspect sampled resolution and integration settings before using the result."
        )
    elif (table["status"] == "review").any():
        st.warning(
            "The numerical values agree within tolerance, but the adaptive reference reported a quadrature warning. Inspect the reference-warning column before relying on this check."
        )
    else:
        st.success(
            "The fixed sampled-grid σ₈ result agrees with this independent adaptive quadrature check within the declared tolerance. This does not replace external validation."
        )
    st.caption(report["scope_limit"])


def performance_view():
    st.markdown(
        '<div class="page-head"><span>PERFORMANCE LAB</span><h2>Measure the core. Do not guess at responsiveness.</h2><p>Run a reproducible timing and allocation probe on this saved calculation. It is kept separate from scientific agreement checks.</p></div>',
        unsafe_allow_html=True,
    )
    ready = require_run()
    if not ready:
        return
    run, _result, _sigma = ready
    run_key = st.session_state.get("current_run_id", "active")
    report_key = f"performance_benchmark_{run_key}"
    repetitions = st.slider(
        "Core rebuild repetitions", min_value=1, max_value=10, value=3
    )
    if st.button("Profile this run", type="primary"):
        with st.spinner("Rebuilding σ(M) and the eligible HMF core locally…"):
            report = profile_core_pipeline(run, repetitions=repetitions)
        st.session_state[report_key] = report
        stored = run_storage.load_run(run_key)
        if stored is not None:
            stored.setdefault("performance_benchmarks", {})[
                report["benchmark_version"]
            ] = report
            update_run_metadata(stored)
            generate_run_exports(stored)
    report = st.session_state.get(report_key)
    if report is None:
        stored = run_storage.load_run(run_key)
        if stored is not None:
            report = stored.get("performance_benchmarks", {}).get(
                PERFORMANCE_BENCHMARK_VERSION
            )
    if not report:
        st.info(
            "Profile this completed run to save local core performance evidence alongside its scientific benchmark results."
        )
        return
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "probe": "σ(M) + eligible HMF core rebuild",
                    "median seconds": report["core_rebuild_median_seconds"],
                    "budget seconds": report["core_rebuild_budget_seconds"],
                    "state": report["core_rebuild_state"],
                    "peak Python allocation MiB": report["python_peak_allocation_mib"],
                    "budget MiB": report["python_peak_allocation_budget_mib"],
                    "allocation state": report["python_peak_allocation_state"],
                }
            ]
        ),
        width="stretch",
        hide_index=True,
    )
    st.caption(report["scope_limit"])
    with st.expander("Budgets not measured by this probe"):
        st.markdown("\n".join(f"- {item}" for item in report["unmeasured_budgets"]))


def convergence_view():
    st.markdown(
        '<div class="page-head"><span>CONVERGENCE LAB</span><h2>Ask whether a conclusion survives numerical choices.</h2><p>These checks interrogate the sampled calculation. They do not turn finite-range agreement into a physical or calibration guarantee.</p></div>',
        unsafe_allow_html=True,
    )
    ready = require_run()
    if not ready:
        return
    run, _result, sigma = ready
    diagnostics = sigma.get("numerical_diagnostics", {})
    if not diagnostics:
        st.warning(
            "This restored run has no saved numerical diagnostics. Recalculate it to record sampled-range evidence."
        )
        return
    st.caption(diagnostics.get("scope_limit", ""))
    rows = []
    for label, payload in (
        ("Sampled-range coverage", diagnostics.get("coverage", {})),
        ("Remove low-k endpoint", diagnostics.get("low_k_truncation", {})),
        ("Remove high-k endpoint", diagnostics.get("high_k_truncation", {})),
    ):
        rows.append(
            {
                "check": label,
                "state": payload.get("status", "not evaluated"),
                "maximum fractional σ change": payload.get(
                    "maximum_fractional_sigma_change"
                ),
                "mass at maximum [h⁻¹ M☉]": payload.get("mass_at_maximum_Msun"),
                "interpretation": "; ".join(payload.get("notes", []))
                or payload.get("reason", ""),
            }
        )
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    run_id = st.session_state.get("current_run_id", "")
    stored = run_storage.load_run(run_id) if run_id else None
    benchmark = (
        stored.get("benchmarks", {}).get("haloforge-internal-sigma8-v1")
        if stored
        else None
    )
    st.markdown("### Independent integration check")
    if benchmark:
        benchmark_table = pd.DataFrame(benchmark.get("rows", []))
        st.dataframe(benchmark_table, width="stretch", hide_index=True)
        st.caption(
            "Benchmark reference version: "
            + str(benchmark.get("benchmark_version", "unavailable"))
        )
    else:
        st.info(
            "No saved adaptive σ₈ benchmark is attached to this run. Run it in Benchmark Lab before treating fixed-grid agreement as checked."
        )
    st.markdown("### Next controlled tests")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "change only": "k_min",
                    "question": "Do the largest smoothing radii keep the same conclusion?",
                    "success condition": "Endpoint sensitivity stays below your predeclared tolerance.",
                },
                {
                    "change only": "k_max",
                    "question": "Does the selected low-mass scale retain its conclusion?",
                    "success condition": "High-k endpoint removal is immaterial at that mass.",
                },
                {
                    "change only": "k_points / mass_points",
                    "question": "Are curve and derivative details stable under resolution?",
                    "success condition": "A saved controlled sweep shows stable target observables.",
                },
            ]
        ),
        width="stretch",
        hide_index=True,
    )
    st.warning(
        "A check that passes only at one mass, one redshift, or one fit is not evidence for every conclusion. Record the exact tested claim and numerical settings in the Notebook."
    )


@st.cache_data(show_spinner=False, max_entries=3)
def shared_fourier_seed(box: float, n: int, seed: int):
    return _shared_fourier_seed(box, n, seed)


def gaussian_field_slice(k, power, box, n, smoothing, modes, kk):
    return _gaussian_field_slice(k, power, box, n, smoothing, modes, kk)


def structure_view():
    st.markdown(
        '<div class="page-head"><span>STRUCTURE FIELD</span><h2>Compare linear density fields</h2><p>A periodic 3D Gaussian linear-density realization is filtered by each run’s computed P(k,z); all panels use the same Fourier seed and a common normalization.</p></div>',
        unsafe_allow_html=True,
    )
    runs = analysis_runs()
    if not runs:
        st.info("Run a cosmology first.")
        return
    run_map = {r["run_id"]: r for r in runs}
    default_ids = [r["run_id"] for r in runs[-min(3, len(runs)) :]]
    with st.form("structure_controls"):
        selected_ids = st.multiselect(
            "Runs",
            list(run_map),
            default=default_ids,
            format_func=lambda rid: run_map[rid]["name"],
            max_selections=4,
        )
        selected = [run_map[rid] for rid in selected_ids]
        common = (
            sorted(
                set.intersection(
                    *[
                        set(map(float, r["arrays"].get("redshifts", [0])))
                        for r in selected
                    ]
                )
            )
            if selected
            else []
        )
        c = st.columns(5)
        z = c[0].selectbox("Redshift", common or [0.0])
        box = c[1].slider("Box size [Mpc]", 50, 1000, 300, 10)
        smoothing = c[2].slider("Gaussian smoothing [Mpc]", 0.2, 12.0, 2.0, 0.2)
        n = c[3].select_slider("Grid", options=[64, 96, 128], value=96)
        seed = c[4].number_input("Shared seed", 0, 999999, 271828)
        baseline_id = st.selectbox(
            "Field baseline",
            selected_ids or [""],
            format_func=lambda rid: run_map[rid]["name"]
            if rid in run_map
            else "No run",
        )
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
        a = r["arrays"]
        j = z_index(a.get("redshifts", [0]), z)
        fields.append(
            gaussian_field_slice(
                a["k"], a["P_by_z"][j], float(box), int(n), float(smoothing), modes, kk
            )
        )
    baseline_index = selected.index(baseline)
    base_rms = float(np.sqrt(np.mean(fields[baseline_index] ** 2)))
    if base_rms <= 0:
        st.error(
            "The baseline has no resolved field power. Increase the box size or extend the stored spectrum’s k range."
        )
        return
    normalized = [field / base_rms for field in fields]
    all_values = np.concatenate([np.abs(field).ravel() for field in normalized])
    color_extent = max(float(np.percentile(all_values, 99.5)), 1.0)
    coords = (np.arange(n) + 0.5) * box / n

    st.caption(
        "Colors share the baseline slice RMS. The color limits saturate the largest 0.5% of absolute field values; hover values retain their full amplitude."
    )
    cols = st.columns(min(2, len(selected)))
    for i, (r, field) in enumerate(zip(selected, normalized)):
        fig = go.Figure(
            go.Heatmap(
                x=coords,
                y=coords,
                z=field.T,
                colorscale=[
                    [0, "#07131b"],
                    [0.25, "#123d56"],
                    [0.5, "#c5d2c9"],
                    [0.72, "#e8a349"],
                    [1, "#fff1bf"],
                ],
                zmin=-color_extent,
                zmax=color_extent,
                colorbar=dict(title="δ/σbase", thickness=9),
            )
        )
        fig.update_xaxes(
            title="x [Mpc]", showgrid=False, range=[0, box], constrain="domain"
        )
        fig.update_yaxes(
            title="y [Mpc]",
            showgrid=False,
            scaleanchor="x",
            range=[0, box],
            constrain="domain",
        )
        set_plot(
            fig,
            r["name"],
            "Central slice of the same 3D Fourier realization. Colors are normalized by the baseline RMS, not separately by each run.",
        )
        with cols[i % len(cols)]:
            chart(
                fig,
                440,
                key=f"field_{r['run_id']}",
            )
    differences = [
        (i, field - normalized[baseline_index])
        for i, field in enumerate(normalized)
        if i != baseline_index
    ]
    if differences:
        st.markdown(
            '<div class="section-label">DIFFERENCE FROM BASELINE</div>',
            unsafe_allow_html=True,
        )
        dcols = st.columns(min(2, len(differences)))
        st.caption(
            "Difference maps share a separate color scale, saturated at the 99.5th percentile of absolute differences."
        )
        diff_extent = max(
            float(
                np.percentile(
                    np.concatenate([np.abs(diff).ravel() for _, diff in differences]),
                    99.5,
                )
            ),
            1e-4,
        )
        for panel, (i, diff) in enumerate(differences):
            fig = go.Figure(
                go.Heatmap(
                    x=coords,
                    y=coords,
                    z=diff.T,
                    colorscale="RdBu_r",
                    zmid=0,
                    zmin=-diff_extent,
                    zmax=diff_extent,
                    colorbar=dict(title="Δδ/σbase", thickness=9),
                )
            )
            fig.update_xaxes(
                title="x [Mpc]", showgrid=False, range=[0, box], constrain="domain"
            )
            fig.update_yaxes(
                title="y [Mpc]",
                showgrid=False,
                scaleanchor="x",
                range=[0, box],
                constrain="domain",
            )
            set_plot(
                fig,
                f"{selected[i]['name']} − {baseline['name']}",
                "Matched-mode difference. Because phases and normalization are shared, both shape and amplitude changes remain visible.",
            )
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
            fig.add_trace(
                go.Scatter(
                    x=coords,
                    y=field[central],
                    name=selected[i]["name"],
                    line=dict(color=selected[i].get("color", COLORS[i]), width=2.5),
                )
            )
        fig.update_xaxes(title="distance [Mpc]")
        fig.update_yaxes(title="δ/σbaseline")
        set_plot(
            fig,
            "Matched central slice",
            "A one-dimensional cut through the exact same Fourier phases in every selected run.",
        )
        chart(fig, 430, "field_slice", axis_controls=True)
    with right:
        rows = []
        base_flat = fields[baseline_index].ravel()
        for r, field in zip(selected, fields):
            flat = field.ravel()
            rows.append(
                {
                    "run": r["name"],
                    "RMS δ": float(np.sqrt(np.mean(flat**2))),
                    "RMS / baseline": float(np.sqrt(np.mean(flat**2)) / base_rms),
                    "correlation with baseline": float(
                        np.corrcoef(flat, base_flat)[0, 1]
                    ),
                    "difference RMS / baseline": float(
                        np.sqrt(np.mean((flat - base_flat) ** 2)) / base_rms
                    ),
                }
            )
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        fundamental = 2 * np.pi / box
        nyquist = np.pi * n / box
        st.markdown(
            f'<div class="lesson-card compact"><span>FOURIER COVERAGE</span><h3>{n}³ periodic grid</h3><p>kfund = {fundamental:.4f} Mpc⁻¹<br>kNyquist = {nyquist:.4f} Mpc⁻¹<br>Gaussian smoothing = {smoothing:.2f} Mpc</p><b>The quantitative table preserves the actual linear-field RMS. The maps are a visualization, not an N-body halo catalogue.</b></div>',
            unsafe_allow_html=True,
        )


def limitations_view():
    st.markdown(
        '<div class="page-head"><span>KNOWN LIMITATIONS</span><h2>What could make this wrong?</h2><p>These are open, versioned limits—not hidden fine print. A successful run does not override them.</p></div>',
        unsafe_allow_html=True,
    )
    st.caption(f"Registry version: {LIMITATIONS_VERSION}")
    rows = limitations_rows()
    st.dataframe(
        pd.DataFrame(rows),
        width="stretch",
        hide_index=True,
        column_config={
            "statement": st.column_config.TextColumn("Limitation", width="large"),
            "impact": st.column_config.TextColumn("Impact", width="large"),
            "mitigation": st.column_config.TextColumn("What to do", width="large"),
        },
    )
    st.download_button(
        "Download known limitations",
        json.dumps(rows, indent=2),
        "haloforge_known_limitations.json",
        "application/json",
    )
    st.info(
        "A limitation being listed is not a user error. It is evidence that should shape the next calculation, comparison, or external review."
    )


def teach_view():
    st.markdown(
        '<div class="page-head"><span>TEACH THIS TOMORROW</span><h2>Prepared labs that teach evidence, not slider play.</h2><p>These materials are local and private by default. They do not claim classroom accounts, shared grading, or student analytics.</p></div>',
        unsafe_allow_html=True,
    )
    module_map = {module.identifier: module for module in MODULES}
    identifier = st.selectbox(
        "Prepared module",
        list(module_map),
        format_func=lambda key: module_map[key].title,
        key="teaching_module",
    )
    module = get_module(identifier)
    stats = st.columns(3)
    stats[0].metric("Estimated time", f"{module.duration_minutes} min")
    stats[1].metric("Level", module.level)
    stats[2].metric("Prerequisites", len(module.prerequisites))
    st.download_button(
        "Download teach-this-tomorrow bundle",
        teaching_bundle(module),
        f"{module.identifier}_teach_this_tomorrow.zip",
        "application/zip",
        help="Local materials only: no student accounts, analytics, grading, or shared classroom state.",
    )
    st.markdown("### Live question")
    st.markdown(f"**{module.question}**")
    target_experiment = guided_experiment_for_module(module)
    if st.button(
        f"Start this lab — {target_experiment}",
        type="primary",
        key=f"launch_teaching_module_{module.identifier}",
        help="Open the matching prediction-led guided experiment. No calculation starts until you choose a prediction and run it.",
    ):
        st.session_state["teaching_experiment_request"] = target_experiment
        st.session_state["teaching_launch_notice"] = module.title
        st.session_state["hf_command_navigation"] = {
            "primary_mode": "Explore",
            "workspace": None,
        }
        st.rerun()
    tabs = st.tabs(
        [
            "Lecture mode",
            "Local lab section",
            "Student lab",
            "Instructor guide",
            "Notebook",
            "Scientific skepticism",
            "Teaching boundaries",
        ]
    )
    with tabs[0]:
        slides = lecture_slides(module)
        projector = st.toggle(
            "Projector contrast",
            value=True,
            help="Uses a restrained, high-contrast large-type layout. No science values are changed.",
            key=f"projector_{module.identifier}",
        )
        slide_number = st.select_slider(
            "Lecture slide — focus this control and use Left/Right Arrow keys to move through the narrative.",
            options=list(range(len(slides))),
            value=0,
            format_func=lambda index: f"{index + 1} / {len(slides)} · {slides[index].title}",
            key=f"lecture_slide_{module.identifier}",
        )
        slide = slides[slide_number]
        projector_class = " projector" if projector else ""
        st.markdown(
            f'<section class="lecture-slide{projector_class}"><span>{escape(slide.kicker)}</span><h2>{escape(slide.title)}</h2><p>{escape(slide.body)}</p><div><b>Ask the room</b>{escape(slide.prompt)}</div></section>',
            unsafe_allow_html=True,
        )
        with st.expander("Speaker note"):
            st.write(slide.speaker_note)
        st.caption(
            "Keyboard control is provided by the focused slide selector. This mode is local-only and keeps scientific caveats in the lecture arc."
        )
        st.download_button(
            "Download lecture outline",
            lecture_outline(module),
            f"{module.identifier}_lecture_outline.md",
            "text/markdown",
        )
    with tabs[1]:
        runs = load_all_runs()
        section_sources = {"HaloForge defaults": deepcopy(DEFAULT_PARAMS)}
        section_sources.update({run["name"]: run["params"] for run in runs})
        source_name = st.selectbox(
            "Locked baseline",
            list(section_sources),
            key=f"section_baseline_{module.identifier}",
        )
        defaults_by_module = {
            "primordial-tilt": ["n_s"],
            "numerical-coverage": ["k_max", "k_points"],
            "ede-structure": ["f_EDE", "log10_a_c"],
            "rare-tail-statistics": ["A_s", "n_s"],
            "numerical-methods": ["k_max", "k_points"],
        }
        permitted = st.multiselect(
            "Parameters students may vary",
            list(CONTROL_RANGES),
            default=defaults_by_module.get(module.identifier, ["n_s"]),
            key=f"section_parameters_{module.identifier}",
        )
        try:
            section = local_lab_section(module, section_sources[source_name], permitted)
        except ValueError as exc:
            st.warning(str(exc))
        else:
            st.caption(
                f"One local section · {module.duration_minutes} estimated minutes · four evidence-first prompts"
            )
            columns = st.columns(2)
            columns[0].download_button(
                "Download student lab section",
                student_section_bundle(section, module),
                f"{module.identifier}_student_section.zip",
                "application/zip",
            )
            columns[1].download_button(
                "Download instructor companion",
                instructor_section_bundle(section, module),
                f"{module.identifier}_instructor_companion.zip",
                "application/zip",
            )
            st.info(
                "Student and instructor materials are separate downloads. This is an operational separation only: HaloForge has no accounts or access control, so it cannot enforce hidden solutions, roles, response collection, analytics, or grade export."
            )
    with tabs[2]:
        st.markdown(student_handout(module))
        st.download_button(
            "Download student handout",
            student_handout(module),
            f"{module.identifier}_student_handout.md",
            "text/markdown",
        )
    with tabs[3]:
        st.markdown(instructor_guide(module))
        st.download_button(
            "Download instructor guide",
            instructor_guide(module),
            f"{module.identifier}_instructor_guide.md",
            "text/markdown",
        )
    with tabs[4]:
        st.caption(
            "Open this after exporting a run bundle; it loads the bundled Parquet tables locally."
        )
        st.download_button(
            "Download Jupyter notebook template",
            notebook_template(module),
            f"{module.identifier}.ipynb",
            "application/x-ipynb+json",
        )
    with tabs[5]:
        st.markdown("### A smooth curve can still be wrong")
        st.caption(
            "These local exercises do not record answers or assign grades. The goal is to practice narrowing a claim to the evidence actually available."
        )
        exercise_map = {exercise.identifier: exercise for exercise in EXERCISES}
        exercise_id = st.selectbox(
            "Scenario",
            list(exercise_map),
            format_func=lambda key: key.replace("-", " ").capitalize(),
            key="skepticism_exercise",
        )
        exercise = get_exercise(exercise_id)
        st.info(exercise.setup)
        answer = st.radio(
            exercise.prompt,
            list(range(len(exercise.choices))),
            format_func=lambda index: exercise.choices[index],
            key=f"skepticism_answer_{exercise_id}",
        )
        if st.button("Check reasoning", key=f"check_skepticism_{exercise_id}"):
            if answer == exercise.unjustified_choice:
                st.success(
                    "Correct: that conclusion goes beyond the available evidence."
                )
            else:
                st.warning(
                    "That statement may be supported, but it is not the overclaim in this scenario."
                )
            st.write(exercise.explanation)
            st.caption("Next evidence step: " + exercise.follow_up)
    with tabs[6]:
        st.info(
            "HaloForge currently offers local materials, reproducible exports, accessible transcripts, and reduced motion. Classroom roles, assignments, hidden solutions, aggregate responses, analytics, and grade exports require authenticated, privacy-reviewed collaboration infrastructure and are not implemented here."
        )


def learn_view():
    st.markdown(
        '<div class="page-head"><span>VISUAL COURSE</span><h2>Learn the pipeline by touching it.</h2><p>One connected journey from initial fluctuations to predicted halo counts.</p></div>',
        unsafe_allow_html=True,
    )
    ready = require_run()
    steps = [
        "1 · Seeds",
        "2 · Processing",
        "3 · Choose mass",
        "4 · Smooth",
        "5 · Variance",
        "6 · Collapse",
        "7 · Count halos",
    ]
    step = st.segmented_control("Pipeline step", steps, default=steps[0])
    index = steps.index(step)
    st.markdown(
        '<div class="pipeline-track">'
        + "".join(
            f'<div class="{"active" if i == index else "done" if i < index else ""}"><i>{i + 1}</i><span>{s.split(" · ")[1]}</span></div>'
            for i, s in enumerate(steps)
        )
        + "</div>",
        unsafe_allow_html=True,
    )
    if not ready:
        chart(primordial_fig(params), 430, "learn_pre")
        st.info(
            "The primordial controls can be inspected before a run. Complete an AxiCLASS run to continue through processed matter and halos."
        )
        return
    run, result, sigma = ready
    p = run["params"]
    left, right = st.columns([1.6, 1])
    visuals = [
        primordial_fig(p),
        power_fig(result, [0]),
        radius_fig(sigma, p),
        integrand_fig(result, sigma, p),
        sigma_fig(sigma, p, [p["single_z"]]),
        multiplicity_fig(p, ["Press-Schechter 1974", "Sheth-Tormen 2001"]),
        hmf_fig(run, [p["fitting"]], [p["single_z"]]),
    ]
    messages = [
        (
            "Primordial seeds",
            "Aₛ sets overall strength; nₛ decides how that strength is distributed across scale.",
            "Amplitude shifts vertically. Tilt pivots around kₚ.",
        ),
        (
            "The universe processes the seeds",
            "Radiation, matter, baryons, expansion, and EDE reshape the initial spectrum into P(k,z).",
            "Large k means smaller comoving structure.",
        ),
        (
            "Mass becomes a radius",
            "At the mean cosmic density, each target mass corresponds to a smoothing radius R.",
            "A cluster samples a much larger region than a dwarf halo.",
        ),
        (
            "The window selects modes",
            "W(kR) weights which Fourier scales contribute to the chosen mass.",
            "The displayed integrand is the contribution to σ² per ln k.",
        ),
        (
            "Variance measures fluctuation strength",
            "σ(M,z) is the RMS smoothed density contrast and usually falls toward high mass.",
            "Massive regions are rarer because larger volumes average fluctuations down.",
        ),
        (
            "Collapse maps peaks to halos",
            "f(σ) encodes a collapse prescription or simulation-calibrated multiplicity relation.",
            "Fits differ because halo definitions and calibrations differ.",
        ),
        (
            "The HMF predicts abundance",
            "dn/dlnM combines density, multiplicity, and the σ slope.",
            "The high-mass tail is exponentially sensitive to small cosmological changes.",
        ),
    ]
    with left:
        chart(visuals[index], 530, key=f"learn_{index}")
    with right:
        h, body, take = messages[index]
        st.markdown(
            f'<div class="lesson-card"><span>STEP {index + 1} OF 7</span><h3>{h}</h3><p>{body}</p><b>{take}</b></div>',
            unsafe_allow_html=True,
        )
        if index == 0:
            st.latex(r"\mathcal P_\mathcal R(k)=A_s(k/k_*)^{n_s-1}")
        elif index == 2:
            st.latex(r"R(M)=\left(3M/4\pi\rho_{m,0}\right)^{1/3}")
        elif index == 3:
            st.latex(r"W_{TH}(y)=3(\sin y-y\cos y)/y^3")
        elif index == 4:
            st.latex(r"\sigma^2(R,z)=\frac1{2\pi^2}\int k^2P(k,z)W^2(kR)dk")
        elif index == 6:
            st.latex(
                r"\frac{dn}{d\ln M}=\frac{\rho_{m,0}}M f(\sigma)\left|\frac{d\ln\sigma}{d\ln M}\right|"
            )


def atlas_view():
    st.markdown(
        '<div class="page-head"><span>FIT + WINDOW ATLAS</span><h2>Expose every modeling choice.</h2></div>',
        unsafe_allow_html=True,
    )
    active = get_active_params()
    view = st.segmented_control(
        "Atlas view",
        [
            "Multiplicity fits",
            "Window functions",
            "Window impact on σ",
            "Calibration table",
        ],
        default="Multiplicity fits",
    )
    if view == "Multiplicity fits":
        chosen = st.multiselect(
            "Visible fits",
            FITTING_NAMES,
            default=[
                "Press-Schechter 1974",
                "Sheth-Tormen 2001",
                "Tinker 2008",
                "Watson FOF 2013",
            ],
            max_selections=8,
        )
        chart(multiplicity_fig(active, chosen), 580, "atlas_fit", axis_controls=True)
    elif view == "Window functions":
        a, b = st.columns(2)
        with a:
            chart(windows_fig(False), 450, "atlas_w")
        with b:
            chart(windows_fig(True), 450, "atlas_w2")
        chart(taylor_fig(), 400, "atlas_taylor")
    elif view == "Window impact on σ":
        ready = require_run()
        if ready:
            run, result, sigma = ready
            p = run["params"]
            f = go.Figure()
            for i, w in enumerate(WINDOWS):
                g = sigma_grid(
                    sigma["M"],
                    result["k"],
                    result["P"],
                    {"h": result["derived"]["h"], "Omega_m": p["Omega_m"]},
                    w,
                    int(p.get("quad_limit", 200)),
                )
                f.add_trace(
                    go.Scatter(
                        x=sigma["M_h"],
                        y=g["sigma"],
                        name=w,
                        line=dict(color=COLORS[i], width=2.7),
                    )
                )
            f.update_xaxes(type="log", title="M [h⁻¹ M☉]")
            f.update_yaxes(type="log", title="σ(M)")
            set_plot(
                f,
                "Same P(k), different smoothing window",
                "This isolates the modeling choice W(kR) while holding the completed matter spectrum fixed.",
            )
            chart(f, 540, "atlas_sigma", axis_controls=True)
    else:
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "fit": n,
                        "published range": FIT_METADATA[n][0],
                        "redshift": FIT_METADATA[n][1],
                        "calibration": FIT_METADATA[n][2],
                    }
                    for n in FITTING_NAMES
                ]
            ),
            width="stretch",
            hide_index=True,
        )
        st.warning(
            "A fit is publication-ready only when its halo definition, overdensity, cosmology, redshift, and calibrated range match your analysis."
        )


def workspace_zip():
    out = BytesIO()
    with ZipFile(out, "w", ZIP_DEFLATED) as zf:
        checksums = {}
        for folder in (
            run_storage.RUN_DIR,
            run_storage.EXPORT_DIR,
            run_storage.STATE_DIR,
        ):
            if folder.exists():
                for path in folder.rglob("*"):
                    if path.is_file() and path.name != ".gitkeep":
                        name = path.relative_to(run_storage.DATA_ROOT).as_posix()
                        content = path.read_bytes()
                        zf.writestr(name, content)
                        checksums[name] = sha256_bytes(content)
        zf.writestr(
            WORKSPACE_MANIFEST,
            json.dumps(
                {"schema_version": "haloforge-workspace-v1", "files": checksums},
                indent=2,
                sort_keys=True,
            ),
        )
    return out.getvalue()


def restore_workspace(payload: bytes, overwrite: bool = False) -> int:
    """Import a preflighted local workspace without implicit replacement."""
    return len(
        import_workspace(payload, run_storage.DATA_ROOT, overwrite=overwrite).files
    )


def runs_view():
    st.markdown(
        '<div class="page-head"><span>RUN VAULT</span>'
        "<h2>Every completed universe is already saved.</h2>"
        f"<p>{privacy_statement()} Runs are written atomically, so refreshes, browser closes, "
        "app restarts, and container rebuilds do not erase them.</p></div>",
        unsafe_allow_html=True,
    )

    st.code(
        str(run_storage.DATA_ROOT),
        language="text",
    )
    deletion_id = st.session_state.get("last_deleted_run_id")
    if deletion_id:
        st.info("The most recently deleted run is in this device’s local trash.")
        if st.button("Restore most recently deleted run"):
            try:
                restored_id = restore_deleted_run(deletion_id)
                st.session_state.pop("last_deleted_run_id", None)
                load_run_into_session(restored_id)
                clear_widgets()
                st.success("Deleted run restored with its saved arrays and exports.")
                st.rerun()
            except ValueError as exc:
                show_failure(exc)

    upload = st.file_uploader(
        "Restore workspace",
        type=["zip"],
    )

    import_plan = None
    if upload:
        try:
            import_plan = plan_workspace_import(
                upload.getvalue(), run_storage.DATA_ROOT
            )
            integrity_label = (
                "verified integrity"
                if import_plan.integrity == "verified"
                else "legacy archive — no integrity manifest"
            )
            st.caption(
                f"Archive preflight: {len(import_plan.files)} files · {import_plan.total_bytes / 1_000_000:.1f} MB · {integrity_label}"
            )
            if import_plan.collisions:
                st.warning(
                    f"{len(import_plan.collisions)} existing local file(s) would be replaced. Review before continuing."
                )
        except BundleValidationError as exc:
            show_failure(exc)
    replace_conflicts = bool(
        import_plan
        and import_plan.collisions
        and st.checkbox(
            "I understand this will replace the conflicting local files",
            key="restore_replace_confirm",
        )
    )

    if (
        upload
        and import_plan
        and st.button(
            "Restore now",
            disabled=bool(import_plan.collisions and not replace_conflicts),
        )
    ):
        try:
            count = restore_workspace(upload.getvalue(), overwrite=replace_conflicts)

            restored = load_all_runs()

            if restored:
                load_run_into_session(restored[-1]["run_id"])
                clear_widgets()

            st.success(f"Restored {count} files")

            st.rerun()

        except Exception as exc:
            show_failure(exc)

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

    run_map = {str(run["run_id"]): run for run in runs}

    run_ids = list(run_map)

    preferred_run_id = next(
        (run_id for run_id in run_ids if run_map[run_id].get("is_baseline")),
        run_ids[-1],
    )

    _prepare_scalar_widget_state(
        "run_vault_selected_id",
        run_ids,
    )

    chosen_kwargs = {}

    if "run_vault_selected_id" not in st.session_state:
        chosen_kwargs["index"] = run_ids.index(preferred_run_id)

    chosen_id = st.selectbox(
        "Selected run",
        run_ids,
        format_func=lambda run_id: get_run_label(run_map[run_id]),
        key="run_vault_selected_id",
        **chosen_kwargs,
    )

    chosen = run_map[chosen_id]
    is_only_baseline = bool(chosen.get("is_baseline")) and len(runs) == 1
    if is_only_baseline:
        st.warning(
            "This is the only saved baseline. Create or select another baseline before deleting it so comparisons retain an explicit reference."
        )
    delete_confirmed = st.checkbox(
        f"I confirm that I want to move “{chosen['name']}” to local trash.",
        key=f"delete_confirm_{chosen_id}",
        disabled=is_only_baseline,
    )
    if chosen.get("integrity_status", {}).get("state") == "invalid":
        st.error(
            "This run failed integrity verification. Restore the original files or calculate a new run. "
            "The workspace download above preserves the damaged record for inspection."
        )
        st.json(chosen["integrity_status"])
        if st.button(
            "Delete selected run",
            type="secondary",
            disabled=not delete_confirmed or is_only_baseline,
        ):
            st.session_state["last_deleted_run_id"] = delete_run(chosen_id)
            st.session_state.pop("run_vault_selected_id", None)
            st.rerun()
        return
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
                    f"P_z{z:g}_Mpc3": (arrays["P_by_z"][i])
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
                    f"sigma_z{z:g}": (arrays["sigma_by_z"][i])
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
                    "baseline": run.get("is_baseline"),
                    "backend": run.get("class_status"),
                    "created": run.get("created_at"),
                    "EDE": run["params"].get("enable_ede"),
                    "fEDE": run["params"].get("f_EDE"),
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
        disabled=not delete_confirmed or is_only_baseline,
    ):
        deleted_was_loaded = st.session_state.get("loaded_run_id") == chosen_id

        deleted_was_baseline = bool(chosen.get("is_baseline"))

        st.session_state["last_deleted_run_id"] = delete_run(chosen_id)

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
            and not any(run.get("is_baseline") for run in remaining)
        ):
            set_baseline(remaining[0]["run_id"])
            remaining = load_all_runs()

        if deleted_was_loaded:
            if remaining:
                load_run_into_session(remaining[-1]["run_id"])
                clear_widgets()
            else:
                clear_loaded_run()

        st.rerun()


def diagnostics_view():
    st.markdown(
        '<div class="page-head"><span>VALIDATION</span><h2>Inspect every assumption.</h2></div>',
        unsafe_allow_html=True,
    )
    diag = environment_diagnostics()
    ok = diag["classy_imports"]
    st.markdown(
        f'<div class="diagnostic {"pass" if ok else "fail"}"><i></i><b>{"classy is importable" if ok else "classy unavailable"}</b><span>{diag.get("classy_path") or diag.get("classy_error")}</span></div>',
        unsafe_allow_html=True,
    )
    st.info(
        "Every CLASS solve runs in an isolated child process. A native AxiCLASS failure can end that worker, but it cannot take down the app or overwrite the last completed run. "
        f"Policy: {diag['class_timeout_seconds']} s maximum; up to {diag['class_transient_retry_limit']} retry for a worker crash or unreadable artifact; no retry for a timeout or CLASS-reported input error."
    )
    if st.button("Run real CLASS smoke test", type="primary"):
        with st.spinner("Running minimal ΛCDM solve in an isolated worker…"):
            smoke = tiny_class_smoke_test(get_params())
        (st.success if smoke["ok"] else st.error)(smoke["message"])
    ready = require_run()
    if ready:
        run, result, sigma = ready
        rows = []
        for i, z in enumerate(result["redshifts"]):
            d = float(result["growth_class"][i])
            p_growth = float(
                np.sqrt(
                    np.interp(0.01, result["k"], result["P_by_z"][i])
                    / np.interp(0.01, result["k"], result["P_by_z"][0])
                )
            )
            rows.append(
                {
                    "z": z,
                    "σ8 pipeline": sigma["sigma8_pipeline_by_z"][i],
                    "D CLASS": d,
                    "D from P(k=.01)": p_growth,
                    "fractional difference": p_growth / d - 1,
                }
            )
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        chart(growth_fig(result, sigma), 440, "diag_growth")
        numerical = sigma.get("numerical_diagnostics")
        if numerical:
            st.subheader("Numerical range and truncation sensitivity")
            st.caption(numerical["method"] + " " + numerical["scope_limit"])
            rows = []
            for label, payload in (
                ("Sampled-range coverage", numerical["coverage"]),
                ("Remove high-k endpoint", numerical["high_k_truncation"]),
                ("Remove low-k endpoint", numerical["low_k_truncation"]),
            ):
                rows.append(
                    {
                        "check": label,
                        "status": payload.get("status"),
                        "maximum fractional σ change": payload.get(
                            "maximum_fractional_sigma_change"
                        ),
                        "mass at maximum [h⁻¹ M☉]": payload.get("mass_at_maximum_Msun"),
                        "notes": "; ".join(payload.get("notes", []))
                        or payload.get("reason", ""),
                    }
                )
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    a, b = st.columns(2)
    with a:
        st.subheader("Runtime summary")
        runtime_rows = [
            {"item": "CLASS worker", "value": diag["worker_executable"]},
            {"item": "AxiCLASS available", "value": str(diag["classy_imports"])},
            {"item": "Isolation", "value": diag["class_isolation"]},
            {"item": "Timeout", "value": f"{diag['class_timeout_seconds']} seconds"},
            {"item": "Retry limit", "value": str(diag["class_transient_retry_limit"])},
            {"item": "Platform", "value": diag["platform"]},
            {"item": "NumPy", "value": diag["numpy"]},
        ]
        st.dataframe(pd.DataFrame(runtime_rows), hide_index=True, width="stretch")
        with st.expander("Full runtime diagnostic record"):
            st.json(diag)
            st.download_button(
                "Download runtime diagnostic JSON",
                data=json.dumps(diag, indent=2),
                file_name="haloforge_runtime_diagnostics.json",
                mime="application/json",
                key="download_runtime_diagnostics",
            )
    with b:
        st.subheader("Next CLASS calculation")
        st.caption("The settings below will be sent to the next solver run.")
        with st.expander("Inspect exact CLASS/AxiCLASS settings"):
            st.code(
                json.dumps(build_class_settings(get_params()), indent=2), language="json"
            )


def evolution_studio_view():
    st.markdown(
        '<div class="page-head"><span>EVOLUTION STUDIO</span>'
        "<h2>Redshift Evolution Movie: z = 20 → 0</h2>"
        "<p>Scientifically sampled continuous cosmic evolution across scale factor a and cosmic time. "
        "Fixed axes maintain true physical growth dynamics without misleading autoscale breathing.</p>"
        "</div>",
        unsafe_allow_html=True,
    )
    ready = require_run()
    if not ready:
        return
    run, result, sigma = ready
    # The movie must remain bound to the saved calculation, not a newer staged
    # draft that happens to be visible in the sidebar.
    params = dict(run["params"])

    c1, c2, c3, c4 = st.columns([1.5, 1, 1, 1])
    observable = c1.selectbox(
        "Observable",
        [
            "Matter Power P(k)",
            "Dimensionless Power Δ²(k)",
            "Mass Variance σ(M)",
            "Differential HMF dn/dlnM",
            "Cumulative HMF n(>M)",
        ],
        key="evo_observable",
    )
    sampling_rule = c2.selectbox(
        "Sampling rule",
        ["uniform_a", "uniform_z", "uniform_log_a"],
        format_func=lambda s: {
            "uniform_a": "Uniform in a (Physically spaced)",
            "uniform_z": "Uniform in z (Redshift grid)",
            "uniform_log_a": "Uniform in ln(a) (Log expansion)",
        }.get(s, s),
        key="evo_sampling",
    )
    num_frames = c3.slider(
        "Frame count", min_value=6, max_value=24, value=12, key="evo_frames"
    )
    playback_speed = c4.selectbox(
        "Playback speed",
        ["0.25x (Teaching)", "0.5x", "1.0x", "2.0x"],
        index=2,
        key="evo_speed",
    )

    zs = evolution_redshifts(20.0, 0.0, num_frames, sampling_rule)
    source_redshifts = np.asarray(result.get("redshifts", [0.0]), dtype=float)
    source_power = np.asarray(result.get("P_by_z", [result["P"]]), dtype=float)
    source_cosmic_time = np.asarray(
        result.get("background_cosmic_time_gyr_by_z", []), dtype=float
    )
    z0_source_index = redshift_index(source_redshifts, 0.0)
    if source_redshifts.size < 2:
        st.warning(
            "This saved run has only one calculated redshift. Evolution frames would require an approximation, so add a redshift grid and rerun CLASS/AxiCLASS."
        )
        return
    cosmic_time_by_frame = None
    if source_cosmic_time.shape == source_redshifts.shape:
        source_log_a = np.log(1.0 / (1.0 + source_redshifts))[::-1]
        cosmic_time_by_frame = np.interp(
            np.log(1.0 / (1.0 + zs)), source_log_a, source_cosmic_time[::-1]
        )
    elif params.get("enable_ede"):
        st.warning(
            "This historical EDE run has no stored solver background proper time. "
            "Cosmic age is shown as unavailable rather than approximated; rerun to record it."
        )
    try:
        frames = calculate_evolution_frames(
            result["k"],
            source_power[z0_source_index],
            zs,
            None,
            params,
            power_by_z=source_power,
            source_redshifts=source_redshifts,
            cosmic_time_gyr_by_z=cosmic_time_by_frame,
        )
    except ValueError as exc:
        st.warning(
            f"Evolution cannot be generated from this run: {exc} Add the requested redshift range to the CLASS/AxiCLASS run and calculate again."
        )
        return

    evolution_status = weakest_status(
        *(str(frame["scientific_status"]) for frame in frames)
    )
    if evolution_status != CALCULATED_LINEAR_THEORY:
        validated = [
            frame["interpolation_validation_median_fractional_error"]
            for frame in frames
            if frame["interpolation_validation_median_fractional_error"] is not None
        ]
        error_text = (
            f" Withheld-spectrum interpolation checks have median fractional error "
            f"{float(np.median(validated)):.2%}."
            if validated
            else " No withheld interior solver spectra are available to quantify interpolation error."
        )
        st.warning(
            "Some movie frames are log P–log a interpolations between stored CLASS/AxiCLASS spectra, "
            "so this export is labelled Approximation rather than calculated linear theory."
            + error_text
        )

    t1, t2, t3, t4 = st.tabs(
        [
            "Interactive Movie Scrubber",
            "Static Contact Sheet",
            "Frame Data",
            "Export Data",
        ]
    )

    theme = st.session_state.get("accessibility_theme", "Dark")

    with t1:
        frame_idx = (
            st.slider(
                "Cosmic Timeline Scrubber",
                min_value=1,
                max_value=len(frames),
                value=len(frames),
                format=f"Frame %d of {len(frames)}",
                key="evo_scrubber",
            )
            - 1
        )

        active_frame = frames[frame_idx]
        fig = build_evolution_figure(
            frames, frame_idx, observable, fixed_axes=True, theme=theme
        )
        chart(fig, 500, f"evo_chart_{frame_idx}", axis_controls=True)

        scol1, scol2, scol3 = st.columns(3)
        scol1.metric("Redshift z", f"{active_frame['redshift']:.2f}")
        scol2.metric("Scale Factor a", f"{active_frame['scale_factor']:.4f}")
        cosmic_time = active_frame.get("cosmic_time_gyr")
        scol3.metric(
            "Cosmic Time",
            f"{float(cosmic_time):.2f} Gyr"
            if cosmic_time is not None
            else "Unavailable",
        )
        st.caption("Cosmic time source: " + active_frame["cosmic_time_source"])
        if active_frame.get("milestones"):
            st.info(
                "Milestones: "
                + " · ".join(m["name"] for m in active_frame["milestones"])
            )

    with t2:
        st.caption(
            "Synchronized small multiples at key cosmic epochs for publication and static review."
        )
        contact_fig = evolution_contact_sheet(
            frames, observable, num_panels=4, theme=theme
        )
        chart(contact_fig, 580, "evo_contact_sheet", axis_controls=False)

    with t3:
        st.caption(
            "Exact per-frame redshift, scale factor, cosmic time, milestones, and spectrum-source records."
        )
        summary_rows = evolution_frame_summary_table(frames)
        st.dataframe(pd.DataFrame(summary_rows), hide_index=True, width="stretch")

    with t4:
        st.caption(
            "Frames are rendered with fixed axes from this exact sequence. The manifest documents the scientific source and video settings."
        )
        manifest_json = json.dumps(
            {
                "cosmology": run.get("name", "Active Run"),
                "scientific_status": evolution_status,
                "observable": observable,
                "frame_count": len(frames),
                "sampling_rule": sampling_rule,
                "frames": [
                    {
                        "index": f["frame_index"],
                        "z": f["redshift"],
                        "a": f["scale_factor"],
                        "time_gyr": f["cosmic_time_gyr"],
                        "cosmic_time_source": f["cosmic_time_source"],
                        "milestones": [m["name"] for m in f.get("milestones", [])],
                        "power_source": f["power_source"],
                        "scientific_status": f["scientific_status"],
                        "interpolation_validation_median_fractional_error": f[
                            "interpolation_validation_median_fractional_error"
                        ],
                        "interpolation_validation_max_fractional_error": f[
                            "interpolation_validation_max_fractional_error"
                        ],
                    }
                    for f in frames
                ],
            },
            indent=2,
        )
        st.download_button(
            "Download Evolution Manifest (JSON)",
            data=manifest_json,
            file_name="evolution_manifest.json",
            mime="application/json",
        )
        vcol1, vcol2, vcol3 = st.columns(3)
        video_format = vcol1.selectbox(
            "Video format", list(VIDEO_FORMATS), format_func=str.upper, key="evo_video_format"
        )
        video_fps = vcol2.number_input(
            "Frames per second", min_value=0.1, max_value=30.0,
            value={"0.25x (Teaching)": 0.5, "0.5x": 1.0, "1.0x": 2.0, "2.0x": 4.0}[playback_speed],
            step=0.5, key="evo_video_fps",
        )
        video_size = vcol3.selectbox(
            "Export size", [(960, 540), (1280, 720), (1920, 1080)],
            format_func=lambda s: f"{s[0]} × {s[1]}", key="evo_video_size",
        )
        if st.button("Render evolution video", type="primary", key="evo_render_video"):
            with st.spinner("Rendering deterministic evolution frames…"):
                try:
                    st.session_state["evo_video_bytes"] = render_evolution_video(
                        frames,
                        observable,
                        video_format=video_format,
                        fps=float(video_fps),
                        width=int(video_size[0]),
                        height=int(video_size[1]),
                        theme=theme,
                    )
                    st.session_state["evo_video_spec"] = {
                        "format": video_format,
                        "fps": float(video_fps),
                        "width": int(video_size[0]),
                        "height": int(video_size[1]),
                    }
                except (ValueError, RuntimeError) as exc:
                    st.error(f"Video rendering failed: {exc}")
        video_bytes = st.session_state.get("evo_video_bytes")
        video_spec = st.session_state.get("evo_video_spec", {})
        if video_bytes and video_spec.get("format") == video_format:
            st.download_button(
                f"Download {video_format.upper()} evolution video",
                data=video_bytes,
                file_name=f"haloforge_evolution.{video_format}",
                mime=VIDEO_FORMATS[video_format],
                key="evo_download_video",
            )


def campaign_lab_view():
    st.markdown(
        '<div class="page-head"><span>PARAMETER CAMPAIGN LAB</span>'
        "<h2>Multi-Cosmology Explorations</h2>"
        "<p>Run reproducible CLASS/AxiCLASS parameter campaigns with a declared worker limit, "
        "resource pre-flight, and saved per-member solver evidence. Campaign results are linear-theory calculations, not N-body results.</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns([1.5, 1, 1])
    campaign_name = c1.text_input(
        "Campaign name", value="Cosmology Exploration Alpha", key="camp_name"
    )
    strategy = c2.selectbox(
        "Design strategy",
        ["latin_hypercube", "sobol", "cartesian"],
        format_func=lambda s: {
            "latin_hypercube": "Latin Hypercube Sampling (LHS)",
            "sobol": "Sobol Quasi-Random Sequence",
            "cartesian": "Cartesian Product Grid",
        }.get(s, s),
        key="camp_strategy",
    )
    total_samples = c3.number_input(
        "Target runs", min_value=3, max_value=1000, value=8, step=1, key="camp_samples"
    )

    st.subheader("Select parameters to vary")
    available_params = ["n_s", "f_EDE", "H0", "Omega_m"]
    selected_params = st.multiselect(
        "Varying parameters",
        available_params,
        default=["n_s", "f_EDE"],
        key="camp_selected_params",
    )
    if not selected_params:
        st.warning("Select at least one parameter to sweep.")
        return

    param_bounds = {}
    grid_increments = {}
    param_cols = st.columns(len(selected_params))
    default_ranges = {
        "n_s": (0.92, 1.02),
        "f_EDE": (0.02, 0.16),
        "H0": (64.0, 74.0),
        "Omega_m": (0.28, 0.35),
    }
    for i, p_name in enumerate(selected_params):
        with param_cols[i]:
            dr = default_ranges.get(p_name, (0.9, 1.1))
            val_min = st.number_input(
                f"{p_name} min", value=dr[0], key=f"camp_{p_name}_min"
            )
            val_max = st.number_input(
                f"{p_name} max", value=dr[1], key=f"camp_{p_name}_max"
            )
            grid_increments[p_name] = float(
                st.number_input(
                    f"{p_name} grid increment",
                    min_value=1e-12,
                    value=0.05 if p_name == "H0" else (0.01 if p_name != "f_EDE" else 0.01),
                    format="%.8g",
                    help="Used only by Cartesian grids. Both endpoints are included when the increment lands on the maximum.",
                    key=f"camp_{p_name}_increment",
                )
            )
            if float(val_max) <= float(val_min):
                st.error(f"{p_name} maximum must be greater than its minimum.")
                return
            param_bounds[p_name] = (float(val_min), float(val_max))

    if strategy == "latin_hypercube":
        combos = latin_hypercube_campaign(param_bounds, int(total_samples))
    elif strategy == "sobol":
        combos = sobol_campaign(param_bounds, int(total_samples))
    else:  # cartesian
        grid_dict = {}
        for key, (lo, hi) in param_bounds.items():
            increment = grid_increments[key]
            values = np.arange(lo, hi + increment * 0.5, increment, dtype=float)
            if values.size == 0 or values[-1] > hi + 1e-10:
                raise ValueError(f"Invalid Cartesian increment for {key}.")
            if not np.isclose(values[-1], hi, rtol=0, atol=max(1e-10, increment * 1e-8)):
                values = np.append(values, hi)
            grid_dict[key] = [float(v) for v in values]
        combos = cartesian_campaign(grid_dict, max_runs=1000)

    max_workers = max(1, (os.cpu_count() or 2) - 1)
    worker_count = st.slider(
        "Concurrent CLASS workers",
        1,
        max(max_workers, 1),
        min(2, max_workers),
        help="The detected machine has one additional logical CPU reserved for responsiveness. Each worker launches an isolated CLASS/AxiCLASS process.",
        key="camp_workers",
    )
    timing_samples = completed_member_timings()
    measured_seconds_per_member = (
        float(np.median(timing_samples)) if timing_samples else None
    )
    res_est = estimate_campaign_resources(
        len(combos),
        worker_count=int(worker_count),
        seconds_per_run=measured_seconds_per_member,
    )

    est_col1, est_col2, est_col3, est_col4 = st.columns(4)
    est_col1.metric("Total Jobs", f"{res_est.total_runs}")
    est_col2.metric(
        "Estimated Compute Time",
        f"{res_est.estimated_cpu_seconds:.1f} s"
        if res_est.estimated_cpu_seconds is not None
        else "Unmeasured",
    )
    est_col3.metric("Est. Memory", f"{res_est.estimated_memory_mb:.1f} MB")
    est_col4.metric("Est. Disk", f"{res_est.estimated_disk_mb:.1f} MB")
    st.caption(res_est.timing_basis)

    if res_est.cartesian_warning:
        st.warning(res_est.cartesian_warning)

    if len(combos) > 64:
        st.warning(
            f"This will execute {len(combos)} real CLASS/AxiCLASS calculations. Review the exact grid and estimated resources before starting."
        )
    confirmed_large_campaign = st.checkbox(
        "I reviewed this campaign's exact grid and resource estimate.",
        value=len(combos) <= 64,
        key="camp_confirm_resources",
    )

    stored_campaign_ids = list_campaign_ids()
    if stored_campaign_ids:
        load_col, _ = st.columns([1, 1])
        campaign_to_load = load_col.selectbox(
            "Resume saved campaign",
            stored_campaign_ids,
            key="camp_load_id",
        )
        if load_col.button("Load campaign", key="camp_load_btn"):
            try:
                st.session_state["active_campaign"] = load_campaign(campaign_to_load)
                st.success(f"Loaded campaign {campaign_to_load}.")
            except ValueError as exc:
                st.error(f"Campaign could not be loaded: {exc}")

    if st.button(
        "Create real CLASS/AxiCLASS campaign queue",
        type="primary",
        key="camp_run_btn",
        disabled=not confirmed_large_campaign,
    ):
        base_params = deepcopy(get_params())
        camp = create_campaign(
            campaign_name,
            strategy,
            combos,
            metadata={
                "scientific_status": CALCULATED_LINEAR_THEORY,
                "backend": "CLASS/AxiCLASS",
                "worker_count": int(worker_count),
                "base_parameters": base_params,
                "parameter_grid": grid_dict if strategy == "cartesian" else param_bounds,
            },
        )
        campaign_path = save_campaign(camp)
        st.session_state["active_campaign"] = camp
        st.success(f"Campaign queue created and saved: {campaign_path}")

    camp = st.session_state.get("active_campaign")
    if camp:
        counts = campaign_counts(camp)
        st.subheader("Campaign Queue")
        qcol1, qcol2, qcol3, qcol4, qcol5 = st.columns(5)
        qcol1.metric("Pending", counts["pending"])
        qcol2.metric("Completed", counts["completed"])
        qcol3.metric("Failed", counts["failed"])
        qcol4.metric("Cancelled", counts["cancelled"])
        qcol5.metric("State", str(camp.metadata.get("lifecycle_state", "pending")).title())
        if camp.metadata.get("recovery_note"):
            st.warning(str(camp.metadata["recovery_note"]))

        batch_size = st.number_input(
            "Members to execute in next batch",
            min_value=1,
            max_value=max(1, counts["pending"]),
            value=min(max(1, int(camp.metadata.get("worker_count", worker_count)) * 4), max(1, counts["pending"])),
            step=1,
            key=f"camp_batch_size_{camp.campaign_id}",
        )
        controls = st.columns(4)
        if controls[0].button(
            "Run next real-solver batch",
            type="primary",
            disabled=counts["pending"] == 0 or camp.metadata.get("lifecycle_state") == "paused",
            key=f"camp_batch_run_{camp.campaign_id}",
        ):
            base_params = dict(camp.metadata["base_parameters"])

            def eval_member(delta: dict) -> dict:
                candidate = {**base_params, **delta}
                if "f_EDE" in delta:
                    candidate["enable_ede"] = float(candidate["f_EDE"]) > 0.0
                result = compute_matter_power(candidate)
                derived = result["derived"]
                return {
                    "sigma8": float(derived["sigma8"]),
                    "Omega_m": float(derived["Omega_m"]),
                    "h": float(derived["h"]),
                    "class_status": result["class_status"],
                    "class_settings": result["class_settings"],
                    "solver_execution": result.get("solver_execution", {}),
                    "classy_path": result.get("classy_path", ""),
                }

            with st.spinner("Running isolated CLASS/AxiCLASS workers and checkpointing each result…"):
                execute_campaign_step(
                    camp,
                    eval_member,
                    max_steps=int(batch_size),
                    worker_count=int(camp.metadata["worker_count"]),
                    on_member_complete=save_campaign,
                )
            st.success("Batch finished; completed and failed members were checkpointed locally.")
        if controls[1].button("Pause queue", disabled=counts["pending"] == 0, key=f"camp_pause_{camp.campaign_id}"):
            pause_campaign(camp)
            save_campaign(camp)
            st.rerun()
        if controls[2].button("Resume queue", disabled=camp.metadata.get("lifecycle_state") != "paused", key=f"camp_resume_{camp.campaign_id}"):
            resume_campaign(camp)
            save_campaign(camp)
            st.rerun()
        if controls[3].button("Retry failed", disabled=counts["failed"] == 0, key=f"camp_retry_{camp.campaign_id}"):
            retry_failed_members(camp)
            save_campaign(camp)
            st.rerun()
        if st.button("Cancel unstarted members", disabled=counts["pending"] == 0, key=f"camp_cancel_{camp.campaign_id}"):
            cancelled = cancel_pending_members(camp)
            save_campaign(camp)
            st.info(f"Cancelled {cancelled} unstarted member(s); completed evidence was retained.")

        df = campaign_to_dataframe(camp)
        st.subheader("Campaign Results & Multi-Dimensional Analysis")
        campaign_status = camp.metadata.get("scientific_status", CALCULATED_LINEAR_THEORY)
        st.caption(
            f"Evidence status: {STATUS_LABELS.get(campaign_status, campaign_status)}. "
            "These are stored linear-theory solver calculations, not nonlinear simulation measurements."
        )
        st.download_button(
            "Download campaign reproducibility bundle (ZIP)",
            data=campaign_export_bundle(camp),
            file_name=f"{camp.campaign_id}_bundle.zip",
            mime="application/zip",
            key=f"campaign_bundle_{camp.campaign_id}",
        )
        theme = st.session_state.get("accessibility_theme", "Dark")
        par_fig = campaign_parallel_coordinates(
            df, selected_params, color_metric="sigma8", theme=theme
        )
        chart(par_fig, 460, "camp_par_coords", axis_controls=False)

        if len(selected_params) >= 1:
            prime_param = selected_params[0]
            resp_fig = campaign_response_figure(
                df, prime_param, metric_y="sigma8", theme=theme
            )
            chart(resp_fig, 440, f"camp_resp_{prime_param}", axis_controls=True)

        st.dataframe(df, hide_index=True, width="stretch")

        sensitivities = compute_campaign_sensitivities(df, target="sigma8")
        if sensitivities:
            st.subheader("Parameter Sensitivity Analysis")
            st.dataframe(pd.DataFrame(sensitivities), hide_index=True, width="stretch")


def simulation_lab_view():
    st.markdown(
        '<div class="page-head"><span>SIMULATION PREPARATION LAB</span>'
        "<h2>GADGET-4 Lab</h2>"
        "<p>Plan a collisionless box, inspect recorded snapshots, and compare matched FoF catalogues with the linear HMF. "
        "Execution and EDE dynamics are unavailable until externally validated.</p>"
        "</div>",
        unsafe_allow_html=True,
    )
    params = get_params()

    t_doc, t_box, t_ic, t_conf, t_halo, t_hmf = st.tabs(
        [
            "1 · Doctor",
            "2 · Box",
            "3 · ICs",
            "4 · Runtime",
            "5 · Halos",
            "6 · HMF Compare",
        ]
    )

    with t_doc:
        st.subheader("Host Execution Diagnostics")
        report = run_installation_doctor()
        dcol1, dcol2, dcol3, dcol4 = st.columns(4)
        dcol1.metric("CPU Cores", f"{report.cpu_cores}")
        dcol2.metric("RAM Available", f"{report.ram_gb:.1f} GB")
        dcol3.metric("Free Disk", f"{report.free_disk_gb:.1f} GB")
        dcol4.metric(
            "Acceptance Evidence",
            "Recorded" if report.acceptance_evidence_present else "Missing",
        )
        st.caption(report.recommended_runtime)

        st.markdown(
            f"**GADGET-4 Reference:** {GADGET4_VERSION} (Commit `{GADGET4_PINNED_COMMIT[:8]}`) · License: GPL-3.0"
        )
        st.caption(f"Citation: {GADGET4_CITATION}")
        if not report.ready_for_simulation:
            st.warning(
                "Simulation execution is intentionally gated. A detected Docker/MPI installation is not proof of a reproducible GADGET-4 build or validated EDE dynamics."
            )
        if report.recommendations:
            for rec in report.recommendations:
                st.info(rec)

    with t_box:
        st.subheader("Periodic Box Geometry & Mass Resolution")
        bcol1, bcol2, bcol3 = st.columns(3)
        box_size = bcol1.number_input(
            "Box length L [h⁻¹ Mpc]",
            min_value=10.0,
            max_value=2000.0,
            value=100.0,
            step=10.0,
            key="sim_box_size",
        )
        particles_per_dim = bcol2.selectbox(
            "Particles per dimension N", [32, 64, 128, 256], index=1, key="sim_part_dim"
        )
        bcol3.metric("Physics mode", "Collisionless DM-only")
        bcol3.caption(
            "Gas/SPH and baryonic-feedback workflows are not configured or validated here."
        )

        res = compute_box_resolution(
            box_size, particles_per_dim, params, is_hydro=False
        )

        rcol1, rcol2, rcol3, rcol4 = st.columns(4)
        rcol1.metric("Total Particles N³", f"{res.total_particles:,}")
        rcol2.metric("Particle Mass mp", f"{res.particle_mass_msun_h:.2e} h⁻¹ M☉")
        rcol3.metric("Mean Separation d", f"{res.mean_separation_mpc_h:.3f} h⁻¹ Mpc")
        rcol4.metric("Nyquist Wavenumber", f"{res.k_nyquist_h_mpc:.2f} h/Mpc")

        # A bounded visual geometry preview: it deliberately samples the
        # unperturbed lattice rather than pretending to show evolved matter.
        preview_dim = min(int(particles_per_dim), 12)
        preview_axis = np.linspace(
            float(box_size) / (2.0 * preview_dim),
            float(box_size) - float(box_size) / (2.0 * preview_dim),
            preview_dim,
        )
        preview_x, preview_y, preview_z = np.meshgrid(
            preview_axis, preview_axis, preview_axis, indexing="ij"
        )
        box_preview = go.Figure(
            go.Scatter3d(
                x=preview_x.ravel(),
                y=preview_y.ravel(),
                z=preview_z.ravel(),
                mode="markers",
                marker={"size": 2.5, "color": "#4fc3b7", "opacity": 0.7},
                hovertemplate="x=%{x:.3g}, y=%{y:.3g}, z=%{z:.3g} h⁻¹ Mpc<extra></extra>",
                name="Sampled initial lattice",
            )
        )
        box_edges = np.asarray(
            [
                [0, 0, 0], [box_size, 0, 0], [box_size, box_size, 0], [0, box_size, 0], [0, 0, 0],
                [0, 0, box_size], [box_size, 0, box_size], [box_size, box_size, box_size], [0, box_size, box_size], [0, 0, box_size],
                [np.nan, np.nan, np.nan], [box_size, 0, 0], [box_size, 0, box_size],
                [np.nan, np.nan, np.nan], [box_size, box_size, 0], [box_size, box_size, box_size],
                [np.nan, np.nan, np.nan], [0, box_size, 0], [0, box_size, box_size],
            ],
            dtype=float,
        )
        box_preview.add_trace(
            go.Scatter3d(
                x=box_edges[:, 0], y=box_edges[:, 1], z=box_edges[:, 2], mode="lines",
                line={"color": "#93a4a8", "width": 3}, hoverinfo="skip", showlegend=False,
            )
        )
        box_preview.update_layout(
            height=430,
            margin={"l": 0, "r": 0, "b": 0, "t": 34},
            title="Periodic volume and initial lattice preview",
            meta={
                "caption": (
                    f"A {preview_dim}³ sampled view of the unperturbed {particles_per_dim}³ particle lattice. "
                    "It shows geometry and spacing only; it is not an evolved N-body density field."
                ),
                "mislead": (
                    "Each visible point represents a sampled lattice location, not one plotted simulation particle. "
                    "Inspect a recorded GADGET-4 snapshot in the halo tab for an evolved particle distribution."
                ),
            },
            scene={
                "aspectmode": "cube",
                "xaxis_title": "x [h⁻¹ Mpc]",
                "yaxis_title": "y [h⁻¹ Mpc]",
                "zaxis_title": "z [h⁻¹ Mpc]",
                "xaxis": {"range": [0, box_size]},
                "yaxis": {"range": [0, box_size]},
                "zaxis": {"range": [0, box_size]},
            },
        )
        chart(
            box_preview,
            height=430,
            key="sim_box_geometry_preview",
        )

        st.markdown("#### Halo Particle Count Resolution Thresholds")
        hcol1, hcol2, hcol3, hcol4 = st.columns(4)
        hcol1.metric(
            "M₂₀ (20 particles, min group)",
            f"{res.min_halo_mass_20p_msun_h:.2e} h⁻¹ M☉",
        )
        hcol2.metric(
            "M₁₀₀ (100 particles, abundance)",
            f"{res.reliable_abundance_mass_100p_msun_h:.2e} h⁻¹ M☉",
        )
        hcol3.metric(
            "M₃₀₀ (300 particles, profile)",
            f"{res.well_resolved_mass_300p_msun_h:.2e} h⁻¹ M☉",
        )
        hcol4.metric(
            "M₁₀₀₀ (1000 particles, high-res)",
            f"{res.profile_resolved_mass_1000p_msun_h:.2e} h⁻¹ M☉",
        )

        if res.warnings:
            for w in res.warnings:
                st.warning(w)

    with t_ic:
        st.subheader("Linear (Zel'dovich/1LPT) Initial Conditions")
        st.warning(
            "This generator currently produces first-order Zel'dovich initial conditions. "
            "A separately tested 2LPT numerical kernel is not exposed here until independent "
            "benchmark evidence and validated EDE second-order growth inputs are available."
        )
        ic_col1, ic_col2, ic_col3 = st.columns(3)
        z_start = ic_col1.number_input(
            "Start Redshift z_start",
            min_value=10.0,
            max_value=199.0,
            value=49.0,
            key="sim_z_start",
        )
        seed = ic_col2.number_input("Random Seed", value=42, key="sim_seed")
        paired_fixed = ic_col3.toggle(
            "Paired-Fixed Phase Ensemble", value=False, key="sim_paired_fixed"
        )
        ic_source_run = current_pipeline_run()
        if ic_source_run is None:
            st.info(
                "Complete and load a CLASS/AxiCLASS run before generating ICs."
            )
        else:
            available_ic_redshifts = np.asarray(
                ic_source_run["power_result"]["redshifts"], dtype=float
            )
            st.caption(
                "IC generation requires an exact stored P(k,z_start) slice; available "
                "redshifts: "
                + ", ".join(f"{z:g}" for z in available_ic_redshifts)
                + ". Rerun the solver if your intended z_start is absent."
            )

        if st.button(
            "Generate & Verify Linear Initial Conditions",
            type="primary",
            key="sim_gen_ic_btn",
        ):
            scientific_run = current_pipeline_run()
            if scientific_run is None:
                st.error(
                    "A completed CLASS/AxiCLASS run is required. Initial conditions must use its stored linear P(k), not a substitute spectrum."
                )
                return
            power = scientific_run["power_result"]
            redshifts = np.asarray(power["redshifts"], dtype=float)
            source_params = scientific_run["params"]
            try:
                source_index = redshift_index(redshifts, float(z_start))
                growth = np.asarray(power["growth_class"], dtype=float)
                if growth.shape != redshifts.shape or redshifts.size < 2:
                    raise ValueError(
                        "The stored run needs at least two CLASS/AxiCLASS growth samples "
                        "to derive the IC velocity growth rate."
                    )
                log_a = -np.log1p(redshifts)
                growth_rate = float(np.gradient(np.log(growth), log_a)[source_index])
                background_omega_m = np.asarray(
                    power.get("background_omega_m_by_z", []), dtype=float
                )
                if background_omega_m.shape == redshifts.shape:
                    e_rate = float(
                        np.sqrt(
                            float(source_params["Omega_m"])
                            * (1.0 + float(z_start)) ** 3
                            / background_omega_m[source_index]
                        )
                    )
                elif source_params.get("enable_ede"):
                    raise ValueError(
                        "The stored EDE run has no usable AxiCLASS background sequence. "
                        "Regenerate it before producing EDE initial conditions."
                    )
                else:
                    omega_m0 = float(source_params["Omega_m"])
                    e_rate = float(
                        np.sqrt(
                            omega_m0 * (1.0 + float(z_start)) ** 3
                            + (1.0 - omega_m0)
                        )
                    )
                k_eval = np.asarray(power["k"], dtype=float)
                p_eval = np.asarray(power["P_by_z"], dtype=float)[source_index]
                pos, vel, ids, ic_report = generate_zeldovich_particles(
                    box_size,
                    particles_per_dim,
                    k_eval,
                    p_eval,
                    z_start,
                    source_params,
                    seed=seed,
                    paired_fixed=paired_fixed,
                    spectrum_redshift=float(redshifts[source_index]),
                    growth_rate=growth_rate,
                    expansion_rate_E=e_rate,
                )
            except ValueError as exc:
                st.error(
                    f"Initial conditions were not generated: {exc} Available stored redshifts: "
                    + ", ".join(f"{z:g}" for z in redshifts)
                )
                return
            st.session_state["sim_particles"] = (pos, vel, ids)
            st.session_state["sim_ic_report"] = ic_report
            st.session_state["sim_ic_source"] = {
                "run_id": st.session_state.get("current_run_id"),
                "redshift": float(redshifts[source_index]),
                "spectrum": "CLASS/AxiCLASS linear matter P(k)",
                "growth_rate": growth_rate,
                "expansion_rate_E": e_rate,
            }

        ic_rep = st.session_state.get("sim_ic_report")
        if ic_rep:
            (st.success if ic_rep.checks_passed else st.warning)(ic_rep.summary)
            if not ic_rep.checks_passed:
                st.warning(
                    "The IC realization failed its configured basic verification. Do not use it as a production input; adjust the grid/spectrum and regenerate."
                )
            vcol1, vcol2, vcol3, vcol4 = st.columns(4)
            vcol1.metric(
                "Center of Mass V_cm",
                f"{ic_rep.center_of_mass_velocity_km_s:.2e} km/s",
                help="Must be ~0 km/s for momentum conservation",
            )
            vcol2.metric(
                "Max Displacement", f"{ic_rep.max_displacement_mpc_h:.3f} h⁻¹ Mpc"
            )
            vcol3.metric(
                "Periodicity Check", "PASS" if ic_rep.periodicity_passed else "FAIL"
            )
            vcol4.metric(
                "Measured P(k) Shell Error",
                f"{ic_rep.power_agreement_median_fractional_error:.2%}",
                help=(
                    f"Median fractional difference across {ic_rep.power_agreement_bin_count} "
                    "measured Fourier shells. This is a realization diagnostic, not a 2LPT validation."
                ),
            )

    with t_conf:
        st.subheader("GADGET-4 Configuration Planning Files")
        config_sh = generate_config_sh(
            is_hydro=False, enable_2lpt=False, enable_fof=True, enable_subfind=True
        )
        param_txt = generate_gadget4_parameter_file(
            box_size,
            particles_per_dim,
            "output",
            [10.0, 5.0, 2.0, 1.0, 0.0],
            params,
            start_redshift=z_start,
        )
        output_times_txt = generate_output_times_file(
            [10.0, 5.0, 2.0, 1.0, 0.0], start_redshift=z_start
        )
        cf1, cf2, cf3, cf4 = st.tabs(
            [
                "Config.sh (Compile)",
                "param.txt (Runtime)",
                "output_times.txt",
                "EDE Runtime Status",
            ]
        )
        with cf1:
            st.code(config_sh, language="bash")
        with cf2:
            st.code(param_txt, language="text")
        with cf3:
            st.caption(
                "Desired output scale factors for z=10, 5, 2, 1, and 0. This plain-ASCII "
                "file is required by the generated `OutputListFilename output_times.txt` setting."
            )
            st.code(output_times_txt, language="text")
        with cf4:
            if params.get("enable_ede"):
                st.error(
                    "EDE GADGET-4 execution is unavailable. HaloForge does not emit a "
                    "phenomenological H(a) table or a fictitious runtime setting as a substitute "
                    "for an externally validated EDE GADGET-4 implementation."
                )
            else:
                st.info(
                    "These are standard-background planning files only. GADGET-4 execution "
                    "still requires the pinned image and its recorded official acceptance case."
                )

    with t_halo:
        st.subheader("Friends-of-Friends (FOF) on an Evolved GADGET-4 Snapshot")
        st.caption(
            "FOF is intentionally unavailable for generated initial conditions. Load an actual GADGET-4 PartType1 HDF5 snapshot produced by a recorded simulation run."
        )
        snapshot_path = st.text_input(
            "Local GADGET-4 HDF5 snapshot path",
            placeholder="/absolute/path/to/snap_###.hdf5",
            key="sim_snapshot_path",
        )
        if st.button("Validate and load snapshot", key="sim_load_snapshot"):
            try:
                # Upstream GADGET-4 snapshots can omit Omega0/HubbleParam.
                # Supply them only from the loaded calculation, never from
                # editable draft controls or global defaults.
                bound_run = current_pipeline_run()
                derived = (
                    bound_run.get("power_result", {}).get("derived", {})
                    if bound_run is not None
                    else {}
                )
                snapshot = load_gadget4_dm_snapshot(
                    snapshot_path,
                    omega_m=derived.get("Omega_m"),
                    h=derived.get("h"),
                )
                st.session_state["sim_snapshot"] = snapshot
                st.session_state.pop("sim_halo_catalogue", None)
                st.session_state.pop("sim_snapshot_manifest", None)
                st.success(
                    f"Loaded {len(snapshot.particle_ids):,} DM particles at z={snapshot.redshift:g} from a periodic {snapshot.box_size_mpc_h:g} h⁻¹ Mpc box."
                )
            except (ValueError, RuntimeError) as exc:
                st.error(f"Snapshot was not loaded: {exc}")

        snapshot = st.session_state.get("sim_snapshot")
        if snapshot is None:
            st.info("Load a validated local snapshot to enable FOF halo finding.")
        else:
            st.caption(
                f"Snapshot source: {snapshot.source_path} · z={snapshot.redshift:g} · a={snapshot.scale_factor:.6g} · "
                f"mₚ={snapshot.particle_mass_msun_h:.3e} h⁻¹ M☉ · Ωm={snapshot.omega_m:.6g}"
            )
            active_run_id = st.session_state.get("current_run_id")
            active_saved_run = (
                run_storage.load_run(active_run_id) if active_run_id else None
            )
            if active_saved_run is None:
                st.info(
                    "Load an integrity-checked saved linear run to bind this snapshot "
                    "to an exact cosmology before HMF comparison."
                )
            else:
                sidecar = manifest_path_for_snapshot(snapshot.source_path)
                manifest_columns = st.columns(2)
                if manifest_columns[0].button(
                    "Bind snapshot to active saved run",
                    key="sim_write_snapshot_manifest",
                ):
                    try:
                        written = write_snapshot_manifest(snapshot, active_saved_run)
                        st.success(f"Wrote immutable snapshot manifest: {written.name}")
                    except ValueError as exc:
                        st.error(f"Snapshot was not bound: {exc}")
                if manifest_columns[1].button(
                    "Verify snapshot-run binding",
                    key="sim_verify_snapshot_manifest",
                ):
                    try:
                        st.session_state["sim_snapshot_manifest"] = (
                            load_and_validate_snapshot_manifest(
                                snapshot, active_saved_run
                            )
                        )
                        st.success("Exact snapshot bytes and saved-run identity match.")
                    except ValueError as exc:
                        st.session_state.pop("sim_snapshot_manifest", None)
                        st.error(f"Snapshot-run binding is not valid: {exc}")
                st.caption(
                    f"Required sidecar: {sidecar.name}. This records file identity and headers; it is not a GADGET-4 or EDE validation certificate."
                )
            if st.button("Run Periodic FOF Group Finder", key="sim_fof_btn"):
                cat = find_fof_halos(
                    snapshot.positions_mpc_h,
                    snapshot.velocities_raw,
                    snapshot.box_size_mpc_h,
                    snapshot.particle_mass_msun_h,
                    redshift=snapshot.redshift,
                    linking_length_b=0.2,
                    min_particles=20,
                    omega_m=snapshot.omega_m,
                )
                st.session_state["sim_halo_catalogue"] = cat

            cat = st.session_state.get("sim_halo_catalogue")
            if cat:
                st.info(cat.summary)
                theme = st.session_state.get("accessibility_theme", "Dark")
                fig_3d = render_3d_halo_view(cat, theme=theme)
                chart(fig_3d, 520, "sim_3d_halo_view", axis_controls=False)
                df_halos = catalogue_to_dataframe(cat)
                if not df_halos.empty:
                    st.dataframe(df_halos.head(50), hide_index=True, width="stretch")
                snapshot_manifest = st.session_state.get("sim_snapshot_manifest")
                if snapshot_manifest is None:
                    st.info(
                        "Verify the snapshot-to-run binding before saving a durable catalogue."
                    )
                elif st.button("Save versioned FOF catalogue", key="sim_save_fof_catalogue"):
                    try:
                        stored_catalogue = save_fof_catalogue(
                            cat, snapshot, snapshot_manifest
                        )
                        st.success(
                            "Saved HDF5, Parquet, and manifest records in the local research vault: "
                            + stored_catalogue.catalogue_id
                        )
                    except ValueError as exc:
                        st.error(f"Catalogue was not saved: {exc}")

    with t_hmf:
        st.subheader("Finite-Bin Abundance vs. Analytic Halo Mass Function")
        cat = st.session_state.get("sim_halo_catalogue")
        if not cat or cat.is_empty_due_to_resolution:
            st.info(
                "A resolved halo catalogue is required to perform HMF abundance validation."
            )
        else:
            scientific_run = current_pipeline_run()
            if scientific_run is None:
                st.error(
                    "A completed CLASS/AxiCLASS run is required for comparison. No surrogate σ(M) is used."
                )
            else:
                fof_fits = [
                    name
                    for name in FITTING_NAMES
                    if fit_contract(name).mass_definition == "fof_b0.2"
                ]
                comparison_fit = st.selectbox(
                    "FOF b=0.2-compatible HMF fit",
                    fof_fits,
                    index=fof_fits.index(params["fitting"])
                    if params.get("fitting") in fof_fits
                    else 0,
                    help="Spherical-overdensity and analytic-top-hat fits are unavailable: this catalogue contains only FOF b=0.2 masses.",
                    key="sim_hmf_fof_fit",
                )
                sigma_result = scientific_run["sigma_result"]
                active_h = float(scientific_run["power_result"]["derived"]["h"])
                active_omega_m = float(scientific_run["power_result"]["derived"]["Omega_m"])
                snapshot = st.session_state.get("sim_snapshot")
                if snapshot is None:
                    st.error("The catalogue is missing its validated snapshot provenance.")
                    return
                active_run_id = st.session_state.get("current_run_id")
                active_saved_run = (
                    run_storage.load_run(active_run_id) if active_run_id else None
                )
                if active_saved_run is None:
                    st.error("Load the saved linear run that produced this comparison.")
                    return
                try:
                    manifest = load_and_validate_snapshot_manifest(
                        snapshot, active_saved_run
                    )
                    st.session_state["sim_snapshot_manifest"] = manifest
                except ValueError as exc:
                    st.error(
                        "Exact cosmology comparison is blocked until the snapshot is "
                        f"bound and verified against this saved run: {exc}"
                    )
                    return
                if not (
                    np.isclose(snapshot.h, active_h, rtol=0, atol=1e-8)
                    and np.isclose(snapshot.omega_m, active_omega_m, rtol=0, atol=1e-8)
                ):
                    st.error(
                        "Snapshot and active linear run have different Hubble or matter-density headers. "
                        "Cross-cosmology HMF comparison is prohibited."
                    )
                    return
                sigma_redshifts = np.asarray(sigma_result["redshifts"], dtype=float)
                sigma_index = redshift_index(sigma_redshifts, float(cat.redshift))
                if not np.isclose(
                    sigma_redshifts[sigma_index], float(cat.redshift), rtol=0, atol=1e-8
                ):
                    st.error(
                        f"The active linear run has no exact σ(M) slice at snapshot z={cat.redshift:g}. "
                        "Rerun CLASS/AxiCLASS with that redshift; nearest-redshift substitution is prohibited."
                    )
                    return
                h = active_h
                comp_report = compare_catalogue_to_analytic_hmf(
                    cat,
                    mass_grid_h=np.asarray(sigma_result["M_h"], dtype=float),
                    sigma_grid=np.asarray(sigma_result["sigma_by_z"], dtype=float)[sigma_index],
                    rho0=float(sigma_result["rho0"]),
                    h=h,
                    fitting=comparison_fit,
                    num_mass_bins=8,
                )
                st.caption(
                    f"Analytic input: stored CLASS/AxiCLASS σ(M,z={sigma_redshifts[sigma_index]:g}) from the active run. "
                    "The immutable sidecar binds these exact snapshot bytes to the active saved linear run; H0 and Ωm headers match. The catalogue is strictly FOF b=0.2; no spherical-overdensity mass is inferred. This remains an unvalidated comparison, especially for EDE."
                )
                st.info(comp_report.interpretation)
                with st.expander("Uncertainty accounting", expanded=False):
                    st.caption(
                        "Only the explicitly included source appears as an error bar. "
                        "Unquantified sources are retained as open evidence, not silently treated as zero."
                    )
                    st.dataframe(
                        pd.DataFrame(comp_report.uncertainty_sources),
                        hide_index=True,
                        width="stretch",
                    )
                theme = st.session_state.get("accessibility_theme", "Dark")
                hmf_fig = render_hmf_comparison_plot(comp_report, theme=theme)
                chart(hmf_fig, 540, "sim_hmf_comp_chart", axis_controls=True)


def render_project_header():
    loaded_id = st.session_state.get("current_run_id") or st.session_state.get(
        "loaded_run_id"
    )
    run = run_storage.load_run(loaded_id) if loaded_id else None
    runs = run_storage.load_all_runs()
    baseline = next((r for r in runs if r.get("is_baseline")), None)

    active_name = run["name"] if run else "Draft (unsaved)"
    baseline_name = baseline["name"] if baseline else "None selected"
    draft_status = "Saved" if run else "Draft parameters"

    params = get_params()
    z_val = float(st.session_state.get("hf_z_index", 0.0))
    if run and "redshifts" in run.get("arrays", {}):
        z_val = float(run["arrays"]["redshifts"][0])

    validity_text = "Standard ΛCDM"
    validity_cls = "badge-valid"
    if params.get("enable_ede"):
        validity_text = "Early Dark Energy (Caveats apply)"
        validity_cls = "badge-warning"

    st.markdown(
        f"""
        <div class="project-header">
            <div class="project-header-item">
                <span class="project-header-label">ACTIVE UNIVERSE</span>
                <span class="project-header-val">{active_name}</span>
            </div>
            <div class="project-header-item">
                <span class="project-header-label">BASELINE</span>
                <span class="project-header-val">{baseline_name}</span>
            </div>
            <div class="project-header-item">
                <span class="project-header-label">STATE</span>
                <span class="project-header-val">{draft_status}</span>
            </div>
            <div class="project-header-item">
                <span class="project-header-label">PHYSICAL VALIDITY</span>
                <span class="project-badge {validity_cls}">{validity_text}</span>
            </div>
            <div class="project-header-item">
                <span class="project-header-label">TARGET REDSHIFT</span>
                <span class="project-header-val">z = {z_val:.2f}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


section = sidebar()
if section != "Explore":
    render_project_header()

views = {
    "Explore": explore_view,
    "Dashboard": dashboard_view,
    "Graph studio": graph_studio_view,
    "Compare lab": compare_view,
    "Sensitivity explorer": sensitivity_view,
    "Design experiment": experiment_design_view,
    "Benchmark lab": benchmark_view,
    "Performance lab": performance_view,
    "Convergence lab": convergence_view,
    "Notebook": notebook_view,
    "Known limitations": limitations_view,
    "Teach": teach_view,
    "Structure field": structure_view,
    "Learn the pipeline": learn_view,
    "Fit + window atlas": atlas_view,
    "Evolution studio": evolution_studio_view,
    "Campaign lab": campaign_lab_view,
    "Simulation lab": simulation_lab_view,
    "Runs + export": runs_view,
    "Diagnostics": diagnostics_view,
}

try:
    views[section]()
except Exception as exc:
    if os.environ.get("PYTEST_CURRENT_TEST"):
        raise exc
    st.error("### HaloForge Recoverable Error")
    st.markdown(
        f"An unexpected error occurred while rendering the **{section}** workspace. "
        "Your saved runs and local data files remain completely intact."
    )
    with st.expander("Technical Diagnostic Record & Stack Trace", expanded=False):
        import traceback

        tb = traceback.format_exc()
        st.code(tb, language="python")
        st.download_button(
            "Download Diagnostic Record",
            data=json.dumps(
                {"error": str(exc), "traceback": tb, "workspace": section}, indent=2
            ),
            file_name=f"haloforge_error_{section.lower().replace(' ', '_')}.json",
            mime="application/json",
        )
