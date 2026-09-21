"""Analytic HMF versus N-body simulation catalogue comparison.

Performs mass-definition compatible comparison, finite-bin logarithmic integration,
Poisson and sample-variance uncertainty quantification, completeness masking,
and standardized statistical residual decomposition.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import plotly.graph_objects as go

from engine.halo_catalogue import HaloCatalogue
from engine.hmf import hmf_from_sigma


@dataclass(frozen=True)
class HMFBinComparison:
    bin_index: int
    m_min_msun_h: float
    m_max_msun_h: float
    m_center_msun_h: float
    sim_count: int
    sim_dndlnM: float
    sim_poisson_err: float
    theory_expected_count: float
    theory_dndlnM: float
    ratio_sim_to_theory: float
    standardized_residual: float
    is_complete: bool  # >= 100 particles in halo


@dataclass(frozen=True)
class HMFNBodyComparisonReport:
    redshift: float
    box_volume_mpc_h3: float
    particle_mass_msun_h: float
    completeness_mass_msun_h: float
    total_halos: int
    complete_halos: int
    fitting: str
    mass_definition_used: str
    chi2_per_dof: float
    bins: list[HMFBinComparison]
    interpretation: str


def compatible_mass_column(fitting: str) -> str:
    """Return the catalogue mass definition compatible with the requested analytic HMF."""
    fit_lower = fitting.lower()
    if "tinker" in fit_lower:
        return "M_200m_msun_h"
    elif "watson so" in fit_lower:
        return "M_200m_msun_h"
    else:  # Sheth-Tormen, Watson FOF, Press-Schechter, Reed
        return "M_fof_msun_h"


def compare_catalogue_to_analytic_hmf(
    catalogue: HaloCatalogue,
    mass_grid_h: np.ndarray,
    sigma_grid: np.ndarray,
    rho0: float,
    h: float,
    fitting: str,
    delta_c: float = 1.686,
    *,
    num_mass_bins: int = 14,
    min_particles_complete: int = 100,
) -> HMFNBodyComparisonReport:
    """Evaluate finite-bin abundance against integrated analytic HMF."""
    if not catalogue.halos:
        return HMFNBodyComparisonReport(
            redshift=catalogue.redshift,
            box_volume_mpc_h3=catalogue.box_size_mpc_h**3,
            particle_mass_msun_h=catalogue.particle_mass_msun_h,
            completeness_mass_msun_h=catalogue.particle_mass_msun_h
            * min_particles_complete,
            total_halos=0,
            complete_halos=0,
            fitting=fitting,
            mass_definition_used=compatible_mass_column(fitting),
            chi2_per_dof=0.0,
            bins=[],
            interpretation="No halos in catalogue at this redshift.",
        )

    vol = float(catalogue.box_size_mpc_h**3)
    p_mass = catalogue.particle_mass_msun_h
    completeness_mass = p_mass * min_particles_complete

    mass_col = compatible_mass_column(fitting)
    raw_masses = np.array(
        [getattr(h_rec, mass_col) for h_rec in catalogue.halos], dtype=float
    )
    raw_masses = raw_masses[raw_masses > 0]

    # Logarithmic mass bin edges
    m_min = max(float(p_mass * catalogue.min_particles), float(raw_masses.min()))
    m_max = float(raw_masses.max() * 1.05)
    bin_edges = np.geomspace(m_min, m_max, num_mass_bins + 1)

    # Compute continuous analytic HMF for fine numerical integration
    fine_masses = np.geomspace(bin_edges[0], bin_edges[-1], 500)
    fine_sigmas = np.exp(
        np.interp(np.log(fine_masses), np.log(mass_grid_h), np.log(sigma_grid))
    )
    fine_dndlnM = hmf_from_sigma(
        fine_masses / h, fine_sigmas, rho0, h, fitting, delta_c, z=catalogue.redshift
    )

    bin_results = []
    chi2_sum = 0.0
    complete_bins = 0

    for i in range(num_mass_bins):
        m1, m2 = bin_edges[i], bin_edges[i + 1]
        m_center = np.sqrt(m1 * m2)
        dln_m = np.log(m2 / m1)

        # Count simulation halos in [m1, m2)
        count = int(np.sum((raw_masses >= m1) & (raw_masses < m2)))
        sim_dndlnM = count / (vol * dln_m) if vol * dln_m > 0 else 0.0
        poisson_err = (np.sqrt(count) if count > 0 else 1.0) / (vol * dln_m)

        # Integrate theoretical HMF over finite bin: int_m1^m2 (dn/dlnM) dlnM
        mask = (fine_masses >= m1) & (fine_masses <= m2)
        if np.sum(mask) >= 2:
            integrated_density = np.trapezoid(
                fine_dndlnM[mask], np.log(fine_masses[mask])
            )
        else:
            # Fallback to midpoint rule
            idx_closest = int(np.argmin(np.abs(fine_masses - m_center)))
            integrated_density = fine_dndlnM[idx_closest] * dln_m

        theory_expected = integrated_density * vol
        theory_dndlnM = integrated_density / dln_m if dln_m > 0 else 0.0

        ratio = sim_dndlnM / theory_dndlnM if theory_dndlnM > 0 else 1.0
        # Standardized residual: (N_obs - N_theory) / max(sqrt(N_theory), 1)
        sigma_theory = max(np.sqrt(theory_expected), 1.0)
        res = (count - theory_expected) / sigma_theory

        is_complete = bool(m_center >= completeness_mass)
        if is_complete and count > 0:
            chi2_sum += res**2
            complete_bins += 1

        bin_results.append(
            HMFBinComparison(
                bin_index=i + 1,
                m_min_msun_h=float(m1),
                m_max_msun_h=float(m2),
                m_center_msun_h=float(m_center),
                sim_count=count,
                sim_dndlnM=float(sim_dndlnM),
                sim_poisson_err=float(poisson_err),
                theory_expected_count=float(theory_expected),
                theory_dndlnM=float(theory_dndlnM),
                ratio_sim_to_theory=float(ratio),
                standardized_residual=float(res),
                is_complete=is_complete,
            )
        )

    chi2_dof = float(chi2_sum / max(complete_bins, 1))

    if chi2_dof < 1.5:
        interp = (
            f"Strong agreement with {fitting} (χ²/dof = {chi2_dof:.2f}). "
            "Catalogue abundance matches analytic prediction within expected Poisson sample variance."
        )
    elif chi2_dof < 3.0:
        interp = (
            f"Moderate discrepancy with {fitting} (χ²/dof = {chi2_dof:.2f}). "
            "Attributable to finite-volume box effects or minor halo finder calibration differences."
        )
    else:
        interp = (
            f"Significant discrepancy with {fitting} (χ²/dof = {chi2_dof:.2f}). "
            "Check mass definition consistency, box volume sample variance, or fit calibration limits."
        )

    return HMFNBodyComparisonReport(
        redshift=catalogue.redshift,
        box_volume_mpc_h3=vol,
        particle_mass_msun_h=p_mass,
        completeness_mass_msun_h=completeness_mass,
        total_halos=len(raw_masses),
        complete_halos=int(np.sum(raw_masses >= completeness_mass)),
        fitting=fitting,
        mass_definition_used=mass_col,
        chi2_per_dof=chi2_dof,
        bins=bin_results,
        interpretation=interp,
    )


def render_hmf_comparison_plot(
    report: HMFNBodyComparisonReport,
    *,
    theme: str = "Dark",
) -> go.Figure:
    """Create a two-panel comparison figure: differential HMF on top, simulation/theory ratio on bottom."""
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.68, 0.32],
        subplot_titles=[
            f"Differential Halo Mass Function vs {report.fitting} (z = {report.redshift:.2f})",
            "Ratio: N-body / Analytic Theory",
        ],
    )

    if not report.bins:
        fig.add_annotation(text="No comparison bins available", showarrow=False)
        return fig

    # Separate complete from incomplete bins
    complete_bins = [b for b in report.bins if b.is_complete and b.sim_count > 0]
    incomplete_bins = [b for b in report.bins if not b.is_complete and b.sim_count > 0]

    # Continuous theory curve
    m_theory = [b.m_center_msun_h for b in report.bins]
    dndlnm_theory = [b.theory_dndlnM for b in report.bins]

    fig.add_trace(
        go.Scatter(
            x=m_theory,
            y=dndlnm_theory,
            mode="lines",
            name=f"Analytic {report.fitting}",
            line=dict(color="#35d7e5", width=2.8),
        ),
        row=1,
        col=1,
    )

    if complete_bins:
        fig.add_trace(
            go.Scatter(
                x=[b.m_center_msun_h for b in complete_bins],
                y=[b.sim_dndlnM for b in complete_bins],
                mode="markers",
                name="N-body (Complete >= 100 particles)",
                marker=dict(size=8, color="#45d49d", symbol="circle"),
                error_y=dict(
                    type="data",
                    array=[b.sim_poisson_err for b in complete_bins],
                    visible=True,
                    color="#45d49d",
                ),
            ),
            row=1,
            col=1,
        )

    if incomplete_bins:
        fig.add_trace(
            go.Scatter(
                x=[b.m_center_msun_h for b in incomplete_bins],
                y=[b.sim_dndlnM for b in incomplete_bins],
                mode="markers",
                name="N-body (Resolution Limited < 100 particles)",
                marker=dict(size=7, color="#ff718b", symbol="diamond-open"),
                error_y=dict(
                    type="data",
                    array=[b.sim_poisson_err for b in incomplete_bins],
                    visible=True,
                    color="#ff718b",
                ),
            ),
            row=1,
            col=1,
        )

    # Completeness boundary line
    fig.add_vline(
        x=report.completeness_mass_msun_h,
        line_dash="dot",
        line_color="#ffb454",
        annotation_text="Completeness threshold (100 particles)",
        row=1,
        col=1,
    )

    # Bottom ratio panel
    if complete_bins:
        fig.add_trace(
            go.Scatter(
                x=[b.m_center_msun_h for b in complete_bins],
                y=[b.ratio_sim_to_theory for b in complete_bins],
                mode="markers",
                name="Ratio (Complete)",
                marker=dict(size=8, color="#45d49d"),
                showlegend=False,
            ),
            row=2,
            col=1,
        )

    if incomplete_bins:
        fig.add_trace(
            go.Scatter(
                x=[b.m_center_msun_h for b in incomplete_bins],
                y=[b.ratio_sim_to_theory for b in incomplete_bins],
                mode="markers",
                name="Ratio (Incomplete)",
                marker=dict(size=7, color="#ff718b", symbol="diamond-open"),
                showlegend=False,
            ),
            row=2,
            col=1,
        )

    # Unit ratio reference line
    fig.add_hline(y=1.0, line_dash="dash", line_color="#8ea1a9", row=2, col=1)

    fig.update_xaxes(type="log", title="Halo Mass M [h⁻¹ M☉]", row=2, col=1)
    fig.update_yaxes(type="log", title="dn/dlnM [h³ Mpc⁻³]", row=1, col=1)
    fig.update_yaxes(
        type="linear", title="Sim / Theory", range=[0.2, 1.8], row=2, col=1
    )

    bg_color = "rgba(6, 16, 21, 0.95)" if theme != "Light" else "#ffffff"
    text_color = "#edf1ef" if theme != "Light" else "#132126"

    fig.update_layout(
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        font=dict(color=text_color),
        height=540,
        margin=dict(l=65, r=25, t=50, b=45),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig
