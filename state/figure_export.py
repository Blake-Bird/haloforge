"""Static publication figures derived solely from a saved HaloForge run."""

from __future__ import annotations

import tempfile
import textwrap
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

from engine.redshift import redshift_index


class FigureExportError(RuntimeError):
    """Raised when static figure export is unavailable or invalid."""


PAPER_COLORS = ("#17627f", "#8a5a14", "#804d87")
GRAYSCALE_COLORS = ("#111111", "#5f5f5f", "#969696")
GRAYSCALE_DASHES = ("solid", "dash", "dot")


def _paper_layout(fig: go.Figure, title: str, caption: str) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        width=960,
        height=650,
        title=dict(
            text=title, font=dict(family="Arial", size=22, color="#102a3b"), x=0.02
        ),
        font=dict(family="Arial", size=14, color="#17252c"),
        margin=dict(l=90, r=36, t=90, b=150),
        legend=dict(orientation="h", y=1.02, x=0),
        annotations=[
            dict(
                text="<br>".join(
                    line
                    for paragraph in caption.split("<br>")
                    for line in textwrap.wrap(paragraph, width=110)
                ),
                x=0,
                xref="paper",
                y=-0.23,
                yanchor="top",
                xanchor="left",
                yref="paper",
                showarrow=False,
                align="left",
                font=dict(size=11, color="#40535d"),
            )
        ],
    )
    fig.update_xaxes(exponentformat="power", showexponent="all", minorloglabels="none")
    fig.update_yaxes(exponentformat="power", showexponent="all", minorloglabels="none")
    return fig


def _static_formats(figures: list[tuple[str, go.Figure]]) -> dict[str, bytes]:
    """Batch render all figures so Kaleido starts once per output format."""
    output: dict[str, bytes] = {}
    try:
        with tempfile.TemporaryDirectory(prefix="haloforge-figures-") as folder:
            root = Path(folder)
            for image_format in ("pdf", "svg", "png"):
                paths = [root / f"{stem}.{image_format}" for stem, _ in figures]
                pio.write_images(
                    [figure for _, figure in figures],
                    paths,
                    format=image_format,
                    width=960,
                    height=650,
                    scale=2 if image_format == "png" else 1,
                )
                output.update({path.name: path.read_bytes() for path in paths})
    except Exception as exc:
        raise FigureExportError(f"Static figure export failed: {exc}") from exc
    return output


def static_figure_export_smoke_test() -> dict[str, int]:
    """Render one minimal figure in every promised static format.

    This deliberately tests the deployed Kaleido/browser runtime, not a
    scientific run. It is used by container CI so missing browser binaries are
    caught before a release can advertise PDF, SVG, or PNG exports.
    """
    figure = go.Figure(go.Scatter(x=[1.0, 2.0], y=[1.0, 4.0], mode="lines"))
    files = _static_formats([("static_export_smoke", figure)])
    expected = {
        "static_export_smoke.pdf",
        "static_export_smoke.svg",
        "static_export_smoke.png",
    }
    if set(files) != expected or any(not content for content in files.values()):
        raise FigureExportError("Static figure smoke test did not produce all formats.")
    return {filename: len(content) for filename, content in files.items()}


def _grayscale_variant(fig: go.Figure, title: str, caption: str) -> go.Figure:
    """Make a print-safe figure whose series remain distinct without color."""
    variant = go.Figure(fig)
    for index, trace in enumerate(variant.data):
        trace.line.color = GRAYSCALE_COLORS[index % len(GRAYSCALE_COLORS)]
        trace.line.dash = GRAYSCALE_DASHES[index % len(GRAYSCALE_DASHES)]
        trace.line.width = 3.2 if index == 0 else 2.8
    return _paper_layout(
        variant,
        title,
        f"{caption}<br>Line patterns distinguish the series in grayscale.",
    )


def _with_grayscale_variants(
    figures: list[tuple[str, go.Figure, str, str]],
) -> list[tuple[str, go.Figure]]:
    paired: list[tuple[str, go.Figure]] = []
    for stem, figure, title, caption in figures:
        paired.append((stem, figure))
        paired.append((f"{stem}_grayscale", _grayscale_variant(figure, title, caption)))
    return paired


def build_publication_figures(run: dict) -> list[tuple[str, go.Figure]]:
    """Build inspectable figures from saved samples before invoking a renderer."""
    if run.get("integrity_status", {}).get("state") == "invalid":
        raise FigureExportError(
            "Cannot export figures from a run with failed integrity checks."
        )
    arrays = run.get("arrays", {})
    k, power = (
        np.asarray(arrays.get("k", []), dtype=float),
        np.asarray(arrays.get("P", []), dtype=float),
    )
    if (
        k.ndim != 1
        or power.shape != k.shape
        or k.size < 2
        or not np.isfinite(k).all()
        or not np.isfinite(power).all()
        or np.any(np.diff(k) <= 0)
        or np.any(k <= 0)
        or np.any(power <= 0)
    ):
        raise FigureExportError(
            "A positive stored P(k) grid is required for static figure export."
        )
    redshift = float(run.get("params", {}).get("single_z", 0.0))
    redshifts = np.asarray(arrays.get("redshifts", [0.0]), dtype=float)
    try:
        index = redshift_index(redshifts, redshift)
    except ValueError as exc:
        raise FigureExportError(str(exc)) from exc
    if "P_by_z" in arrays:
        spectra = np.asarray(arrays["P_by_z"], dtype=float)
        if spectra.shape != (redshifts.size, k.size):
            raise FigureExportError("Stored P(k,z) dimensions do not match its grids.")
        power = spectra[index]
    elif redshift != 0:
        raise FigureExportError(
            "Focused-redshift matter power is not stored in this run."
        )
    if not np.isfinite(power).all() or np.any(power <= 0):
        raise FigureExportError("Focused matter power must be finite and positive.")
    figures: list[tuple[str, go.Figure, str, str]] = []
    power_fig = go.Figure(
        go.Scatter(
            x=k,
            y=power,
            mode="lines",
            line=dict(color=PAPER_COLORS[0], width=3),
            name="stored linear P(k)",
            hovertemplate="k=%{x:.3e} Mpc⁻¹<br>P=%{y:.3e} Mpc³<extra></extra>",
        )
    )
    power_fig.update_xaxes(type="log", title="k [Mpc⁻¹]")
    power_fig.update_yaxes(type="log", title="Linear P(k) [Mpc³]")
    power_title = f"Linear matter power · z = {redshift:g}"
    power_caption = f"Stored CLASS/AxiCLASS linear spectrum at z = {redshift:g}. Solver settings and provenance accompany this figure in class_settings.json and provenance.json."
    figures.append(
        (
            "matter_power",
            _paper_layout(power_fig, power_title, power_caption),
            power_title,
            power_caption,
        )
    )

    masses, sigma = (
        np.asarray(arrays.get("M_h", []), dtype=float),
        np.asarray(arrays.get("sigma", []), dtype=float),
    )
    if "sigma_by_z" in arrays:
        variances = np.asarray(arrays["sigma_by_z"], dtype=float)
        if variances.shape != (redshifts.size, masses.size):
            raise FigureExportError(
                "Stored sigma(M,z) dimensions do not match its grids."
            )
        sigma = variances[index]
    elif sigma.size and redshift != 0:
        raise FigureExportError(
            "Focused-redshift mass variance is not stored in this run."
        )
    if masses.size or sigma.size:
        if (
            masses.ndim != 1
            or masses.size < 2
            or sigma.shape != masses.shape
            or not np.isfinite(masses).all()
            or np.any(masses <= 0)
            or np.any(np.diff(masses) <= 0)
            or not np.isfinite(sigma).all()
            or np.any(sigma <= 0)
        ):
            raise FigureExportError(
                "A positive finite increasing mass grid and matching sigma values are required."
            )
        sigma_fig = go.Figure(
            go.Scatter(
                x=masses,
                y=sigma,
                mode="lines",
                line=dict(color=PAPER_COLORS[1], width=3),
                name="stored sigma(M)",
            )
        )
        sigma_fig.update_xaxes(type="log", title="M [h⁻¹ M☉]")
        sigma_fig.update_yaxes(type="log", title="σ(M)")
        sigma_title = f"Mass variance · z = {redshift:g}"
        window = run.get(
            "window_type", run.get("params", {}).get("window_type", "Top-hat")
        )
        sigma_caption = f"Stored {window} smoothing at z = {redshift:g}. Sampled-range checks do not establish convergence. Sources and limitations: citation_metadata.json and scientific_validity.json."
        figures.append(
            (
                "mass_variance",
                _paper_layout(sigma_fig, sigma_title, sigma_caption),
                sigma_title,
                sigma_caption,
            )
        )

    ps, st = (
        np.asarray(arrays.get("hmf_press_schechter_z0", []), dtype=float),
        np.asarray(arrays.get("hmf_sheth_tormen_z0", []), dtype=float),
    )
    if ps.size or st.size:
        if (
            ps.shape != masses.shape
            or st.shape != masses.shape
            or masses.size < 2
            or not np.isfinite(ps).all()
            or not np.isfinite(st).all()
            or np.any(ps < 0)
            or np.any(st < 0)
        ):
            raise FigureExportError(
                "Stored HMF reference arrays must be finite, nonnegative, and match the mass grid."
            )
        hmf_fig = go.Figure()
        hmf_fig.add_trace(
            go.Scatter(
                x=masses,
                y=np.where(ps > 0, ps, np.nan),
                mode="lines",
                line=dict(color=PAPER_COLORS[0], width=2.8),
                name="Press-Schechter 1974",
            )
        )
        hmf_fig.add_trace(
            go.Scatter(
                x=masses,
                y=np.where(st > 0, st, np.nan),
                mode="lines",
                line=dict(color=PAPER_COLORS[2], width=2.8),
                name="Sheth-Tormen 2001",
            )
        )
        hmf_fig.update_xaxes(type="log", title="M [h⁻¹ M☉]")
        hmf_fig.update_yaxes(type="log", title="dn/dln M [h³ Mpc⁻³]")
        hmf_title = "Analytic halo mass-function references"
        hmf_caption = "Top-hat reference curves at z = 0. Zero abundances lie outside the logarithmic axis. These curves do not establish cosmology-specific calibration; see scientific_validity.json."
        figures.append(
            (
                "analytic_hmf_reference",
                _paper_layout(hmf_fig, hmf_title, hmf_caption),
                hmf_title,
                hmf_caption,
            )
        )
    return _with_grayscale_variants(figures)


def build_figure_exports(run: dict) -> dict[str, bytes]:
    """Render matching PDF, SVG, and PNG files from validated saved samples."""
    return _static_formats(build_publication_figures(run))


def build_figure_pdfs(run: dict) -> dict[str, bytes]:
    """Backward-compatible PDF-only view of :func:`build_figure_exports`."""
    return {
        filename: content
        for filename, content in build_figure_exports(run).items()
        if filename.endswith(".pdf")
    }
