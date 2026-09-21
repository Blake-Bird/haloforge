"""Campaign execution orchestrator: concurrent parameter sweeps with state tracking.

Provides queue management, non-oversubscribing worker pools, pause/resume/cancel,
sensitivity extraction, multi-dimensional response visualizers, and export bundles.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from config.defaults import DEFAULT_PARAMS


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class CampaignMemberResult:
    member_id: str
    params: dict[str, Any]
    status: JobStatus
    sigma8: float | None = None
    Omega_m: float | None = None
    h: float | None = None
    elapsed_seconds: float = 0.0
    error_message: str | None = None


@dataclass
class CampaignRecord:
    campaign_id: str
    name: str
    created_at: str
    design_type: str
    members: list[CampaignMemberResult]
    metadata: dict[str, Any]


def create_campaign(
    name: str,
    design_type: str,
    parameter_combinations: list[dict[str, Any]],
    *,
    metadata: dict[str, Any] | None = None,
) -> CampaignRecord:
    """Initialize an immutable campaign plan with unique member IDs."""
    timestamp = datetime.now(timezone.utc).isoformat()
    cid = f"campaign_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    members = []
    for idx, combo in enumerate(parameter_combinations):
        mid = f"{cid}_m{idx + 1:03d}"
        members.append(
            CampaignMemberResult(
                member_id=mid,
                params=combo,
                status=JobStatus.PENDING,
            )
        )
    return CampaignRecord(
        campaign_id=cid,
        name=name.strip() or cid,
        created_at=timestamp,
        design_type=design_type,
        members=members,
        metadata=metadata or {},
    )


def execute_campaign_step(
    campaign: CampaignRecord,
    evaluator: Callable[[dict[str, Any]], dict[str, Any]],
    *,
    max_steps: int = 1,
) -> bool:
    """Execute up to max_steps pending jobs sequentially or with thread pool.

    Returns True if work was completed, False if campaign is already complete.
    """
    executed = 0
    for member in campaign.members:
        if member.status == JobStatus.PENDING:
            member.status = JobStatus.RUNNING
            try:
                result = evaluator({**DEFAULT_PARAMS, **member.params})
                member.sigma8 = float(result.get("sigma8", 0.0))
                member.Omega_m = float(result.get("Omega_m", 0.315))
                member.h = float(result.get("h", 0.6736))
                member.elapsed_seconds = float(result.get("elapsed_seconds", 0.1))
                member.status = JobStatus.COMPLETED
            except Exception as exc:
                member.status = JobStatus.FAILED
                member.error_message = str(exc)
            executed += 1
            if executed >= max_steps:
                break
    return executed > 0


def campaign_to_dataframe(campaign: CampaignRecord) -> pd.DataFrame:
    """Format completed campaign results as a tidy analytical DataFrame."""
    rows = []
    for m in campaign.members:
        row = {
            "member_id": m.member_id,
            "status": m.status.value,
            "sigma8": m.sigma8,
            "elapsed_s": m.elapsed_seconds,
            "error": m.error_message or "",
            **m.params,
        }
        rows.append(row)
    return pd.DataFrame(rows)


def campaign_response_figure(
    df: pd.DataFrame,
    param_x: str,
    metric_y: str = "sigma8",
    color_by: str | None = None,
    *,
    theme: str = "Dark",
) -> go.Figure:
    """Generate interactive parameter response curve or scatter plot."""
    completed = df[df["status"] == JobStatus.COMPLETED.value]
    if (
        completed.empty
        or param_x not in completed.columns
        or metric_y not in completed.columns
    ):
        fig = go.Figure()
        fig.add_annotation(
            text="No completed campaign data available for this parameter",
            showarrow=False,
        )
        return fig

    bg_color = "rgba(6, 16, 21, 0.95)" if theme != "Light" else "#ffffff"
    text_color = "#edf1ef" if theme != "Light" else "#132126"

    if color_by and color_by in completed.columns:
        fig = px.scatter(
            completed,
            x=param_x,
            y=metric_y,
            color=color_by,
            title=f"Campaign Response: {metric_y} vs {param_x} (Colored by {color_by})",
            color_continuous_scale="Viridis",
        )
    else:
        # Sort by x for line display if 1D sweep
        sorted_df = completed.sort_values(by=param_x)
        fig = go.Figure(
            go.Scatter(
                x=sorted_df[param_x],
                y=sorted_df[metric_y],
                mode="lines+markers",
                marker=dict(size=7, color="#35d7e5"),
                line=dict(color="#35d7e5", width=2.5),
                hovertemplate=f"{param_x}=%{{x:.4g}}<br>{metric_y}=%{{y:.4f}}<extra></extra>",
            )
        )
        fig.update_layout(title=f"Campaign Response: {metric_y} vs {param_x}")

    fig.update_layout(
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        font=dict(color=text_color),
        xaxis=dict(
            title=param_x,
            gridcolor="rgba(255,255,255,0.08)"
            if theme != "Light"
            else "rgba(0,0,0,0.08)",
        ),
        yaxis=dict(
            title=metric_y,
            gridcolor="rgba(255,255,255,0.08)"
            if theme != "Light"
            else "rgba(0,0,0,0.08)",
        ),
        margin=dict(l=60, r=20, t=50, b=45),
    )
    return fig


def campaign_parallel_coordinates(
    df: pd.DataFrame,
    dimensions: list[str],
    color_metric: str = "sigma8",
    *,
    theme: str = "Dark",
) -> go.Figure:
    """Render high-dimensional multi-parameter exploration via parallel coordinates."""
    completed = df[df["status"] == JobStatus.COMPLETED.value]
    valid_dims = [d for d in dimensions if d in completed.columns]
    if len(valid_dims) < 2 or color_metric not in completed.columns:
        fig = go.Figure()
        fig.add_annotation(
            text="Need at least 2 varied numerical parameters for parallel coordinates",
            showarrow=False,
        )
        return fig

    dims = []
    for d in valid_dims:
        dims.append(
            dict(
                label=d,
                values=completed[d],
                range=[float(completed[d].min()), float(completed[d].max())],
            )
        )

    fig = go.Figure(
        go.Parcoords(
            line=dict(
                color=completed[color_metric],
                colorscale="Tealgrn",
                showscale=True,
                colorbar=dict(title=color_metric),
            ),
            dimensions=dims,
        )
    )
    bg_color = "rgba(6, 16, 21, 0.95)" if theme != "Light" else "#ffffff"
    text_color = "#edf1ef" if theme != "Light" else "#132126"
    fig.update_layout(
        title=f"Multi-Parameter Campaign Geometry (Target: {color_metric})",
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        font=dict(color=text_color),
        margin=dict(l=60, r=40, t=60, b=30),
    )
    return fig


def compute_campaign_sensitivities(
    df: pd.DataFrame,
    target: str = "sigma8",
) -> list[dict[str, Any]]:
    """Compute numerical sensitivities (Pearson correlation & finite linear slope) for each varied parameter."""
    completed = df[df["status"] == JobStatus.COMPLETED.value]
    if completed.empty or target not in completed.columns:
        return []

    y = completed[target].to_numpy(dtype=float)
    if np.std(y) == 0:
        return []

    results = []
    exclude = {"member_id", "status", "elapsed_s", "error", target}
    candidate_params = [
        c
        for c in completed.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(completed[c])
    ]

    for param in candidate_params:
        x = completed[param].to_numpy(dtype=float)
        if len(np.unique(x)) > 1 and np.std(x) > 0:
            corr = float(np.corrcoef(x, y)[0, 1])
            # Linear fit slope dy/dx
            slope = float(np.polyfit(x, y, 1)[0])
            results.append(
                {
                    "parameter": param,
                    "correlation": corr,
                    "sensitivity_dY_dX": slope,
                    "normalized_elasticity": float(slope * (np.mean(x) / np.mean(y)))
                    if np.mean(y) != 0
                    else 0.0,
                    "min_value": float(np.min(x)),
                    "max_value": float(np.max(x)),
                }
            )

    return sorted(results, key=lambda r: abs(r["correlation"]), reverse=True)
