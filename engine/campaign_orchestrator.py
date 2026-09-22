"""Campaign execution orchestrator: concurrent parameter sweeps with state tracking.

Provides queue management, non-oversubscribing worker pools, pause/resume/cancel,
sensitivity extraction, multi-dimensional response visualizers, and export bundles.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed
from time import perf_counter
from typing import Any, Callable
from uuid import uuid4

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

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
    evidence: dict[str, Any] | None = None


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
    cid = (
        f"campaign_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_"
        f"{uuid4().hex[:10]}"
    )
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
    record_metadata = dict(metadata or {})
    record_metadata.setdefault("lifecycle_state", "pending")
    return CampaignRecord(
        campaign_id=cid,
        name=name.strip() or cid,
        created_at=timestamp,
        design_type=design_type,
        members=members,
        metadata=record_metadata,
    )


def campaign_counts(campaign: CampaignRecord) -> dict[str, int]:
    """Return stable per-state counts for a persisted campaign queue."""
    return {
        status.value: sum(member.status == status for member in campaign.members)
        for status in JobStatus
    }


def pause_campaign(campaign: CampaignRecord) -> None:
    """Pause future batches; in-flight solver processes are never killed silently."""
    if any(member.status == JobStatus.RUNNING for member in campaign.members):
        raise RuntimeError("Campaign has running members and cannot be paused safely")
    campaign.metadata["lifecycle_state"] = "paused"


def resume_campaign(campaign: CampaignRecord) -> None:
    """Make a paused campaign eligible for its next explicit batch."""
    if campaign.metadata.get("lifecycle_state") == "cancelled":
        raise RuntimeError("Cancelled campaigns cannot be resumed; retry selected members instead")
    campaign.metadata["lifecycle_state"] = "pending"


def cancel_pending_members(campaign: CampaignRecord) -> int:
    """Cancel only unstarted members, preserving all completed evidence."""
    cancelled = 0
    for member in campaign.members:
        if member.status == JobStatus.PENDING:
            member.status = JobStatus.CANCELLED
            cancelled += 1
    campaign.metadata["lifecycle_state"] = "cancelled"
    return cancelled


def retry_failed_members(campaign: CampaignRecord) -> int:
    """Requeue failed members without overwriting their original parameters."""
    retried = 0
    for member in campaign.members:
        if member.status == JobStatus.FAILED:
            member.status = JobStatus.PENDING
            member.error_message = None
            retried += 1
    if retried:
        campaign.metadata["lifecycle_state"] = "pending"
    return retried


def execute_campaign_step(
    campaign: CampaignRecord,
    evaluator: Callable[[dict[str, Any]], dict[str, Any]],
    *,
    max_steps: int = 1,
    worker_count: int = 1,
    on_member_complete: Callable[[CampaignRecord], None] | None = None,
) -> bool:
    """Execute a bounded batch of pending campaign members.

    ``worker_count`` is intentionally explicit: callers must surface it to the
    researcher and persist it in campaign metadata.  The evaluator may launch
    an isolated external solver, so threads coordinate jobs but do not claim
    that a solver itself is thread-safe.

    Returns True when at least one member was attempted, False when there is
    no pending work.
    """
    if not isinstance(max_steps, int) or max_steps < 1:
        raise ValueError("max_steps must be a positive integer")
    if not isinstance(worker_count, int) or worker_count < 1:
        raise ValueError("worker_count must be a positive integer")
    if campaign.metadata.get("lifecycle_state") in {"paused", "cancelled"}:
        return False

    pending = [m for m in campaign.members if m.status == JobStatus.PENDING]
    batch = pending[:max_steps]
    if not batch:
        return False

    for member in batch:
        member.status = JobStatus.RUNNING
        member.error_message = None
    campaign.metadata["lifecycle_state"] = "running"

    def evaluate(member: CampaignMemberResult) -> tuple[CampaignMemberResult, dict[str, Any], float]:
        started = perf_counter()
        # ``member.params`` is a controlled delta. The caller owns the
        # immutable campaign baseline and must merge it exactly once. Adding
        # global defaults here previously overwrote a saved baseline's EDE and
        # numerical settings without showing that change in the member diff.
        result = evaluator(dict(member.params))
        return member, result, perf_counter() - started

    with ThreadPoolExecutor(max_workers=min(worker_count, len(batch))) as pool:
        futures = {pool.submit(evaluate, member): member for member in batch}
        for future in as_completed(futures):
            member = futures[future]
            try:
                _member, result, measured_elapsed = future.result()
                member.sigma8 = float(result["sigma8"])
                member.Omega_m = float(result["Omega_m"])
                member.h = float(result["h"])
                member.elapsed_seconds = float(result.get("elapsed_seconds", measured_elapsed))
                member.evidence = {
                    key: value
                    for key, value in result.items()
                    if key
                    not in {"sigma8", "Omega_m", "h", "elapsed_seconds"}
                }
                member.status = JobStatus.COMPLETED
            except Exception as exc:
                member.status = JobStatus.FAILED
                member.error_message = str(exc)
            if on_member_complete is not None:
                on_member_complete(campaign)
    if not any(member.status == JobStatus.PENDING for member in campaign.members):
        campaign.metadata["lifecycle_state"] = "completed"
    else:
        campaign.metadata["lifecycle_state"] = "pending"
    if on_member_complete is not None:
        on_member_complete(campaign)
    return True


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
            "backend": (m.evidence or {}).get("class_status", ""),
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
