"""Atomic local persistence for reproducible CLASS/AxiCLASS campaigns."""

from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import asdict
from io import BytesIO, StringIO
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import csv

from engine.campaign_orchestrator import CampaignMemberResult, CampaignRecord, JobStatus
from state.storage_policy import default_data_root, require_safe_persistent_storage


CAMPAIGN_SCHEMA_VERSION = "haloforge-campaign-v1"
CAMPAIGN_DIR = default_data_root() / "campaigns"
_SAFE_ID = re.compile(r"campaign_[0-9]{8}_[0-9]{6}(?:_[A-Za-z0-9_-]+)?\Z")


def _path(campaign_id: str) -> Path:
    if not isinstance(campaign_id, str) or not _SAFE_ID.fullmatch(campaign_id):
        raise ValueError("Campaign identifier is not a safe vault token")
    return CAMPAIGN_DIR / f"{campaign_id}.json"


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temp = Path(temp_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def save_campaign(campaign: CampaignRecord) -> Path:
    """Persist a complete campaign record atomically in the local research vault."""
    require_safe_persistent_storage()
    path = _path(campaign.campaign_id)
    payload = {
        "schema_version": CAMPAIGN_SCHEMA_VERSION,
        **asdict(campaign),
    }
    _atomic_write(path, json.dumps(payload, sort_keys=True, indent=2, default=str))
    return path


def load_campaign(campaign_id: str) -> CampaignRecord:
    """Load and validate one stored campaign record."""
    require_safe_persistent_storage()
    path = _path(campaign_id)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError("Campaign record could not be read") from exc
    except json.JSONDecodeError as exc:
        raise ValueError("Campaign record is not valid JSON") from exc
    if payload.get("schema_version") != CAMPAIGN_SCHEMA_VERSION:
        raise ValueError("Campaign record schema is unsupported")
    members = []
    for item in payload.get("members", []):
        if not isinstance(item, dict):
            raise ValueError("Campaign member record is invalid")
        try:
            status = JobStatus(item["status"])
            # A browser/app restart cannot prove a previously running external
            # process completed. Requeue it explicitly rather than fabricating
            # a completion or permanently stranding the campaign.
            if status == JobStatus.RUNNING:
                status = JobStatus.PENDING
            members.append(
                CampaignMemberResult(
                    member_id=str(item["member_id"]),
                    params=dict(item["params"]),
                    status=status,
                    sigma8=item.get("sigma8"),
                    Omega_m=item.get("Omega_m"),
                    h=item.get("h"),
                    elapsed_seconds=float(item.get("elapsed_seconds", 0.0)),
                    error_message=item.get("error_message"),
                    evidence=dict(item["evidence"]) if item.get("evidence") else None,
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Campaign member record is invalid") from exc
    if not members:
        raise ValueError("Campaign record has no members")
    metadata = dict(payload.get("metadata", {}))
    if any(item.get("status") == JobStatus.RUNNING.value for item in payload.get("members", [])):
        metadata["recovery_note"] = (
            "Members recorded as running during the previous session were requeued after restart; verify external solver logs before interpreting results."
        )
        metadata["lifecycle_state"] = "pending"
    return CampaignRecord(
        campaign_id=str(payload["campaign_id"]),
        name=str(payload["name"]),
        created_at=str(payload["created_at"]),
        design_type=str(payload["design_type"]),
        members=members,
        metadata=metadata,
    )


def list_campaign_ids() -> list[str]:
    """Return campaign identifiers in newest-file-first order."""
    require_safe_persistent_storage()
    if not CAMPAIGN_DIR.exists():
        return []
    return [
        path.stem
        for path in sorted(CAMPAIGN_DIR.glob("campaign_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        if _SAFE_ID.fullmatch(path.stem)
    ]


def completed_member_timings() -> list[float]:
    """Return positive elapsed solver times from readable local campaigns."""
    timings: list[float] = []
    for campaign_id in list_campaign_ids():
        try:
            campaign = load_campaign(campaign_id)
        except ValueError:
            continue
        timings.extend(
            float(member.elapsed_seconds)
            for member in campaign.members
            if member.status == JobStatus.COMPLETED and member.elapsed_seconds > 0
        )
    return timings


def campaign_export_bundle(campaign: CampaignRecord) -> bytes:
    """Return a self-contained reproducibility ZIP without mutating the vault.

    The archive contains a canonical JSON record, a flat member table suitable
    for R/Python, one evidence document per completed member, and a concise
    readme stating the scope of the solver results.
    """
    manifest = {
        "schema_version": CAMPAIGN_SCHEMA_VERSION,
        **asdict(campaign),
    }
    csv_buffer = StringIO(newline="")
    fields = [
        "member_id",
        "status",
        "sigma8",
        "Omega_m",
        "h",
        "elapsed_seconds",
        "error_message",
        "parameters_json",
        "evidence_json",
    ]
    writer = csv.DictWriter(csv_buffer, fieldnames=fields)
    writer.writeheader()
    for member in campaign.members:
        writer.writerow(
            {
                "member_id": member.member_id,
                "status": member.status.value,
                "sigma8": member.sigma8,
                "Omega_m": member.Omega_m,
                "h": member.h,
                "elapsed_seconds": member.elapsed_seconds,
                "error_message": member.error_message or "",
                "parameters_json": json.dumps(member.params, sort_keys=True),
                "evidence_json": json.dumps(member.evidence or {}, sort_keys=True),
            }
        )
    readme = (
        "HaloForge campaign reproducibility bundle\n\n"
        "campaign.json is the canonical immutable queue record. members.csv is a flat analysis table. "
        "evidence/ contains per-member CLASS/AxiCLASS settings and execution metadata when available. "
        "These are calculated linear-theory solver results; they are not nonlinear N-body simulation measurements.\n"
    )
    output = BytesIO()
    with ZipFile(output, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("campaign.json", json.dumps(manifest, sort_keys=True, indent=2, default=str))
        archive.writestr("members.csv", csv_buffer.getvalue())
        archive.writestr("README.txt", readme)
        for member in campaign.members:
            if member.evidence:
                archive.writestr(
                    f"evidence/{member.member_id}.json",
                    json.dumps(member.evidence, sort_keys=True, indent=2, default=str),
                )
    return output.getvalue()
