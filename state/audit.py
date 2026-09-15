"""Local, app-level mutation history for one saved HaloForge run.

This is deliberately not a multi-user audit system: events have no actor,
server timestamp, authorization state, or cryptographic trust anchor. The
hash chain makes accidental corruption and in-app reordering visible, while a
person with filesystem access can still alter a local record.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone


AUDIT_SCHEMA_VERSION = "haloforge-local-run-audit-v1"


def _canonical(value: dict) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _event_hash(event: dict) -> str:
    unsigned = {key: value for key, value in event.items() if key != "event_hash"}
    return hashlib.sha256(_canonical(unsigned).encode("utf-8")).hexdigest()


def normalize_audit_trail(events: object) -> list[dict]:
    """Keep only structurally safe event records without silently repairing them."""
    if not isinstance(events, list):
        return []
    return [dict(event) for event in events if isinstance(event, dict)]


def append_audit_event(
    run: dict,
    action: str,
    changed_fields: list[str] | None = None,
    *,
    timestamp: str | None = None,
) -> dict:
    """Append a data-minimizing local event and return the new event.

    Only field names are recorded, never notebook text, parameter values, or
    other potentially sensitive user content.
    """
    trail = normalize_audit_trail(run.get("audit_trail"))
    event = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "sequence": len(trail) + 1,
        "recorded_at": timestamp or datetime.now(timezone.utc).isoformat(),
        "action": str(action),
        "changed_fields": sorted({str(field) for field in (changed_fields or [])}),
        "previous_event_hash": trail[-1].get("event_hash") if trail else None,
    }
    event["event_hash"] = _event_hash(event)
    trail.append(event)
    run["audit_trail"] = trail
    return event


def audit_status(run: dict) -> dict:
    """Verify chain structure; this detects changes but cannot authenticate a local file."""
    trail = normalize_audit_trail(run.get("audit_trail"))
    scope = "Local app-level history only; it does not identify people or secure a shared workspace, and cannot prevent filesystem edits."
    prior_hash = None
    for index, event in enumerate(trail, start=1):
        if (
            event.get("schema_version") != AUDIT_SCHEMA_VERSION
            or event.get("sequence") != index
        ):
            return {"state": "invalid", "events": len(trail), "scope": scope}
        if event.get("previous_event_hash") != prior_hash or event.get(
            "event_hash"
        ) != _event_hash(event):
            return {"state": "invalid", "events": len(trail), "scope": scope}
        prior_hash = event["event_hash"]
    return {
        "state": "verified" if trail else "unavailable",
        "events": len(trail),
        "scope": scope,
    }
