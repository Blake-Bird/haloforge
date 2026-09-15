"""Strictly local, opt-in operational diagnostics for HaloForge."""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path


OBSERVABILITY_SCHEMA_VERSION = "haloforge-local-observability-v1"
PREFERENCES_FILE = "observability_preferences.json"
EVENTS_FILE = "local_diagnostics.json"
MAX_EVENTS = 200
ALLOWED_EVENTS = frozenset(
    {
        "calculation_completed",
        "calculation_failed",
        "benchmark_completed",
        "export_completed",
    }
)


def _state_dir(data_root: Path) -> Path:
    return Path(data_root) / "state"


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temp = Path(temp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def local_diagnostics_enabled(data_root: Path) -> bool:
    """Return false unless the user has explicitly opted in on this device."""
    try:
        payload = json.loads(
            (_state_dir(data_root) / PREFERENCES_FILE).read_text(encoding="utf-8")
        )
        return bool(payload.get("enabled", False))
    except (OSError, ValueError, TypeError):
        return False


def set_local_diagnostics_enabled(data_root: Path, enabled: bool) -> None:
    """Persist only the local opt-in preference; never transmit it."""
    _atomic_json(
        _state_dir(data_root) / PREFERENCES_FILE,
        {
            "schema_version": OBSERVABILITY_SCHEMA_VERSION,
            "enabled": bool(enabled),
            "scope": "Local operational event categories and optional durations only. No network transmission, research parameters, run identifiers, IP addresses, or notes.",
        },
    )


def _read_events(data_root: Path) -> list[dict]:
    try:
        payload = json.loads(
            (_state_dir(data_root) / EVENTS_FILE).read_text(encoding="utf-8")
        )
        events = payload.get("events", [])
        return (
            [event for event in events if isinstance(event, dict)]
            if isinstance(events, list)
            else []
        )
    except (OSError, ValueError, TypeError):
        return []


def record_local_diagnostic(
    data_root: Path, enabled: bool, event: str, *, duration_seconds: float | None = None
) -> bool:
    """Record a deliberately minimal local diagnostic only after opt-in."""
    if not enabled:
        return False
    if event not in ALLOWED_EVENTS:
        raise ValueError(f"Unsupported local diagnostic event: {event}")
    entry = {"event": event, "recorded_at": datetime.now(timezone.utc).isoformat()}
    if duration_seconds is not None:
        duration = float(duration_seconds)
        if duration < 0 or duration != duration or duration == float("inf"):
            raise ValueError("Diagnostic duration must be finite and non-negative")
        entry["duration_ms"] = round(duration * 1000, 1)
    events = (_read_events(data_root) + [entry])[-MAX_EVENTS:]
    _atomic_json(
        _state_dir(data_root) / EVENTS_FILE,
        {
            "schema_version": OBSERVABILITY_SCHEMA_VERSION,
            "events": events,
            "scope": "Local-only operational categories and durations; never research inputs, run IDs, notes, IP addresses, or network telemetry.",
        },
    )
    return True


def local_diagnostics_summary(data_root: Path) -> dict:
    """Return aggregate-only local evidence suitable for a settings panel."""
    events = _read_events(data_root)
    counts = {event: 0 for event in sorted(ALLOWED_EVENTS)}
    durations = []
    for item in events:
        event = item.get("event")
        if event in counts:
            counts[event] += 1
        if isinstance(item.get("duration_ms"), (int, float)):
            durations.append(float(item["duration_ms"]))
    return {
        "enabled": local_diagnostics_enabled(data_root),
        "event_count": len(events),
        "counts": counts,
        "mean_recorded_duration_ms": round(sum(durations) / len(durations), 1)
        if durations
        else None,
        "scope": "Local-only aggregate diagnostics. No network transmission or scientific/run identity data is recorded.",
    }


def clear_local_diagnostics(data_root: Path) -> None:
    """Delete recorded event history but preserve the user's opt-in preference."""
    (_state_dir(data_root) / EVENTS_FILE).unlink(missing_ok=True)
