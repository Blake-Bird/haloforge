"""Structured experiment-notebook metadata, independent of the UI framework."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


NOTEBOOK_SCHEMA_VERSION = "haloforge-notebook-v2"


def empty_notebook_entry() -> dict:
    return {
        "schema_version": NOTEBOOK_SCHEMA_VERSION,
        "research_question": "",
        "hypothesis": "",
        "prediction": "",
        "conclusion": "",
        "caveats": [],
        "citations": [],
        "annotations": [],
        "parent_run_id": None,
        "follow_up_run_ids": [],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def normalize_notebook_entry(value: dict | None) -> dict:
    entry = empty_notebook_entry()
    if isinstance(value, dict):
        entry.update(deepcopy(value))
    entry["caveats"] = [
        str(item).strip() for item in entry.get("caveats", []) if str(item).strip()
    ]
    entry["citations"] = [
        str(item).strip() for item in entry.get("citations", []) if str(item).strip()
    ]
    entry["annotations"] = [
        item for item in entry.get("annotations", []) if isinstance(item, dict)
    ]
    entry["parent_run_id"] = (
        str(entry["parent_run_id"]) if entry.get("parent_run_id") else None
    )
    # Follow-up links are intentionally IDs rather than copied titles: the
    # linked run remains the source of truth even if it is renamed. Preserve
    # unknown IDs for imported bundles; known self-links are filtered when a
    # run graph is assembled.
    seen: set[str] = set()
    follow_ups = []
    for item in entry.get("follow_up_run_ids", []):
        run_id = str(item).strip()
        if run_id and run_id not in seen:
            seen.add(run_id)
            follow_ups.append(run_id)
    entry["follow_up_run_ids"] = follow_ups
    return entry


def parameter_diff(baseline: dict, candidate: dict) -> list[dict]:
    """Serializable changes only; array data never belongs in notebook metadata."""
    ignored = {"mode", "z_presets_selected", "custom_z_list", "z_values"}
    rows = []
    for key in sorted(set(baseline) | set(candidate)):
        if key in ignored or baseline.get(key) == candidate.get(key):
            continue
        rows.append(
            {"parameter": key, "before": baseline.get(key), "after": candidate.get(key)}
        )
    return rows


def lineage_rows(runs: list[dict]) -> list[dict]:
    """Return stable tree rows even when a parent was imported or later removed."""
    known = {str(run.get("run_id")) for run in runs}
    rows = []
    for run in sorted(runs, key=lambda item: item.get("created_at", "")):
        notebook = normalize_notebook_entry(run.get("notebook"))
        parent = notebook.get("parent_run_id")
        rows.append(
            {
                "run_id": str(run.get("run_id")),
                "name": run.get("name", "Untitled run"),
                "parent_run_id": parent if parent in known else None,
                "follow_up_run_ids": [
                    run_id
                    for run_id in notebook["follow_up_run_ids"]
                    if run_id in known and run_id != str(run.get("run_id"))
                ],
                "missing_follow_up_run_ids": [
                    run_id
                    for run_id in notebook["follow_up_run_ids"]
                    if run_id not in known
                ],
                "research_question": notebook["research_question"],
                "hypothesis": notebook["hypothesis"],
                "reproducibility_hash": run.get("reproducibility_hash", ""),
            }
        )
    return rows


def annotations_for_chart(entry: dict | None, chart_title: str) -> list[dict]:
    """Return only well-formed region annotations for one rendered chart."""
    annotations = normalize_notebook_entry(entry).get("annotations", [])
    selected = []
    for annotation in annotations:
        if annotation.get("chart_title") != chart_title:
            continue
        try:
            start, end = float(annotation["x_start"]), float(annotation["x_end"])
        except (KeyError, TypeError, ValueError):
            continue
        if (
            np.isfinite(start)
            and np.isfinite(end)
            and end > start
            and str(annotation.get("text", "")).strip()
        ):
            selected.append({**annotation, "x_start": start, "x_end": end})
    return selected


def attached_artifact_rows(run: dict) -> list[dict]:
    """Describe exports attached to one run without reading their contents.

    Export paths are derived from the saved run, so a missing local file stays
    visible as missing instead of being silently presented as an attachment.
    """
    exports = run.get("exports", {})
    if not isinstance(exports, dict):
        return []
    rows = []
    for artifact, raw_path in sorted(exports.items()):
        if not isinstance(raw_path, str) or not raw_path.strip():
            continue
        path = Path(raw_path)
        rows.append(
            {
                "artifact": str(artifact),
                "filename": path.name,
                "local state": "attached" if path.is_file() else "missing locally",
                "bytes": path.stat().st_size if path.is_file() else None,
            }
        )
    return rows
