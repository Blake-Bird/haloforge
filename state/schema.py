"""Non-destructive schema evolution for persisted HaloForge run documents."""

from __future__ import annotations

from copy import deepcopy


RUN_STORAGE_SCHEMA_VERSION = "haloforge-run-storage-v1"


class RunSchemaError(ValueError):
    """Raised when a document is newer than this application understands."""


def migrate_run_document(document: dict) -> dict:
    """Return a current-format copy while preserving every legacy field.

    Migrations run in memory first. A later explicit save atomically persists
    the migrated copy; the original keys remain, so recovery never depends on
    reconstructing discarded information.
    """
    if not isinstance(document, dict):
        raise RunSchemaError("A saved run document must be a JSON object.")
    source = document.get("storage_schema_version")
    if source == RUN_STORAGE_SCHEMA_VERSION:
        return deepcopy(document)
    if source not in (None, ""):
        raise RunSchemaError(
            f"Saved run schema {source!r} is newer or unsupported by this HaloForge build."
        )
    migrated = deepcopy(document)
    # Historical documents used either `name` or `run_name`; retain both so a
    # rollback or external reader can recover the original representation.
    name = migrated.get("run_name", migrated.get("name", "Untitled run"))
    migrated.setdefault("name", name)
    migrated.setdefault("run_name", name)
    history = list(migrated.get("migration_history", []))
    history.append(
        {
            "from": "unversioned",
            "to": RUN_STORAGE_SCHEMA_VERSION,
            "strategy": "non-destructive additive migration",
            "recovery": "Original fields are retained; no arrays, parameters, provenance, or user notes are discarded.",
        }
    )
    migrated["migration_history"] = history
    migrated["storage_schema_version"] = RUN_STORAGE_SCHEMA_VERSION
    migrated["migration_status"] = "migrated_in_memory"
    return migrated
