import json

import numpy as np
import pytest

from state import run_storage
from state.schema import (
    RUN_STORAGE_SCHEMA_VERSION,
    RunSchemaError,
    migrate_run_document,
)


def test_unversioned_run_migration_is_additive_and_recoverable():
    legacy = {
        "run_id": "legacy",
        "name": "Old run",
        "custom_legacy_key": {"keep": True},
    }
    migrated = migrate_run_document(legacy)
    assert migrated["storage_schema_version"] == RUN_STORAGE_SCHEMA_VERSION
    assert migrated["run_name"] == "Old run"
    assert migrated["custom_legacy_key"] == {"keep": True}
    assert (
        migrated["migration_history"][0]["strategy"]
        == "non-destructive additive migration"
    )
    assert "storage_schema_version" not in legacy


def test_unknown_future_schema_fails_closed():
    with pytest.raises(RunSchemaError, match="newer or unsupported"):
        migrate_run_document({"storage_schema_version": "haloforge-run-storage-v999"})


def test_loading_legacy_document_migrates_then_persists_on_save(tmp_path, monkeypatch):
    for key, path in {
        "DATA_ROOT": tmp_path,
        "RUN_DIR": tmp_path / "saved_runs",
        "EXPORT_DIR": tmp_path / "exports",
        "STATE_DIR": tmp_path / "state",
    }.items():
        monkeypatch.setattr(run_storage, key, path)
    run_storage.RUN_DIR.mkdir(parents=True)
    legacy = {
        "run_id": "legacy",
        "name": "Old run",
        "arrays_file": "legacy.npz",
        "params": {},
        "custom": "retained",
    }
    (run_storage.RUN_DIR / "legacy.json").write_text(json.dumps(legacy))
    with (run_storage.RUN_DIR / "legacy.npz").open("wb") as handle:
        np.savez_compressed(handle, k=np.array([0.1]), P=np.array([1.0]))
    loaded = run_storage.load_run("legacy")
    assert loaded["migration_status"] == "migrated_in_memory"
    assert loaded["custom"] == "retained"
    run_storage.save_run(loaded)
    persisted = json.loads((run_storage.RUN_DIR / "legacy.json").read_text())
    assert persisted["migration_status"] == "current"
    assert persisted["storage_schema_version"] == RUN_STORAGE_SCHEMA_VERSION
