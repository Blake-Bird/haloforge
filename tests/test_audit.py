from state.audit import append_audit_event, audit_status
from state import run_storage


def _storage_paths(tmp_path, monkeypatch):
    for key, path in {
        "DATA_ROOT": tmp_path,
        "RUN_DIR": tmp_path / "saved_runs",
        "EXPORT_DIR": tmp_path / "exports",
        "STATE_DIR": tmp_path / "state",
    }.items():
        monkeypatch.setattr(run_storage, key, path)


def test_new_saved_run_has_a_verified_local_creation_event(tmp_path, monkeypatch):
    _storage_paths(tmp_path, monkeypatch)
    run_storage.save_run({"run_id": "audit", "name": "Audit", "params": {}})
    loaded = run_storage.load_run("audit")
    assert loaded["audit_trail"][0]["action"] == "run_created"
    assert audit_status(loaded)["state"] == "verified"


def test_metadata_and_name_mutations_append_minimal_events(tmp_path, monkeypatch):
    _storage_paths(tmp_path, monkeypatch)
    monkeypatch.setattr(run_storage, "generate_run_exports", lambda run: {})
    run_storage.save_run(
        {"run_id": "audit", "name": "Audit", "params": {}, "notebook": {}}
    )
    run = run_storage.load_run("audit")
    run["notebook"] = {"research_question": "Does this change?"}
    run_storage.update_run_metadata(run)
    renamed = run_storage.rename_run("audit", "Renamed")
    actions = [event["action"] for event in renamed["audit_trail"]]
    assert actions == ["run_created", "metadata_updated", "renamed"]
    metadata = renamed["audit_trail"][1]
    assert metadata["changed_fields"] == ["notebook"]
    assert "Does this change?" not in str(metadata)
    assert audit_status(renamed)["state"] == "verified"


def test_local_audit_chain_detects_tampering():
    run = {"audit_trail": []}
    append_audit_event(run, "run_created")
    run["audit_trail"][0]["action"] = "rewritten"
    assert audit_status(run)["state"] == "invalid"


def test_existing_legacy_run_is_not_relabelled_as_historically_audited(
    tmp_path, monkeypatch
):
    _storage_paths(tmp_path, monkeypatch)
    run_storage.RUN_DIR.mkdir(parents=True)
    (run_storage.RUN_DIR / "legacy.json").write_text(
        '{"run_id": "legacy", "name": "Legacy"}'
    )
    loaded = run_storage.load_run("legacy")
    assert loaded["audit_trail"] == []
    assert audit_status(loaded)["state"] == "unavailable"
    run_storage.save_run(loaded)
    assert (
        run_storage.load_run("legacy")["audit_trail"][0]["action"]
        == "audit_initialized"
    )
