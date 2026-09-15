import json

import pytest

from state.observability import (
    EVENTS_FILE,
    MAX_EVENTS,
    clear_local_diagnostics,
    local_diagnostics_enabled,
    local_diagnostics_summary,
    record_local_diagnostic,
    set_local_diagnostics_enabled,
)


def test_diagnostics_are_disabled_by_default_and_do_not_write_events(tmp_path):
    assert not local_diagnostics_enabled(tmp_path)
    assert not record_local_diagnostic(
        tmp_path, False, "calculation_completed", duration_seconds=1.2
    )
    assert not (tmp_path / "state" / EVENTS_FILE).exists()


def test_opted_in_diagnostics_are_minimal_bounded_and_aggregate_only(tmp_path):
    set_local_diagnostics_enabled(tmp_path, True)
    for index in range(MAX_EVENTS + 2):
        assert record_local_diagnostic(
            tmp_path, True, "calculation_completed", duration_seconds=index / 10
        )
    payload = json.loads((tmp_path / "state" / EVENTS_FILE).read_text())
    assert len(payload["events"]) == MAX_EVENTS
    assert set(payload["events"][-1]) <= {"event", "recorded_at", "duration_ms"}
    summary = local_diagnostics_summary(tmp_path)
    assert summary["enabled"]
    assert summary["counts"]["calculation_completed"] == MAX_EVENTS


def test_diagnostics_reject_unknown_events_and_can_be_cleared(tmp_path):
    with pytest.raises(ValueError, match="Unsupported"):
        record_local_diagnostic(tmp_path, True, "parameter_changed")
    record_local_diagnostic(tmp_path, True, "calculation_failed")
    clear_local_diagnostics(tmp_path)
    assert local_diagnostics_summary(tmp_path)["event_count"] == 0
