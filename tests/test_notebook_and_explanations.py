from config.defaults import DEFAULT_PARAMS
from engine.explanations import explain_change
from state.notebook import (
    annotations_for_chart,
    attached_artifact_rows,
    lineage_rows,
    normalize_notebook_entry,
    parameter_diff,
)
from engine.assurance import assurance_report
from engine.figure_transcript import chart_transcript
import plotly.graph_objects as go


def test_notebook_normalization_and_parameter_diff():
    entry = normalize_notebook_entry(
        {
            "research_question": " What changes? ",
            "caveats": ["", "fit domain"],
            "parent_run_id": 4,
            "follow_up_run_ids": ["convergence", "convergence", 8, ""],
        }
    )
    assert entry["research_question"] == " What changes? "
    assert entry["caveats"] == ["fit domain"]
    assert entry["parent_run_id"] == "4"
    assert entry["follow_up_run_ids"] == ["convergence", "8"]
    candidate = {**DEFAULT_PARAMS, "n_s": 0.99}
    assert parameter_diff(DEFAULT_PARAMS, candidate) == [
        {"parameter": "n_s", "before": 0.965, "after": 0.99}
    ]


def test_causal_explanation_marks_numerical_edits_as_numerical():
    candidate = {**DEFAULT_PARAMS, "n_s": 0.99, "k_max": 400.0}
    steps = explain_change(DEFAULT_PARAMS, candidate)
    assert [step.parameter for step in steps] == ["n_s", "k_max"]
    assert "numerical coverage" in steps[1].caveat


def test_lineage_handles_missing_parents():
    rows = lineage_rows(
        [
            {"run_id": "base", "name": "Baseline", "created_at": "1", "notebook": {}},
            {
                "run_id": "child",
                "name": "Child",
                "created_at": "2",
                "notebook": {"parent_run_id": "base"},
            },
            {
                "run_id": "orphan",
                "name": "Orphan",
                "created_at": "3",
                "notebook": {"parent_run_id": "gone"},
            },
        ]
    )
    assert rows[1]["parent_run_id"] == "base"
    assert rows[2]["parent_run_id"] is None


def test_lineage_keeps_available_follow_up_evidence_and_reports_missing_links():
    rows = lineage_rows(
        [
            {
                "run_id": "base",
                "name": "Baseline",
                "created_at": "1",
                "notebook": {"follow_up_run_ids": ["check", "gone", "base"]},
            },
            {
                "run_id": "check",
                "name": "Convergence check",
                "created_at": "2",
                "notebook": {},
            },
        ]
    )
    assert rows[0]["follow_up_run_ids"] == ["check"]
    assert rows[0]["missing_follow_up_run_ids"] == ["gone"]


def test_chart_annotations_require_a_valid_matching_region():
    annotations = annotations_for_chart(
        {
            "annotations": [
                {
                    "chart_title": "Mass variance",
                    "x_start": "1e10",
                    "x_end": "1e12",
                    "text": "inspect tail",
                },
                {
                    "chart_title": "Mass variance",
                    "x_start": 3,
                    "x_end": 2,
                    "text": "bad",
                },
            ]
        },
        "Mass variance",
    )
    assert len(annotations) == 1
    assert annotations[0]["x_start"] == 1e10


def test_attached_artifacts_distinguish_present_and_missing_local_exports(tmp_path):
    attached = tmp_path / "bundle.zip"
    attached.write_bytes(b"bundle")
    rows = attached_artifact_rows(
        {
            "exports": {
                "exports.zip": str(attached),
                "summary": str(tmp_path / "missing.md"),
            }
        }
    )
    assert rows == [
        {
            "artifact": "exports.zip",
            "filename": "bundle.zip",
            "local state": "attached",
            "bytes": 6,
        },
        {
            "artifact": "summary",
            "filename": "missing.md",
            "local state": "missing locally",
            "bytes": None,
        },
    ]


def test_assurance_does_not_promote_endpoint_check_to_publication_ready():
    run = {
        "class_status": "AXICLASS",
        "arrays": {"k": [0.1]},
        "params": {"fitting": "Sheth-Tormen 2001"},
        "numerical_diagnostics": {
            "coverage": {"status": "range_looks_adequate"},
            "high_k_truncation": {"status": "low_sensitivity"},
            "low_k_truncation": {"status": "low_sensitivity"},
        },
    }
    report = assurance_report(run)
    assert report[0]["state"] == "pass"
    assert report[1]["state"] == "review"
    assert report[-1]["state"] == "not established"


def test_chart_transcript_preserves_series_and_samples():
    chart = go.Figure(go.Scatter(x=[1, 2], y=[3, 4], name="matter power"))
    transcript = chart_transcript(chart)
    assert transcript.to_dict("records") == [
        {"series": "matter power", "x": 1, "y": 3},
        {"series": "matter power", "x": 2, "y": 4},
    ]
