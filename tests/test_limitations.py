from pathlib import Path

from content.limitations import LIMITATIONS, LIMITATIONS_VERSION, limitations_rows


def test_known_limitations_are_versioned_and_actionable():
    assert LIMITATIONS_VERSION.endswith("v1")
    rows = limitations_rows()
    assert len(rows) == len(LIMITATIONS)
    assert all(row["status"] == "open" and row["mitigation"] for row in rows)


def test_published_limitations_table_exposes_the_registry_impact_dimension():
    document = (
        Path(__file__).resolve().parents[1] / "docs" / "KNOWN-LIMITATIONS.md"
    ).read_text(encoding="utf-8")

    assert "| Impact |" in document
    for limitation in LIMITATIONS:
        assert limitation.identifier in document
