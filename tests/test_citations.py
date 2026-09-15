from state.citations import export_citations


def test_export_citations_carries_figure_sources_and_user_notebook_references():
    record = export_citations(
        {
            "class_status": "AXICLASS",
            "params": {"fitting": "Tinker 2008"},
            "provenance": {"axiclass_commit": "abc"},
            "notebook": {"citations": ["A user-selected source"]},
        }
    )
    assert record["schema_version"]
    assert record["figures"]["matter_power.pdf"][0]["id"] == "stored-solver-provenance"
    assert any(
        entry["citation"] == "Tinker et al. 2008"
        for entry in record["figures"]["analytic_hmf_reference.pdf"]
    )
    assert record["user_notebook_citations"] == ["A user-selected source"]
