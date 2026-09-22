from engine.navigation import command_index, search_commands


class Concept:
    label = "Mass variance σ(M)"


def test_index_covers_destinations_figures_equations_concepts_runs_and_exports():
    commands = command_index(
        concepts=[Concept()],
        runs=[{"run_id": "run-1", "name": "EDE comparison", "exports": {"csv": "x"}}],
    )
    identifiers = {item["identifier"] for item in commands}
    assert {
        "explore",
        "figure:Halo mass function",
        "equation:Variance σ²(M)",
        "concept:Mass variance σ(M)",
        "run:run-1",
        "export:run-1",
    } <= identifiers


def test_search_is_case_insensitive_ranked_and_does_not_expose_arrays():
    commands = command_index(
        runs=[{"run_id": "run-1", "name": "Cluster baseline", "arrays": {"P": [1, 2]}}]
    )
    matches = search_commands("cluster", commands)
    assert matches[0]["identifier"] == "run:run-1"
    assert matches[0]["run_id"] == "run-1"
    assert "arrays" not in matches[0]


def test_search_handles_empty_query_and_rejects_bad_limit():
    commands = command_index()
    assert [item["identifier"] for item in search_commands("", commands, limit=3)] == [
        "explore",
        "compare",
        "evolution",
    ]
    try:
        search_commands("x", commands, limit=0)
    except ValueError as exc:
        assert "positive integer" in str(exc)
    else:
        raise AssertionError("expected invalid limit to be rejected")
