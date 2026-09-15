from content.concepts import CONCEPTS, concept_by_label


def test_every_required_core_concept_has_all_four_zoom_layers():
    labels = {concept.label for concept in CONCEPTS}
    assert {
        "Primordial amplitude Aₛ",
        "Primordial tilt nₛ",
        "Transfer function",
        "Mass variance σ(M)",
        "σ₈",
        "Top-hat filter",
        "Early dark energy",
        "Halo mass definition",
    } <= labels
    for concept in CONCEPTS:
        assert all(
            isinstance(layer, str) and len(layer) > 40
            for layer in (
                concept.intuition,
                concept.course,
                concept.research,
                concept.implementation,
            )
        )


def test_concept_lookup_is_stable_and_unknown_concepts_fail_closed():
    assert concept_by_label("σ₈").course.startswith("It is σ(R=8")
    try:
        concept_by_label("not a HaloForge concept")
    except KeyError:
        pass
    else:
        raise AssertionError(
            "Unknown concepts must not receive a generic, ungrounded explanation."
        )
