import json
from hashlib import sha256
from io import BytesIO
from zipfile import ZipFile

import pytest

from content.lab_sections import (
    SECTION_SCHEMA_VERSION,
    instructor_section_bundle,
    local_lab_section,
    student_section_bundle,
)
from content.modules import (
    GUIDED_EXPERIMENT_BY_MODULE,
    MODULES,
    guided_experiment_for_module,
    get_module,
    instructor_guide,
    lecture_outline,
    lecture_slides,
    notebook_template,
    student_handout,
    teaching_bundle,
)
from content.skepticism import EXERCISES, get_exercise


def test_every_module_has_reusable_student_and_instructor_materials():
    assert len(MODULES) >= 5
    assert {module.level for module in MODULES} >= {
        "Introductory cosmology",
        "Computational physics",
        "Statistics",
        "Structure formation",
        "Numerical methods",
    }
    for module in MODULES:
        assert module.question in student_handout(module)
        assert "Privacy" in instructor_guide(module)
        assert len(lecture_slides(module)) == 6
        assert module.question in lecture_outline(module)
        notebook = json.loads(notebook_template(module))
        assert notebook["nbformat"] == 4
        assert (
            notebook["metadata"]["haloforge_module"]["identifier"] == module.identifier
        )


def test_module_lookup_is_explicit():
    assert get_module("ede-structure").duration_minutes == 50


def test_every_prepared_module_has_a_matching_live_guided_experiment():
    assert set(GUIDED_EXPERIMENT_BY_MODULE) == {module.identifier for module in MODULES}
    assert {
        guided_experiment_for_module(module) for module in MODULES
    } <= {"Early expansion", "More small-scale power", "Why kmax matters"}


def test_teach_this_tomorrow_bundle_is_complete_and_local_only():
    with ZipFile(BytesIO(teaching_bundle(MODULES[0]))) as archive:
        names = set(archive.namelist())
        assert {
            "README.md",
            "student_handout.md",
            "assignment.md",
            "instructor_guide.md",
            "solution_guide.md",
            "lecture_outline.md",
            "lecture_slides.json",
            "notebook.ipynb",
            "accessibility.md",
            "manifest.json",
        } <= names
        readme = archive.read("README.md").decode("utf-8")
        assert "does not create a classroom" in readme
        manifest = json.loads(archive.read("manifest.json"))
        handout = archive.read("student_handout.md")
        assert (
            manifest["files"]["student_handout.md"]["sha256"]
            == sha256(handout).hexdigest()
        )
        assert "no student answers" in archive.read("accessibility.md").decode("utf-8")
        slides = json.loads(archive.read("lecture_slides.json"))
        assert slides[0]["body"] == MODULES[0].question


def test_skepticism_exercises_identify_a_single_specific_overclaim():
    assert len(EXERCISES) >= 3
    for exercise in EXERCISES:
        assert exercise.choices[exercise.unjustified_choice]
        assert (
            "not" in exercise.explanation.lower()
            or "but" in exercise.explanation.lower()
        )
        assert get_exercise(exercise.identifier) == exercise


def test_local_lab_section_keeps_student_and_instructor_materials_separate():
    module = MODULES[0]
    section = local_lab_section(module, {"n_s": 0.97}, ["n_s", "A_s"])
    assert section["schema_version"] == SECTION_SCHEMA_VERSION
    assert section["baseline_parameters"]["n_s"] == 0.97
    assert section["permitted_parameters"] == ["n_s", "A_s"]
    assert section["delivery_scope"]["access_control"] is False
    with ZipFile(BytesIO(student_section_bundle(section, module))) as student:
        assert {
            "section.json",
            "student_handout.md",
            "assignment.md",
            "manifest.json",
        } <= set(student.namelist())
        assert "solution_guide.md" not in student.namelist()
    with ZipFile(BytesIO(instructor_section_bundle(section, module))) as instructor:
        assert "solution_guide.md" in instructor.namelist()


def test_local_lab_section_rejects_unknown_or_empty_permitted_parameters():
    with pytest.raises(ValueError, match="at least one"):
        local_lab_section(MODULES[0], None, [])
    with pytest.raises(ValueError, match="Unknown"):
        local_lab_section(MODULES[0], None, ["not-a-control"])
