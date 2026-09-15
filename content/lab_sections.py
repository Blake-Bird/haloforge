"""Portable, local-only lab-section contracts for instructors.

These files let an instructor distribute a fixed baseline and a deliberately
small parameter surface without claiming accounts, access control, answer
collection, or any other hosted-classroom behavior.
"""

from __future__ import annotations

import json
from hashlib import sha256
from io import BytesIO
from zipfile import ZIP_DEFLATED, ZipFile

from config.defaults import DEFAULT_PARAMS
from config.ranges import CONTROL_RANGES
from content.modules import (
    LabModule,
    assignment_brief,
    instructor_guide,
    solution_guide,
    student_handout,
)


SECTION_SCHEMA_VERSION = "haloforge-local-lab-section-v1"


def _validated_permitted_parameters(permitted_parameters) -> tuple[str, ...]:
    permitted = tuple(dict.fromkeys(str(key) for key in permitted_parameters))
    if not permitted:
        raise ValueError("A local lab section needs at least one permitted parameter")
    unknown = [key for key in permitted if key not in CONTROL_RANGES]
    if unknown:
        raise ValueError("Unknown permitted parameter(s): " + ", ".join(unknown))
    return permitted


def local_lab_section(
    module: LabModule, baseline_params: dict | None, permitted_parameters
) -> dict:
    """Create a self-contained, student-facing local section specification."""
    permitted = _validated_permitted_parameters(permitted_parameters)
    baseline = dict(DEFAULT_PARAMS)
    if baseline_params:
        baseline.update(baseline_params)
    ranges = {
        key: {
            field: value
            for field, value in CONTROL_RANGES[key].items()
            if field in {"min", "max", "step", "format"}
        }
        for key in permitted
    }
    return {
        "schema_version": SECTION_SCHEMA_VERSION,
        "module": {
            "identifier": module.identifier,
            "title": module.title,
            "estimated_completion_minutes": module.duration_minutes,
        },
        "baseline_parameters": baseline,
        "permitted_parameters": list(permitted),
        "permitted_parameter_ranges": ranges,
        "question_sequence": [
            {"step": 1, "kind": "question", "prompt": module.question},
            {"step": 2, "kind": "prediction", "prompt": module.checkpoint},
            {"step": 3, "kind": "investigation", "prompt": module.student_prompt},
            {
                "step": 4,
                "kind": "conclusion",
                "prompt": "State the strongest supported conclusion, a caveat, and the next evidence step.",
            },
        ],
        "scientific_boundary": module.caution,
        "delivery_scope": {
            "private_by_default": True,
            "student_data_collection": False,
            "grade_export": False,
            "access_control": False,
            "note": "This is a portable local section specification. Separate instructor materials operationally; it cannot enforce hidden solutions or permissions.",
        },
    }


def _zip(files: dict[str, str]) -> bytes:
    manifest = {
        "schema_version": SECTION_SCHEMA_VERSION,
        "files": {
            name: {
                "sha256": sha256(content.encode("utf-8")).hexdigest(),
                "bytes": len(content.encode("utf-8")),
            }
            for name, content in files.items()
        },
    }
    files = {
        **files,
        "manifest.json": json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    }
    output = BytesIO()
    with ZipFile(output, "w", ZIP_DEFLATED) as archive:
        for name, content in files.items():
            archive.writestr(name, content)
    return output.getvalue()


def student_section_bundle(section: dict, module: LabModule) -> bytes:
    """Create a student-distribution ZIP without instructor solution content."""
    files = {
        "README.md": "\n".join(
            [
                f"# {section['module']['title']} — local lab section",
                "",
                "This package fixes a baseline and lists the only parameters intended for the investigation. It does not create accounts, collect answers, enforce permissions, or export grades.",
                "",
                f"**Estimated completion time:** {section['module']['estimated_completion_minutes']} minutes",
                "",
                "Use `section.json` as the source of truth for the baseline and permitted ranges. Record a prediction before calculating, then include a caveat with your conclusion.",
            ]
        )
        + "\n",
        "section.json": json.dumps(section, indent=2, sort_keys=True) + "\n",
        "student_handout.md": student_handout(module),
        "assignment.md": assignment_brief(module),
    }
    return _zip(files)


def instructor_section_bundle(section: dict, module: LabModule) -> bytes:
    """Create instructor-only companion materials; distribution remains manual."""
    files = {
        "README.md": "\n".join(
            [
                f"# Instructor companion — {section['module']['title']}",
                "",
                "Keep this companion separate from the student ZIP. HaloForge has no authentication or access control, so it cannot technically hide these materials from someone who receives this archive.",
                "",
                "The section specification records the locked baseline, permitted parameters, question sequence, estimated time, and scientific boundary.",
            ]
        )
        + "\n",
        "section.json": json.dumps(section, indent=2, sort_keys=True) + "\n",
        "instructor_guide.md": instructor_guide(module),
        "solution_guide.md": solution_guide(module),
    }
    return _zip(files)
