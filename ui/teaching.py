"""Teaching Lab workspace and student-controlled work export."""

from copy import deepcopy
from html import escape
from io import BytesIO
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import streamlit as st

from config.defaults import DEFAULT_PARAMS
from config.ranges import CONTROL_RANGES
from content.modules import (
    MODULES,
    get_module,
    guided_experiment_for_module,
    teaching_bundle,
    lecture_slides,
    lecture_outline,
    student_handout,
    instructor_guide,
    notebook_template,
)
from content.lab_sections import (
    local_lab_section,
    student_section_bundle,
    instructor_section_bundle,
)
from content.skepticism import EXERCISES, get_exercise
from state.run_storage import (
    load_all_runs,
    get_run_label,
    generate_run_exports,
)


def teach_view():
    st.markdown(
        '<div class="page-head"><span>TEACHING LAB</span><h2>Run a lab. Keep your evidence.</h2><p>Students explore the question, save their runs and notebook, and export their work when ready.</p></div>',
        unsafe_allow_html=True,
    )
    module_map = {module.identifier: module for module in MODULES}
    identifier = st.selectbox(
        "Prepared module",
        list(module_map),
        format_func=lambda key: module_map[key].title,
        key="teaching_module",
    )
    module = get_module(identifier)
    stats = st.columns(3)
    stats[0].metric("Estimated time", f"{module.duration_minutes} min")
    stats[1].metric("Level", module.level)
    stats[2].metric("Prerequisites", len(module.prerequisites))
    st.download_button(
        "Download teach-this-tomorrow bundle",
        teaching_bundle(module),
        f"{module.identifier}_teach_this_tomorrow.zip",
        "application/zip",
        help="Prepared local materials for running the lab.",
    )
    st.markdown("### Live question")
    st.markdown(f"**{module.question}**")
    target_experiment = guided_experiment_for_module(module)
    if st.button(
        f"Start this lab — {target_experiment}",
        type="primary",
        key=f"launch_teaching_module_{module.identifier}",
        help="Open the matching prediction-led guided experiment. No calculation starts until you choose a prediction and run it.",
    ):
        st.session_state["teaching_experiment_request"] = target_experiment
        st.session_state["teaching_launch_notice"] = module.title
        st.session_state["hf_command_navigation"] = {
            "primary_mode": "Explore",
            "workspace": None,
        }
        st.rerun()
    tabs = st.tabs(
        [
            "Lecture mode",
            "Local lab section",
            "Student lab",
            "Instructor guide",
            "Notebook",
            "Scientific skepticism",
            "Export your work",
        ]
    )
    with tabs[0]:
        slides = lecture_slides(module)
        projector = st.toggle(
            "Projector contrast",
            value=True,
            help="Uses a restrained, high-contrast large-type layout. No science values are changed.",
            key=f"projector_{module.identifier}",
        )
        slide_number = st.select_slider(
            "Lecture slide — focus this control and use Left/Right Arrow keys to move through the narrative.",
            options=list(range(len(slides))),
            value=0,
            format_func=lambda index: f"{index + 1} / {len(slides)} · {slides[index].title}",
            key=f"lecture_slide_{module.identifier}",
        )
        slide = slides[slide_number]
        projector_class = " projector" if projector else ""
        st.markdown(
            f'<section class="lecture-slide{projector_class}"><span>{escape(slide.kicker)}</span><h2>{escape(slide.title)}</h2><p>{escape(slide.body)}</p><div><b>Ask the room</b>{escape(slide.prompt)}</div></section>',
            unsafe_allow_html=True,
        )
        with st.expander("Speaker note"):
            st.write(slide.speaker_note)
        st.caption(
            "Keyboard control is provided by the focused slide selector. This mode is local-only and keeps scientific caveats in the lecture arc."
        )
        st.download_button(
            "Download lecture outline",
            lecture_outline(module),
            f"{module.identifier}_lecture_outline.md",
            "text/markdown",
        )
    with tabs[1]:
        runs = load_all_runs()
        section_sources = {"HaloForge defaults": deepcopy(DEFAULT_PARAMS)}
        section_sources.update({run["name"]: run["params"] for run in runs})
        source_name = st.selectbox(
            "Locked baseline",
            list(section_sources),
            key=f"section_baseline_{module.identifier}",
        )
        defaults_by_module = {
            "primordial-tilt": ["n_s"],
            "numerical-coverage": ["k_max", "k_points"],
            "ede-structure": ["f_EDE", "log10_a_c"],
            "rare-tail-statistics": ["A_s", "n_s"],
            "numerical-methods": ["k_max", "k_points"],
        }
        permitted = st.multiselect(
            "Parameters students may vary",
            list(CONTROL_RANGES),
            default=defaults_by_module.get(module.identifier, ["n_s"]),
            key=f"section_parameters_{module.identifier}",
        )
        try:
            section = local_lab_section(module, section_sources[source_name], permitted)
        except ValueError as exc:
            st.warning(str(exc))
        else:
            st.caption(
                f"One local section · {module.duration_minutes} estimated minutes · four evidence-first prompts"
            )
            columns = st.columns(2)
            columns[0].download_button(
                "Download student lab section",
                student_section_bundle(section, module),
                f"{module.identifier}_student_section.zip",
                "application/zip",
            )
            columns[1].download_button(
                "Download instructor companion",
                instructor_section_bundle(section, module),
                f"{module.identifier}_instructor_companion.zip",
                "application/zip",
            )
            st.caption("Student and instructor materials are separate local downloads.")
    with tabs[2]:
        st.markdown(student_handout(module))
        st.download_button(
            "Download student handout",
            student_handout(module),
            f"{module.identifier}_student_handout.md",
            "text/markdown",
        )
    with tabs[3]:
        st.markdown(instructor_guide(module))
        st.download_button(
            "Download instructor guide",
            instructor_guide(module),
            f"{module.identifier}_instructor_guide.md",
            "text/markdown",
        )
    with tabs[4]:
        st.caption(
            "Open this after exporting a run bundle; it loads the bundled Parquet tables locally."
        )
        st.download_button(
            "Download Jupyter notebook template",
            notebook_template(module),
            f"{module.identifier}.ipynb",
            "application/x-ipynb+json",
        )
    with tabs[5]:
        st.markdown("### A smooth curve can still be wrong")
        st.caption(
            "Use these exercises to test how far the available evidence supports a claim."
        )
        exercise_map = {exercise.identifier: exercise for exercise in EXERCISES}
        exercise_id = st.selectbox(
            "Scenario",
            list(exercise_map),
            format_func=lambda key: key.replace("-", " ").capitalize(),
            key="skepticism_exercise",
        )
        exercise = get_exercise(exercise_id)
        st.info(exercise.setup)
        answer = st.radio(
            exercise.prompt,
            list(range(len(exercise.choices))),
            format_func=lambda index: exercise.choices[index],
            key=f"skepticism_answer_{exercise_id}",
        )
        if st.button("Check reasoning", key=f"check_skepticism_{exercise_id}"):
            if answer == exercise.unjustified_choice:
                st.success(
                    "Correct: that conclusion goes beyond the available evidence."
                )
            else:
                st.warning(
                    "That statement may be supported, but it is not the overclaim in this scenario."
                )
            st.write(exercise.explanation)
            st.caption("Next evidence step: " + exercise.follow_up)
    with tabs[6]:
        st.caption(
            "Choose a saved run to take its data, notebook answers, provenance, and a reusable analysis notebook with you."
        )
        saved_runs = load_all_runs()
        if saved_runs:
            run_by_id = {run["run_id"]: run for run in saved_runs}
            selected_id = st.selectbox(
                "Saved lab run",
                list(run_by_id),
                format_func=lambda run_id: get_run_label(run_by_id[run_id]),
                key=f"student_export_run_{module.identifier}",
            )
            if st.button(
                "Prepare my lab export",
                key=f"prepare_student_export_{module.identifier}",
            ):
                try:
                    selected_run = run_by_id[selected_id]
                    # Refresh the export so recently edited notebook answers
                    # cannot be replaced by an older cached ZIP.
                    exports = generate_run_exports(selected_run)
                    export_path = Path(exports["exports.zip"])
                    out = BytesIO()
                    with ZipFile(out, "w", ZIP_DEFLATED) as archive:
                        archive.writestr("student_handout.md", student_handout(module))
                        archive.writestr(
                            "analysis_notebook.ipynb", notebook_template(module)
                        )
                        archive.writestr(
                            "lab_run_exports.zip", export_path.read_bytes()
                        )
                    st.session_state[f"student_export_{module.identifier}"] = (
                        selected_id,
                        out.getvalue(),
                    )
                except (OSError, ValueError) as exc:
                    st.error(f"The lab export could not be prepared: {exc}")
            prepared = st.session_state.get(f"student_export_{module.identifier}")
            if prepared and prepared[0] == selected_id:
                st.download_button(
                    "Download my lab work",
                    prepared[1],
                    f"{module.identifier}_{selected_id}_lab_work.zip",
                    "application/zip",
                    key=f"download_student_export_{module.identifier}",
                )
        else:
            st.info(
                "Run a lab and save it first; your data and notebook will then be available here."
            )
