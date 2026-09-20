"""Regression checks for the source and container distribution boundary."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_container_includes_every_first_party_package_imported_by_the_app():
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")

    for package in ("assets", "config", "content", "engine", "state"):
        assert f"COPY {package} ./{package}" in dockerfile


def test_container_builds_the_pinned_axiclass_binding_without_pip_isolation():
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")

    assert "make class libclass.a -j2" in dockerfile
    assert "pip install --no-build-isolation ." in dockerfile


def test_container_includes_a_browser_runtime_for_promised_static_figure_exports():
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")

    assert "chromium" in dockerfile


def test_local_axiclass_setup_uses_the_same_explicit_binding_install_contract():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "make class libclass.a -j2" in readme
    assert "python -m pip install --no-build-isolation ." in readme


def test_local_research_artifacts_are_not_part_of_the_source_distribution():
    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    dockerignore = (ROOT / ".dockerignore").read_text(encoding="utf-8")

    assert "/data/" in gitignore
    assert "/output/" in gitignore
    assert "data" in dockerignore.splitlines()


def test_citation_metadata_does_not_claim_a_release_before_release_approval():
    citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    release_policy = (ROOT / "docs" / "RELEASE-POLICY.md").read_text(encoding="utf-8")

    assert "not release-approved" in release_policy
    assert "date-released:" not in citation


def test_github_contributor_templates_preserve_privacy_and_security_routing():
    templates = ROOT / ".github" / "ISSUE_TEMPLATE"
    config = (templates / "config.yml").read_text(encoding="utf-8")
    bug = (templates / "bug-report.yml").read_text(encoding="utf-8")
    feature = (templates / "feature-request.yml").read_text(encoding="utf-8")

    assert "security/advisories/new" in config
    assert "private research data" in bug
    assert "scientific" in bug
    assert "calibration" in feature


def test_github_dependency_updates_cover_python_and_ci_actions():
    dependabot = (ROOT / ".github" / "dependabot.yml").read_text(encoding="utf-8")

    assert "package-ecosystem: pip" in dependabot
    assert "package-ecosystem: github-actions" in dependabot
    assert "interval: monthly" in dependabot


def test_pull_request_template_requires_evidence_and_privacy_review():
    template = (ROOT / ".github" / "pull_request_template.md").read_text(
        encoding="utf-8"
    )

    assert "pytest -q" in template
    assert "calibration" in template
    assert "auto-upload" in template
    assert "private research artifacts" in template


def test_ci_runs_the_shipped_image_through_health_and_real_solver_checks():
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")

    assert "docker build --tag haloforge:" in workflow
    assert "/_stcore/health" in workflow
    assert "tiny_class_smoke_test(DEFAULT_PARAMS)" in workflow
    assert "docker build --target test" in workflow
    assert "python -m pytest -q -p no:cacheprovider" in workflow
    # Export verification must run before the shell EXIT trap stops the container.
    shipped_step = workflow.split("- name: Verify the shipped container", 1)[1]
    shipped_script = shipped_step.split("\n      - name:", 1)[0]
    assert "static_figure_export_smoke_test" in shipped_script
    assert "trap cleanup EXIT" in shipped_script


def test_streamlit_runtime_disables_usage_statistics_and_ships_configuration():
    import tomllib

    config = tomllib.loads((ROOT / ".streamlit/config.toml").read_text())
    assert config["browser"]["gatherUsageStats"] is False
    assert config["client"]["toolbarMode"] == "minimal"
    assert "COPY .streamlit ./.streamlit" in (ROOT / "Dockerfile").read_text()
