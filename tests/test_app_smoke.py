"""End-to-end startup evidence for the assembled local Streamlit application."""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.request import urlopen

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


def _unused_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def test_streamlit_application_starts_with_clean_local_storage(tmp_path):
    """Exercise the assembled app without requiring an installed CLASS solver.

    A first launch must render its guided, solver-free entry state.  Numerical
    execution is deliberately tested separately because production runs require
    the compiled AxiCLASS binding, which is supplied by the container image.
    """
    port = _unused_local_port()
    environment = {
        **os.environ,
        "HALOFORGE_DATA_DIR": str(tmp_path / "haloforge-data"),
        "HALOFORGE_DEPLOYMENT": "local",
    }
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            "app.py",
            "--server.headless=true",
            f"--server.port={port}",
            "--server.address=127.0.0.1",
        ],
        cwd=ROOT,
        env=environment,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        health_url = f"http://127.0.0.1:{port}/_stcore/health"
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise AssertionError("Streamlit exited before becoming healthy.")
            try:
                with urlopen(health_url, timeout=0.5) as response:
                    if response.status == 200 and response.read() == b"ok":
                        break
            except OSError:
                time.sleep(0.1)
        else:
            raise AssertionError("Streamlit did not become healthy within 15 seconds.")

        with urlopen(f"http://127.0.0.1:{port}/", timeout=2) as response:
            assert response.status == 200
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


def test_first_use_requires_a_deliberate_prediction_before_staging(
    monkeypatch, tmp_path
):
    """Guard the prediction-first teaching contract without invoking a solver."""
    monkeypatch.setenv("HALOFORGE_DATA_DIR", str(tmp_path / "haloforge-data"))
    monkeypatch.setenv("HALOFORGE_DEPLOYMENT", "local")

    app = AppTest.from_file(str(ROOT / "app.py"))
    app.run(timeout=15)

    assert not app.exception
    assert app.selectbox[0].label == "Curated experiment"
    assert app.radio[0].value is None
    assert app.button[0].label == "Set up this experiment"
    assert app.button[0].disabled

    app.radio[0].set_value("Fewer massive halos").run(timeout=15)
    assert not app.button[0].disabled

    app.button[0].click().run(timeout=15)
    assert app.session_state["onboarding_baseline_params"]["enable_ede"] is False
    assert app.session_state["params"]["enable_ede"] is True
    app.selectbox[0].set_value("More small-scale power").run(timeout=15)
    assert not app.exception
    assert app.radio[0].value is None
    assert app.button[0].disabled
    assert all(button.label != "Calculate the guided universe" for button in app.button)
