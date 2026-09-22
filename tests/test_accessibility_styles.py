from pathlib import Path


def test_stylesheet_honors_system_reduced_motion_preference():
    css = (Path(__file__).resolve().parents[1] / "assets" / "custom.css").read_text(
        encoding="utf-8"
    )
    assert "@media (prefers-reduced-motion: reduce)" in css
    assert "animation-iteration-count:1!important" in css


def test_guided_causal_motion_has_a_named_visual_and_respects_shared_motion_policy():
    root = Path(__file__).resolve().parents[1]
    css = (root / "assets" / "custom.css").read_text(encoding="utf-8")
    app = (root / "app.py").read_text(encoding="utf-8")

    assert ".causal-reveal" in css
    assert "@keyframes causalPulse" in css
    # The global reduced-motion rule applies to this animation as well.
    assert "@media (prefers-reduced-motion: reduce)" in css
    assert "def causal_reveal_visual" in app
    assert "Motion is illustrative, not evidence" in app


def test_light_theme_explicitly_keeps_streamlit_alert_copy_readable():
    app = (Path(__file__).resolve().parents[1] / "app.py").read_text(
        encoding="utf-8"
    )
    assert '[data-testid="stAlert"]' in app
    assert '[data-testid="stAlert"] *{color:#132126!important}' in app
