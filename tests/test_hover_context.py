from engine.hover_context import hover_context, with_hover_context


def test_hover_context_is_specific_for_major_hmf_chart_and_honest_for_unknown_charts():
    assert "dotted" in hover_context("Differential halo mass function")
    assert "stored plotted sample" in hover_context("Unregistered diagnostic")


def test_hover_context_preserves_values_escapes_text_and_is_idempotent():
    template = "M=%{x:.3e}<br>n=%{y:.4e}<extra></extra>"
    result = with_hover_context(template, "<calibration & caveat>")
    assert "M=%{x:.3e}" in result
    assert "&lt;calibration &amp; caveat&gt;" in result
    assert result.endswith("<extra></extra>")
    assert with_hover_context(result, "replacement") == result
