"""PDF reports preserve literal notes and wrap scientific evidence."""

from io import BytesIO

from pypdf import PdfReader
from reportlab.pdfbase import pdfmetrics

from state.pdf_report import build_run_pdf


def report_fixture():
    return {
        "name": "ΛCDM <baseline> & σ₈ comparison",
        "class_status": "AXICLASS",
        "reproducibility_hash": "a" * 64,
        "params": {
            "single_z": 2.0,
            "z_values": [0.0, 2.0, 10.0],
            "window_type": "Top-hat",
        },
        "scientific_validity": {
            "overall_state": "computed_with_unresolved_numerical_evidence",
            "claims": [
                {
                    "claim": "Numerically converged",
                    "state": "not established",
                    "detail": "Compare σ(M) and dn/dlnM over a wider k range. " * 14,
                },
                {
                    "claim": "Cosmology support",
                    "state": "review",
                    "detail": "A smooth curve does not establish HMF calibration.",
                },
            ],
        },
        "notebook": {
            "research_question": "Does σ₈ < 1 imply fewer halos?",
            "hypothesis": '<img src="/does/not/exist"/> is literal notebook text.',
            "conclusion": "Check Ωₘ, M☉, and P(k).\nA & B remain distinct.",
            "caveats": ["No <b>markup</b> interpretation."],
        },
        "derived": {"sigma8": 0.81, "note": "x < y & z"},
    }


def test_report_preserves_symbols_and_literal_markup():
    pdf = build_run_pdf(report_fixture())
    reader = PdfReader(BytesIO(pdf))
    text = "\n".join(page.extract_text() for page in reader.pages)
    for expected in (
        "ΛCDM <baseline> & σ₈",
        "Does σ₈ < 1",
        '<img src="/does/not/exist"/>',
        "No <b>markup</b>",
        "x < y & z",
        "single_z",
        "2.0",
    ):
        assert expected in text
    cmap = pdfmetrics.getFont("HaloSans").face.charToGlyph
    assert all(ord(symbol) in cmap for symbol in "ΛσΩ₈ₘ⁻☉α→")
    assert all("Page " in page.extract_text() for page in reader.pages)


def test_claim_longer_than_a_page_splits_without_losing_text():
    run = report_fixture()
    run["scientific_validity"]["claims"][0]["detail"] = (
        "Evidence line. " * 1000 + "END_OF_EVIDENCE"
    )
    reader = PdfReader(BytesIO(build_run_pdf(run)))
    text = "\n".join(page.extract_text() for page in reader.pages)
    assert "END_OF_EVIDENCE" in text
    assert text.split().count("Evidence") == 1000
    assert text.split().count("line.") == 1000
