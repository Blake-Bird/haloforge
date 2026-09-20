"""A compact, human-readable PDF report for a HaloForge run bundle."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def _text(value: object, fallback: str = "not recorded") -> str:
    text = str(value).strip() if value is not None else ""
    return escape(text or fallback).replace("\n", "<br/>")


def _header_footer(canvas, document) -> None:
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#cad4d8"))
    canvas.line(
        document.leftMargin, 0.55 * inch, letter[0] - document.rightMargin, 0.55 * inch
    )
    canvas.setFillColor(colors.HexColor("#54646d"))
    canvas.setFont("HaloSans", 8)
    canvas.drawString(
        document.leftMargin, 0.36 * inch, "HaloForge - local scientific run report"
    )
    canvas.drawRightString(
        letter[0] - document.rightMargin, 0.36 * inch, f"Page {document.page}"
    )
    canvas.restoreState()


def build_run_pdf(run: dict) -> bytes:
    """Render a provenance-first report; it intentionally contains no claims beyond saved evidence."""
    font_dir = Path(__file__).resolve().parents[1] / "assets" / "fonts"
    for name, filename in (
        ("HaloSans", "DejaVuSans.ttf"),
        ("HaloSans-Bold", "DejaVuSans-Bold.ttf"),
    ):
        if name not in pdfmetrics.getRegisteredFontNames():
            pdfmetrics.registerFont(TTFont(name, str(font_dir / filename)))
    pdfmetrics.registerFontFamily("HaloSans", normal="HaloSans", bold="HaloSans-Bold")
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        leftMargin=0.7 * inch,
        rightMargin=0.7 * inch,
        topMargin=0.65 * inch,
        bottomMargin=0.75 * inch,
        title=f"HaloForge - {_text(run.get('name'))}",
        author="HaloForge",
    )
    styles = getSampleStyleSheet()
    title = ParagraphStyle(
        "RunTitle",
        parent=styles["Title"],
        fontName="HaloSans-Bold",
        fontSize=21,
        leading=25,
        textColor=colors.HexColor("#102a3b"),
        alignment=TA_LEFT,
        spaceAfter=6,
    )
    subhead = ParagraphStyle(
        "RunSubhead",
        parent=styles["Heading2"],
        fontName="HaloSans-Bold",
        fontSize=12,
        leading=15,
        textColor=colors.HexColor("#14536d"),
        spaceBefore=14,
        spaceAfter=5,
    )
    body = ParagraphStyle(
        "RunBody",
        parent=styles["BodyText"],
        fontName="HaloSans",
        fontSize=9.2,
        leading=13,
        textColor=colors.HexColor("#1f2e35"),
        spaceAfter=4,
    )
    small = ParagraphStyle(
        "RunSmall",
        parent=body,
        fontSize=7.8,
        leading=10.5,
        textColor=colors.HexColor("#40535d"),
    )
    story = [
        Paragraph(_text(run.get("name"), "Untitled HaloForge run"), title),
        Paragraph(
            "Saved parameters, calculation identity, and scientific checks. Solver completion alone does not establish convergence or validate halo abundances.",
            body,
        ),
    ]
    params, derived, notebook = (
        run.get("params", {}),
        run.get("derived", {}),
        run.get("notebook", {}),
    )
    overview = [
        ["Backend", _text(run.get("class_status"))],
        ["Created", _text(run.get("created_at"))],
        ["Reproducibility hash", _text(run.get("reproducibility_hash"))],
        [
            "Validity state",
            _text(run.get("scientific_validity", {}).get("overall_state")),
        ],
    ]

    def cells(rows, style=small):
        return [[Paragraph(str(value), style) for value in row] for row in rows]

    table = Table(cells(overview), colWidths=[1.45 * inch, 5.05 * inch])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#e9f0f2")),
                ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#1f2e35")),
                ("FONTNAME", (0, 0), (0, -1), "HaloSans-Bold"),
                ("FONTNAME", (1, 0), (1, -1), "HaloSans"),
                ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                ("LEADING", (0, 0), (-1, -1), 11),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#c5d2d7")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    story.extend([Spacer(1, 7), table, Paragraph("Submitted parameters", subhead)])
    parameter_rows = [
        ["Parameter", "Value"],
        *[
            [key, _text(params.get(key))]
            for key in (
                "H0",
                "Omega_m",
                "Omega_b",
                "A_s",
                "n_s",
                "Omega_k",
                "single_z",
                "z_values",
                "window_type",
                "enable_ede",
                "f_EDE",
                "log10_a_c",
                "k_min",
                "k_max",
                "k_points",
                "mass_points",
                "fitting",
                "mass_definition",
            )
        ],
    ]
    parameter_table = Table(
        cells(parameter_rows), colWidths=[2.2 * inch, 4.3 * inch], repeatRows=1
    )
    parameter_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e9f0f2")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "HaloSans-Bold"),
                ("FONTNAME", (0, 1), (-1, -1), "HaloSans"),
                ("FONTSIZE", (0, 0), (-1, -1), 8.3),
                ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#cfdbdf")),
                (
                    "ROWBACKGROUNDS",
                    (0, 1),
                    (-1, -1),
                    [colors.white, colors.HexColor("#f5f8f9")],
                ),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    story.extend([parameter_table, Paragraph("Scientific validity", subhead)])
    claims = run.get("scientific_validity", {}).get("claims", [])
    if claims:
        claim_rows = [["Claim", "State", "Detail"]] + [
            [
                _text(item.get("claim")),
                _text(item.get("state")),
                _text(item.get("detail")),
            ]
            for item in claims
        ]
        claim_table = Table(
            cells(claim_rows),
            colWidths=[1.5 * inch, 1.2 * inch, 3.8 * inch],
            repeatRows=1,
            splitInRow=1,
        )
        claim_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f3eadc")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTNAME", (0, 0), (-1, 0), "HaloSans-Bold"),
                    ("FONTNAME", (0, 1), (-1, -1), "HaloSans"),
                    ("FONTSIZE", (0, 0), (-1, -1), 7.5),
                    ("LEADING", (0, 0), (-1, -1), 9.5),
                    ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#dbc9ad")),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 5),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ]
            )
        )
        story.append(claim_table)
    else:
        story.append(
            Paragraph(
                "No machine-readable scientific-validity record is stored for this run.",
                body,
            )
        )
    story.extend(
        [
            Paragraph("Notebook and caveats", subhead),
            Paragraph(
                f"Research question: {_text(notebook.get('research_question'))}", body
            ),
            Paragraph(f"Hypothesis: {_text(notebook.get('hypothesis'))}", body),
            Paragraph(f"Conclusion: {_text(notebook.get('conclusion'))}", body),
        ]
    )
    caveats = notebook.get("caveats", [])
    if caveats:
        story.append(Paragraph("Caveats", body))
        story.extend(Paragraph("- " + _text(caveat), body) for caveat in caveats)
    story.extend(
        [
            Paragraph("Derived values", subhead),
            Paragraph(
                _text(
                    "; ".join(f"{key}={value}" for key, value in derived.items()),
                    "No derived values recorded.",
                )
                or "No derived values recorded.",
                small,
            ),
        ]
    )
    doc.build(story, onFirstPage=_header_footer, onLaterPages=_header_footer)
    return buffer.getvalue()
