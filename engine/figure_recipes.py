"""Named visual recipes for the same underlying scientific figure."""

from __future__ import annotations


RECIPES = ("Interactive", "Paper draft", "Lecture", "Student explanation", "Share card")


def apply_chart_theme(fig, theme: str):
    """Match the app palette while preserving data, ranges and scale types."""
    if theme not in {"Dark", "Light", "High contrast"}:
        raise ValueError(f"Unknown chart theme: {theme}")
    if theme == "Dark":
        return fig
    light = theme == "Light"
    foreground = "#132126" if light else "#ffffff"
    background = "#ffffff" if light else "#000000"
    grid = "#d4dddf" if light else "#666666"
    fig.update_layout(
        paper_bgcolor=background,
        plot_bgcolor=background,
        font=dict(color=foreground),
        title_font=dict(color=foreground),
        legend=dict(bgcolor=background, bordercolor=grid, font=dict(color=foreground)),
        hoverlabel=dict(
            bgcolor=background, bordercolor=foreground, font_color=foreground
        ),
    )
    for update in (fig.update_xaxes, fig.update_yaxes):
        update(
            gridcolor=grid,
            zerolinecolor=grid,
            linecolor=foreground,
            tickfont=dict(color=foreground),
            title_font=dict(color=foreground),
        )
    if light:
        palette = {
            "#35d7e5": "#006f7a",
            "#ffb454": "#985800",
            "#a78bfa": "#6941b2",
            "#ff718b": "#ad2748",
            "#45d49d": "#06764d",
            "#67a7ff": "#245fad",
            "#f5df70": "#786600",
            "#d879ff": "#9141ad",
            "#38bdf8": "#006e9f",
            "#fbbf24": "#856000",
            "#fb7185": "#ad2748",
            "#22c55e": "#06764d",
        }
        for trace in fig.data:
            for attribute in ("line", "marker"):
                style = getattr(trace, attribute, None)
                color = getattr(style, "color", None)
                if isinstance(color, str) and color.lower() in palette:
                    style.color = palette[color.lower()]
    return fig


def apply_figure_recipe(fig, recipe: str):
    """Apply presentation only; never alter a trace's data or scientific scales."""
    if recipe not in RECIPES:
        raise ValueError(f"Unknown figure recipe: {recipe}")
    if recipe == "Interactive":
        return fig
    if recipe == "Paper draft":
        fig.update_layout(
            paper_bgcolor="#ffffff",
            plot_bgcolor="#ffffff",
            font=dict(family="Arial, sans-serif", color="#111111", size=13),
            title_font=dict(color="#111111", size=18),
            legend=dict(
                bgcolor="rgba(255,255,255,.92)",
                bordercolor="#555555",
                font=dict(color="#111111", size=11),
            ),
        )
        fig.update_xaxes(
            gridcolor="#d9d9d9",
            linecolor="#222222",
            tickfont=dict(color="#111111"),
            title_font=dict(color="#111111"),
        )
        fig.update_yaxes(
            gridcolor="#d9d9d9",
            linecolor="#222222",
            tickfont=dict(color="#111111"),
            title_font=dict(color="#111111"),
        )
    elif recipe == "Lecture":
        fig.update_layout(
            font=dict(size=18),
            title_font=dict(size=26),
            legend=dict(font=dict(size=14)),
        )
        fig.update_xaxes(title_font=dict(size=20), tickfont=dict(size=15))
        fig.update_yaxes(title_font=dict(size=20), tickfont=dict(size=15))
    elif recipe == "Student explanation":
        fig.update_layout(title_font=dict(size=21), hovermode="x unified")
        fig.update_xaxes(showspikes=True, spikecolor="#ffb454")
        fig.update_yaxes(showspikes=True, spikecolor="#ffb454")
    elif recipe == "Share card":
        fig.update_layout(
            width=900,
            height=900,
            title_font=dict(size=28),
            legend=dict(orientation="h", y=-0.12, font=dict(size=13)),
        )
    return fig
