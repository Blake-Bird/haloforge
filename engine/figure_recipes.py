"""Named visual recipes for the same underlying scientific figure."""

from __future__ import annotations


RECIPES = ("Interactive", "Paper draft", "Lecture", "Student explanation", "Share card")


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
