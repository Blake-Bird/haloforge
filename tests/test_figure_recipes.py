import plotly.graph_objects as go

from engine.figure_recipes import RECIPES, apply_figure_recipe, apply_chart_theme


def test_recipes_preserve_scientific_trace_data():
    figure = go.Figure(go.Scatter(x=[1, 2], y=[3, 4]))
    for recipe in RECIPES:
        rendered = apply_figure_recipe(figure, recipe)
        assert list(rendered.data[0].x) == [1, 2]
        assert list(rendered.data[0].y) == [3, 4]


def test_paper_recipe_is_visually_distinct_without_changing_data():
    figure = go.Figure(go.Scatter(x=[1], y=[2]))
    apply_figure_recipe(figure, "Paper draft")
    assert figure.layout.paper_bgcolor == "#ffffff"


def test_light_chart_theme_preserves_scales_and_uses_legible_trace_colors():
    figure = go.Figure(go.Scatter(x=[1, 2], y=[1e-6, 2e-6], line_color="#35d7e5"))
    figure.update_yaxes(type="log", range=[-7, -5])
    apply_chart_theme(figure, "Light")
    assert figure.layout.plot_bgcolor == "#ffffff"
    assert figure.data[0].line.color == "#006f7a"
    assert list(figure.data[0].y) == [1e-6, 2e-6]
    assert figure.layout.yaxis.type == "log"
    assert list(figure.layout.yaxis.range) == [-7, -5]
