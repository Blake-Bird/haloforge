import plotly.graph_objects as go

from engine.figure_recipes import RECIPES, apply_figure_recipe


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
