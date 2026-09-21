import pytest

from engine.campaign import cartesian_campaign


def test_campaign_expansion_is_deterministic_and_complete():
    assert cartesian_campaign({"n_s": [0.96, 0.98], "A_s": [2, 3]}) == [
        {"n_s": 0.96, "A_s": 2},
        {"n_s": 0.96, "A_s": 3},
        {"n_s": 0.98, "A_s": 2},
        {"n_s": 0.98, "A_s": 3},
    ]


def test_campaign_rejects_duplicates_and_explosions():
    with pytest.raises(ValueError, match="duplicate"):
        cartesian_campaign({"n_s": [0.96, 0.96]})
    with pytest.raises(ValueError, match="above"):
        cartesian_campaign({"a": [1, 2], "b": [1, 2]}, max_runs=3)
