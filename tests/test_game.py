import sys
import os

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/../"))

import pytest
import numpy as np
from simulation.game import Game, CacheEnabledGame


class DummyModel:
    def __init__(self, prob=0.7, with_spread=True):
        self.feature_names_in_ = [
            "Is_Neutral",
            "Home_Rank",
            "Away_Rank",
            "Home_Days_Since_Last_Game",
            "Away_Days_Since_Last_Game",
            "Home_Games_Played",
            "Away_Games_Played",
            "Home_Wins",
            "Away_Wins",
            "Home_Losses",
            "Away_Losses",
        ]
        if with_spread:
            self.feature_names_in_.append("Spread")

        self.prob = prob

    def predict_proba(self, X):
        # Return as array or float depending on test
        return [1 - self.prob, self.prob]


def test_game_simulate_full_and_no_spread():
    features = {k: 1 for k in DummyModel(0.7, with_spread=False).feature_names_in_}
    features["Spread"] = 3
    print(features)
    models = {"full": DummyModel(0.7), "no_spread": DummyModel(0.5, with_spread=False)}
    game = CacheEnabledGame(features, "TeamA", "TeamB", models)
    winner, prob = game.simulate()
    assert winner in ["TeamA", "TeamB"]
    assert 0 <= prob <= 1
    # Test no_spread
    features.pop("Spread")
    game2 = CacheEnabledGame(features, "TeamA", "TeamB", models)
    winner2, prob2 = game2.simulate()
    assert winner2 in ["TeamA", "TeamB"]
    assert 0 <= prob2 <= 1


def test_game_simulate_full_and_no_spread_cached():
    external_cache = {}

    features = {k: 1 for k in DummyModel(0.7).feature_names_in_}
    features["Spread"] = 3
    models = {"full": DummyModel(0.7), "no_spread": DummyModel(0.5)}
    game = CacheEnabledGame(features, "TeamA", "TeamB", models, external_cache)
    winner, prob = game.simulate()
    assert winner in ["TeamA", "TeamB"]
    assert 0 <= prob <= 1

    game = CacheEnabledGame(features, "TeamA", "TeamB", models, external_cache)
    winner, prob = game.simulate()
    assert winner in ["TeamA", "TeamB"]
    assert 0 <= prob <= 1
