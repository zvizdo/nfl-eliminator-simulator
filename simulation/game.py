import numpy as np
import pandas as pd


class Game(object):
    def __init__(self, features, home_team, away_team, models):
        self.features = features
        self.home_team = home_team
        self.away_team = away_team
        self.models = models  # dict: {'full': model, 'no_spread': model}

    def simulate(self):
        # Use full model if Spread is available, else no_spread model
        if "Spread" in self.features and self.features["Spread"] is not None:
            model = self.models["full"]
            X = [self.features[f] for f in self.models["full"].feature_names_in_]
        else:
            model = self.models["no_spread"]
            X = [self.features[f] for f in self.models["no_spread"].feature_names_in_]
        # If predict returns a probability array, take [0][1], else just [0]
        prob = model.predict_proba(pd.DataFrame([X], columns=model.feature_names_in_))[
            0
        ]
        if isinstance(prob, (list, np.ndarray)) and len(prob) > 1:
            prob = prob[1]  # If model returns [prob_lose, prob_win]
        # winner = self.home_team if np.random.rand() < prob else self.away_team
        winner = self.home_team if prob >= 0.5 else self.away_team
        return winner, prob


class CacheEnabledGame(Game):

    def __init__(
        self, features, home_team, away_team, models, external_game_cache: dict = None
    ):
        super().__init__(features, home_team, away_team, models)
        self.external_game_cache = external_game_cache
        if external_game_cache is not None and isinstance(external_game_cache, dict):
            cache_key = [
                f"{k}:{v}" for k, v in sorted(features.items(), key=lambda x: x[0])
            ] + [home_team, away_team]
            self.ckey = "-".join(cache_key)

    def simulate(self):
        if self.external_game_cache is None:
            return super().simulate()

        if self.ckey in self.external_game_cache:
            return self.external_game_cache[self.ckey]

        r = super().simulate()
        self.external_game_cache[self.ckey] = r
        return r
