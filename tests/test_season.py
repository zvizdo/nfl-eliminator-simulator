import sys
import os

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/../"))

from simulation.season import Season, MonteCarloSeason
from simulation.week import Week
from simulation.game import Game
import pandas as pd
import pytest


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


def test_season_simulate():
    # Create a round-robin schedule for 4 teams, each team plays every other team home and away
    teams = ["A", "B", "C", "D"]
    schedule_rows = []
    feature_rows = []
    week = 1
    # Manually create a valid round-robin for 4 teams, 6 weeks, 2 games per week
    matchups = [
        [("A", "B"), ("C", "D")],
        [("A", "C"), ("B", "D")],
        [("A", "D"), ("B", "C")],
        [("B", "A"), ("D", "C")],
        [("C", "A"), ("D", "B")],
        [("D", "A"), ("C", "B")],
    ]
    for week, games in enumerate(matchups, 1):
        for home, away in games:
            schedule_rows.append(
                {"Year": 2024, "Week": week, "Home_Team": home, "Away_Team": away}
            )
            feature_rows.append(
                {
                    "Year": 2024,
                    "Week": week,
                    "Home_Team": home,
                    "Away_Team": away,
                    "Is_Neutral": 0,
                    "Home_Days_Since_Last_Game": 7,
                    "Away_Days_Since_Last_Game": 7,
                }
            )
    schedule_df = pd.DataFrame(schedule_rows)
    feature_df = pd.DataFrame(feature_rows)
    spread = pd.DataFrame(
        {
            "Home_Team": [row["Home_Team"] for row in schedule_rows],
            "Away_Team": [row["Away_Team"] for row in schedule_rows],
            "Spread": [5] * len(schedule_rows),
        }
    )
    rank = pd.DataFrame({"Team": teams, "Rank": [1, 2, 3, 4]})

    rank = rank.set_index("Team")["Rank"].to_dict() if rank is not None else {}
    spread = (
        spread.set_index(["Home_Team", "Away_Team"])["Spread"].to_dict()
        if spread is not None
        else {}
    )

    models = {"full": DummyModel(0.8), "no_spread": DummyModel(0.6, with_spread=False)}
    season = MonteCarloSeason(2024, models, schedule_df, feature_df)
    results = season.simulate(week=1, spread=spread, rank=rank, end_week=6)
    # The output is now a dict with 'results' and 'picks'
    assert "results" in results
    assert "picks" in results
    sim_results = results["results"]
    picks = results["picks"]
    # There should be at least 1 week, up to 6 if survivor never loses
    assert len(sim_results) >= 1
    assert len(sim_results) <= 6
    # Each week should have 2 games
    for w in sim_results:
        assert len(sim_results[w]) == 2
    # Check that spread and rank were used for week 1
    assert season.weeks[0].games[0].features["Spread"] == 5
    assert "Home_Rank" in season.weeks[0].games[0].features
    assert "Away_Rank" in season.weeks[0].games[0].features
    # Check that rank were used for week 2
    if len(season.weeks) > 1:
        assert "Home_Rank" in season.weeks[1].games[0].features
        assert "Away_Rank" in season.weeks[1].games[0].features
    # Check team records
    assert set(season.team_records.keys()) == set(teams)
    total_games = sum([v["wins"] + v["losses"] for v in season.team_records.values()])
    # The number of games depends on how many weeks were simulated
    assert total_games == 4 * len(
        sim_results
    )  # 2 games per week * 2 teams per game * weeks
    # Picks should be as many as weeks simulated
    assert len(picks) == len(sim_results)


def test_season_simulate_many():
    # Create a round-robin schedule for 4 teams, each team plays every other team home and away
    teams = ["A", "B", "C", "D"]
    schedule_rows = []
    feature_rows = []
    week = 1
    # Manually create a valid round-robin for 4 teams, 6 weeks, 2 games per week
    matchups = [
        [("A", "B"), ("C", "D")],
        [("A", "C"), ("B", "D")],
        [("A", "D"), ("B", "C")],
        [("B", "A"), ("D", "C")],
        [("C", "A"), ("D", "B")],
        [("D", "A"), ("C", "B")],
    ]
    for week, games in enumerate(matchups, 1):
        for home, away in games:
            schedule_rows.append(
                {"Year": 2024, "Week": week, "Home_Team": home, "Away_Team": away}
            )
            feature_rows.append(
                {
                    "Year": 2024,
                    "Week": week,
                    "Home_Team": home,
                    "Away_Team": away,
                    "Is_Neutral": 0,
                    "Home_Days_Since_Last_Game": 7,
                    "Away_Days_Since_Last_Game": 7,
                }
            )
    schedule_df = pd.DataFrame(schedule_rows)
    feature_df = pd.DataFrame(feature_rows)
    spread = pd.DataFrame(
        {
            "Home_Team": [row["Home_Team"] for row in schedule_rows],
            "Away_Team": [row["Away_Team"] for row in schedule_rows],
            "Spread": [5] * len(schedule_rows),
        }
    )
    rank = pd.DataFrame({"Team": teams, "Rank": [1, 2, 3, 4]})
    models = {"full": DummyModel(0.8), "no_spread": DummyModel(0.6, with_spread=False)}
    pick_paths = []
    for _ in range(1000):
        season = MonteCarloSeason(2024, models, schedule_df, feature_df)
        results = season.simulate(week=1, spread=spread, rank=rank, end_week=6)
        picks = results["picks"]
        # Store the tuple of picks (can be of length < 6)
        pick_paths.append(tuple(picks))
    # Create a DataFrame of pick paths
    pick_df = pd.DataFrame({"path": pick_paths})
    # Count the frequency of each unique path
    freq = pick_df["path"].value_counts().reset_index()
    freq.columns = ["path", "count"]
    # The most frequent path should be at the top
    most_frequent_path = freq.iloc[0]["path"]
    most_frequent_count = freq.iloc[0]["count"]
    # Assert that the most frequent path occurred at least once
    assert most_frequent_count > 0
    # Optionally print the top 5 most frequent paths for debugging
    print(freq.head())
