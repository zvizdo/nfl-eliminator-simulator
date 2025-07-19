#!/usr/bin/env python
# coding: utf-8

import argparse
import duckdb
import cloudpickle as pickle
import pandas as pd
import numpy as np
import sys
import os
import json

# Add the project root to sys.path for direct script execution
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/../"))
from simulation.season import BeamExploreSeason


def get_season_schedule(db, year):
    """
    Fetch the season schedule for a given year from the database.
    """
    schedule_df = db.sql(
        f"""
        SELECT
            Year, Week, Home_Team, Away_Team,
            Is_Neutral, Home_days_Since_Last_Game, Away_days_Since_Last_Game
        FROM game_features
        WHERE Year = {year}
        ORDER BY Week, Home_Team, Away_Team
        """
    ).df()
    return schedule_df


def get_season_week_speads(db, year, week):
    """
    Fetch the season spreads for a given year and week from the database.
    """
    spread_df = db.sql(
        f"""
        SELECT
            Home_Team, Away_Team, Spread
        FROM game_features
        WHERE Year = {year} AND Week = {week}
        ORDER BY Home_Team, Away_Team
        """
    ).df()
    return spread_df


def get_season_week_rankings(db, year, week):
    """
    Fetch the season rankings for a given year and week from the database.
    """
    rank_df = db.sql(
        f"""
        SELECT
            Team,
            ROW_NUMBER() OVER (PARTITION BY Year, Week ORDER BY Rating DESC) AS Rank
        FROM nfl_rankings
        WHERE Year = {year} AND Week = {week}
        ORDER BY Team
        """
    ).df()
    return rank_df


def get_team_records_from_db(db, year, week):
    """
    Generate team_records dict for all teams up to (but not including) the given week.
    """
    query = f"""
        SELECT Home_Team, Away_Team, Home_Won
        FROM game_features
        WHERE Year = {year} AND Week < {week}
    """
    df = db.sql(query).df()
    team_records = {}

    for _, row in df.iterrows():
        home = row["Home_Team"]
        away = row["Away_Team"]
        home_won = row["Home_Won"]

        for team in [home, away]:
            if team not in team_records:
                team_records[team] = {"wins": 0, "losses": 0, "games_played": 0}

        # Update games played
        team_records[home]["games_played"] += 1
        team_records[away]["games_played"] += 1

        # Update wins/losses
        if home_won:
            team_records[home]["wins"] += 1
            team_records[away]["losses"] += 1
        else:
            team_records[away]["wins"] += 1
            team_records[home]["losses"] += 1

    return team_records


def run_greedy_beam_path(year, models, schedule_df, k=10000):
    survivor_picks = []
    prior_weeks = {}
    path = []
    max_week = schedule_df["Week"].max()
    for wk in range(1, max_week + 1):
        with duckdb.connect("./data/data.db") as db:
            spread_df = get_season_week_speads(db, year, wk)
            rank_df = get_season_week_rankings(db, year, wk)
            prior_weeks = get_team_records_from_db(db, year, wk)

        beams = BeamExploreSeason(year, models, schedule_df, schedule_df.copy())
        bp = beams.resolve(
            week=wk,
            end_week=max_week,
            spread=spread_df,
            rank=rank_df,
            k=k,
            n=1,
            survivor_picks=survivor_picks,
            prior_weeks=prior_weeks,
        )

        pick_scores = {}
        for path_obj in bp:
            pick = path_obj["picks"][wk - 1]
            pick_scores.setdefault(pick, 0)
            pick_scores[pick] += np.exp(path_obj["p"])
        best_pick = max(pick_scores, key=pick_scores.get)
        path.append(best_pick)
        survivor_picks = path.copy()
    return path


def main():
    parser = argparse.ArgumentParser(
        description="Run greedy beam search survivor path for a range of years."
    )
    parser.add_argument(
        "--year_start",
        type=int,
        required=True,
        help="Start year (inclusive)",
    )
    parser.add_argument(
        "--year_end",
        type=int,
        required=True,
        help="End year (inclusive)",
    )
    parser.add_argument(
        "--k",
        type=int,
        required=False,
        default=10000,
        help="Beam width (default: 10000)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=False,
        default="./results/greedy_path_{year}_k{k}.json",
        help="Output file name pattern, e.g. ./results/greedy_path_{year}.json",
    )
    parser.add_argument(
        "--model_full",
        type=str,
        default="./models/lr_full.pkl",
        help="Path to full model pickle",
    )
    parser.add_argument(
        "--model_ns",
        type=str,
        default="./models/lr_no_spread.pkl",
        help="Path to no-spread model pickle",
    )
    args = parser.parse_args()

    with open(args.model_full, "rb") as f:
        full_model = pickle.load(f)
    with open(args.model_ns, "rb") as f:
        no_spread_model = pickle.load(f)
    models = {"full": full_model, "no_spread": no_spread_model}

    for year in range(args.year_start, args.year_end + 1):
        print(f"Running greedy path for year: {year}")
        with duckdb.connect("./data/data.db") as db:
            schedule_df = get_season_schedule(db, year)

        greedy_path = run_greedy_beam_path(year, models, schedule_df, k=args.k)
        print("Best greedy path:", greedy_path)

        output_file = args.output.format(year=year, k=args.k)
        with open(output_file, "wb") as f:
            f.write(json.dumps(greedy_path).encode("utf-8"))

        print(f"Saved greedy path for {year} to {output_file}")


if __name__ == "__main__":
    main()
