import argparse
import duckdb
import cloudpickle as pickle
import pandas as pd
import numpy as np
import sys
import os

# Add the project root to sys.path for direct script execution
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/../"))
from simulation.season import BeamExploreSeason


def get_season_schedule(year, db):
    return db.sql(
        f"""
        SELECT
            Year, Week, Home_Team, Away_Team,
            Is_Neutral, Home_days_Since_Last_Game, Away_days_Since_Last_Game
        FROM game_features
        WHERE Year = {year}
        ORDER BY Week, Home_Team, Away_Team
        """
    ).df()


def get_season_week_spreads(year, week, db):
    return db.sql(
        f"""
        SELECT
            Home_Team, Away_Team, Spread
        FROM game_features
        WHERE Year = {year} AND Week = {week}
        ORDER BY Home_Team, Away_Team
        """
    ).df()


def get_season_week_rankings(year, week, db):
    return db.sql(
        f"""
        SELECT
            Team,
            ROW_NUMBER() OVER (PARTITION BY Year, Week ORDER BY Rating DESC) AS Rank
        FROM nfl_rankings
        WHERE Year = {year} AND Week = {week}
        ORDER BY Team
        """
    ).df()


def get_team_records_from_db(year, week, db):
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


def main():
    parser = argparse.ArgumentParser(
        description="Run NFL season beam search (BeamExploreSeason) and save paths."
    )
    parser.add_argument("--year", type=int, default=2025, help="Season year")
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
    parser.add_argument(
        "--db", type=str, default="./data/data_2025.db", help="Path to DuckDB database"
    )
    parser.add_argument(
        "--week", type=int, default=4, help="Starting week for simulation (default: 1)"
    )
    parser.add_argument(
        "--end_week",
        type=int,
        default=None,
        help="Last week to simulate (default: max week in schedule)",
    )
    parser.add_argument(
        "--picks",
        type=str,
        default="Denver,Baltimore,Seattle",
        help="Picks so far as comma-separated team abbreviations",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=100,
        help="Beam width (number of paths to keep per week)",
    )
    parser.add_argument(
        "--n", type=int, default=1, help="Number of beam search runs (outer loop)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="beam_paths.csv",
        help="Output CSV file for beam search paths",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    args = parser.parse_args()

    np.random.seed(args.seed)

    db = duckdb.connect(args.db)
    schedule_df = get_season_schedule(args.year, db)
    feature_df = schedule_df.copy()
    spread_df = get_season_week_spreads(args.year, args.week, db)
    rank_df = get_season_week_rankings(args.year, args.week, db)
    prior_weeks = get_team_records_from_db(args.year, args.week, db)
    end_week = args.end_week if args.end_week is not None else schedule_df["Week"].max()
    db.close()

    with open(args.model_full, "rb") as f:
        full_model = pickle.load(f)
    with open(args.model_ns, "rb") as f:
        no_spread_model = pickle.load(f)
    models = {"full": full_model, "no_spread": no_spread_model}

    season = BeamExploreSeason(
        args.year,
        models,
        schedule_df[["Year", "Week", "Home_Team", "Away_Team"]],
        feature_df,
    )
    best_paths = season.resolve(
        week=args.week,
        spread=spread_df,
        rank=rank_df,
        prior_weeks=prior_weeks,
        end_week=end_week,
        k=args.k,
        n=args.n,
        survivor_picks=args.picks.split(",") if args.picks else None,
    )

    # Save all paths to CSV (one row per path, columns: week_1, week_2, ..., log_prob)
    out_data = []
    max_len = (end_week - args.week) + 1
    for path in best_paths:
        row = {f"week_{i+1}": t for i, t in enumerate(path["picks"])}
        # Fill missing weeks with None to preserve order
        for i in range(len(path["picks"]), max_len):
            row[f"week_{i+1}"] = None
        row["log_prob"] = path["p"]
        out_data.append(row)
    df = pd.DataFrame(out_data)
    # Ensure columns are ordered: week_1, week_2, ..., log_prob
    week_cols = [f"week_{args.week + i}" for i in range(max_len)]
    df = df[week_cols + ["log_prob"]]
    df.to_csv(args.output, index=False)
    print(f"Beam search paths written to {args.output}")


if __name__ == "__main__":
    main()

    # python simulation/beam_cli.py --year 2024 --week 1 --k 1000 --n 1  --output beam_paths_k1000.csv
    # python simulation/beam_cli.py --year 2024 --week 1 --k 5000 --n 1  --output beam_paths_k5000.csv
    # python simulation/beam_cli.py --year 2024 --week 1 --k 7500 --n 1  --output beam_paths_k7500.csv
    # python simulation/beam_cli.py --year 2024 --week 1 --k 10000 --n 1  --output beam_paths_k10k.csv

    # python beam_cli.py --year 2025 --week 1 --k 10000 --n 1 --db ./data/data_2025.db  --output beam_2025_wk-1_k10000.csv
    # python beam_cli.py --year 2025 --week 2 --k 10000 --n 1 --db ./data/data_2025.db  --output beam_2025_wk-2_k10000.csv
    # python beam_cli.py --year 2025 --week 3 --k 10000 --n 1 --db ./data/data_2025.db  --output beam_2025_wk-3_k10000.csv

    # python beam_cli.py --year 2025 --week 4 --k 10000 --n 1 --db ./data/data_2025.db  --output beam_2025_wk-4_k10000.csv --picks Denver,Baltimore,Seattle
    # python beam_cli.py --year 2025 --week 5 --k 10000 --n 1 --db ./data/data_2025.db  --output beam_2025_wk-5_k10000.csv --picks Denver,Baltimore,Seattle,Detroit
