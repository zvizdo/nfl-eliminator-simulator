from .week import Week
from .game import Game, CacheEnabledGame
import numpy as np
import copy
import pandas as pd
from collections import defaultdict
import sys


def is_notebook():
    return "ipykernel" in sys.modules


if is_notebook():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


class Season(object):
    def __init__(self, year, models, schedule_df, feature_df):
        self.year = year
        self.models = models  # dict: {'full': model, 'no_spread': model}
        self.schedule_df = schedule_df.copy()
        self.feature_df = feature_df.copy()
        self.team_records = (
            {}
        )  # {team: {'wins': int, 'losses': int, 'games_played': int}}
        self.weeks = []

    def pick_team(self, available_teams, picks) -> str:
        raise NotImplementedError()

    def flip_winner_loser(self):
        return False

    def end_of_week_checkin(self, pick, pick_won: bool) -> bool:
        return not pick_won

    def simulate(
        self,
        week=1,
        spread=None,  # Expects a dict
        rank=None,  # Expects a dict
        prior_weeks=None,
        end_week=18,
        survivor_picks=None,
    ):
        self.team_records = {}
        self.weeks = []
        results = {}

        if prior_weeks:
            self.team_records = {
                team: record.copy() for team, record in prior_weeks.items()
            }

        all_teams = set(self.schedule_df["Home_Team"]).union(
            set(self.schedule_df["Away_Team"])
        )
        available_teams = set(all_teams)
        picks = [] if survivor_picks is None else list(survivor_picks)
        if picks:
            available_teams = available_teams - set(picks)

        for wk in range(week, end_week + 1):
            week_games = []
            week_schedule = self.schedule_df[self.schedule_df["Week"] == wk]
            for _, row in week_schedule.iterrows():
                features = self.feature_df[
                    (self.feature_df["Year"] == self.year)
                    & (self.feature_df["Week"] == wk)
                    & (self.feature_df["Home_Team"] == row["Home_Team"])
                    & (self.feature_df["Away_Team"] == row["Away_Team"])
                ]
                features = features.iloc[0].to_dict() if not features.empty else {}

                # Use dict lookups for spread and rank
                if spread is not None and wk == week:
                    features["Spread"] = spread.get(
                        (row["Home_Team"], row["Away_Team"])
                    )

                if rank is not None:
                    features["Home_Rank"] = rank.get(row["Home_Team"])
                    features["Away_Rank"] = rank.get(row["Away_Team"])
                    features["Rank_Age"] = wk - week

                for prefix, team in [
                    ("Home", row["Home_Team"]),
                    ("Away", row["Away_Team"]),
                ]:
                    rec = self.team_records.get(
                        team, {"wins": 0, "losses": 0, "games_played": 0}
                    )
                    features[f"{prefix}_Games_Played"] = rec["games_played"]
                    features[f"{prefix}_Wins"] = rec["wins"]
                    features[f"{prefix}_Losses"] = rec["losses"]

                game = CacheEnabledGame(
                    features,
                    row["Home_Team"],
                    row["Away_Team"],
                    self.models,
                    external_game_cache=(
                        getattr(self, "external_game_cache")
                        if hasattr(self, "external_game_cache")
                        else None
                    ),
                )
                week_games.append(game)

            week_obj = Week(week_games)
            self.weeks.append(week_obj)
            week_result = week_obj.simulate()
            results[wk] = week_result

            pick = self.pick_team(available_teams, picks)
            picks.append(pick)
            available_teams.discard(pick)

            pick_won = False
            for game_result in week_result:
                winner = game_result["winner"]
                home, away = game_result["home_team"], game_result["away_team"]
                loser = away if winner == home else home
                if self.flip_winner_loser() and pick in [home, away] and pick != winner:
                    loser, winner = winner, pick

                for team in [home, away]:
                    if team not in self.team_records:
                        self.team_records[team] = {
                            "wins": 0,
                            "losses": 0,
                            "games_played": 0,
                        }
                    self.team_records[team]["games_played"] += 1

                self.team_records[winner]["wins"] += 1
                self.team_records[loser]["losses"] += 1
                if winner == pick:
                    pick_won = True

            if self.end_of_week_checkin(pick, pick_won):
                break

        return {"results": results, "picks": picks}

    def resolve(**kwargs):
        raise NotImplementedError()


class MonteCarloSeason(Season):

    def pick_team(self, available_teams, picks) -> str:
        # Pick a team for this week (if not already provided)
        if available_teams:
            # Pick a random available team instead of alphabetically
            pick = np.random.choice(list(available_teams))
        else:
            pick = picks[-1]

        return pick

    def resolve(
        self,
        week=1,
        spread=None,
        rank=None,
        prior_weeks=None,
        end_week=18,
        survivor_picks=None,
        **kwargs,
    ):
        n = kwargs["n"]

        rank_dict = rank.set_index("Team")["Rank"].to_dict() if rank is not None else {}
        spread_dict = (
            spread.set_index(["Home_Team", "Away_Team"])["Spread"].to_dict()
            if spread is not None
            else {}
        )

        first_pick_lengths = defaultdict(list)
        for _ in range(n):
            r = self.simulate(
                week, spread_dict, rank_dict, prior_weeks, end_week, survivor_picks
            )
            path = r["picks"]
            first_pick = path[0]
            first_pick_lengths[first_pick].append(len(path))

        data = {
            "Team": [],
            "Average_Path_Length": [],
        }
        for team, lengths in first_pick_lengths.items():
            data["Team"].append(team)
            data["Average_Path_Length"].append(sum(lengths) / len(lengths))

        df = pd.DataFrame(data)
        df = df.sort_values(by="Average_Path_Length", ascending=False).reset_index(
            drop=True
        )

        return df


class BeamExploreSeason(Season):

    def __init__(self, year, models, schedule_df, feature_df):
        super().__init__(year, models, schedule_df, feature_df)
        self.game_cache = {}

    def pick_team(self, available_teams, picks):
        return self.team_to_pick

    def flip_winner_loser(self):
        return True

    def end_of_week_checkin(self, pick, pick_won):
        return False

    def _filter_teams_by_rank(self, all_teams, week_schedule, rank_dict):
        """
        Eliminate any team who is playing against a team with at least 10 higher rank.
        Returns a set of eligible teams.
        """
        if not rank_dict:
            return all_teams

        filtered_teams = set()
        for t in all_teams:
            game_row = week_schedule[
                (week_schedule["Home_Team"] == t) | (week_schedule["Away_Team"] == t)
            ]
            if game_row.empty:
                continue
            row = game_row.iloc[0]
            opponent = row["Away_Team"] if row["Home_Team"] == t else row["Home_Team"]

            t_rank = rank_dict.get(t)
            opp_rank = rank_dict.get(opponent)

            if t_rank is not None and opp_rank is not None and (t_rank > opp_rank + 10):
                continue
            filtered_teams.add(t)

        return filtered_teams

    def resolve(
        self,
        week=1,
        spread=None,
        rank=None,
        prior_weeks=None,
        end_week=18,
        survivor_picks=None,
        **kwargs,
    ):
        k = kwargs.get("k", 100)
        n = kwargs.get("n", 1000)
        self.external_game_cache = {}

        # Convert dataframes to dictionaries for faster lookups
        rank_dict = rank.set_index("Team")["Rank"].to_dict() if rank is not None else {}
        spread_dict = (
            spread.set_index(["Home_Team", "Away_Team"])["Spread"].to_dict()
            if spread is not None
            else {}
        )

        best_paths = []
        for _ in tqdm(range(n), desc="Simulations", total=n, leave=False):
            beam_paths = [
                {
                    "picks": [] if not survivor_picks else survivor_picks,
                    "p": np.log(1.0),
                    "prior_weeks": copy.deepcopy(prior_weeks) if prior_weeks else {},
                }
            ]

            for wk in tqdm(
                range(week, end_week + 1), desc="Week progress", leave=False
            ):
                candidate_paths = []
                week_schedule = self.schedule_df[self.schedule_df["Week"] == wk]

                # Pre-filter teams for the week
                all_teams_in_week = set(week_schedule["Home_Team"]).union(
                    set(week_schedule["Away_Team"])
                )

                # Filter teams by rank once per week
                eligible_teams = self._filter_teams_by_rank(
                    all_teams_in_week, week_schedule, rank_dict
                )

                for path in tqdm(beam_paths, desc="Explore paths", leave=False):
                    available_teams = eligible_teams - set(path["picks"])

                    for team_to_pick in available_teams:
                        self.team_to_pick = team_to_pick

                        # Pass dictionaries instead of dataframes
                        r = self.simulate(
                            week=wk,
                            spread=spread_dict if wk == week else None,
                            rank=rank_dict,
                            prior_weeks=path.get("prior_weeks", None),
                            end_week=wk,
                            survivor_picks=path["picks"],
                        )

                        game = [
                            g
                            for g in r["results"][wk]
                            if team_to_pick in [g["home_team"], g["away_team"]]
                        ][0]

                        p = (
                            game["prob"]
                            if game["home_team"] == team_to_pick
                            else 1 - game["prob"]
                        )

                        new_path = {
                            "picks": path["picks"] + [team_to_pick],
                            "p": path["p"] + np.log(p),
                            "prior_weeks": self.team_records,
                        }
                        candidate_paths.append(new_path)

                candidate_paths.sort(key=lambda x: x["p"], reverse=True)
                beam_paths = candidate_paths[:k]

            best_paths.extend(beam_paths)

        return best_paths
