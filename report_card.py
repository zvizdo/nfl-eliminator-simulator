import pandas as pd
import numpy as np


def report_card(csv_path):
    df = pd.read_csv(csv_path)
    week_cols = [col for col in df.columns if col.startswith("week_")]

    # 1. Most probable path
    most_probable = df.loc[df["log_prob"].idxmax()]
    print("Most Probable Path (highest log_prob):")
    for week in week_cols:
        print(f"{week}: {most_probable[week]}")
    print(f"log_prob: {most_probable['log_prob']}\n")

    # 2. For the lowest week, top 5 teams by cumulative probability
    week1 = week_cols[0]
    top_week1 = df.groupby(week1)["log_prob"].agg(lambda x: np.exp(x).sum())
    total_prob = np.exp(df["log_prob"]).sum()
    top_week1 = top_week1.sort_values(ascending=False).head(5)
    print(f"Top 5 teams for {week1} by cumulative probability (% of total):")
    for team, prob in top_week1.items():
        pct = 100 * prob / total_prob
        print(f"{team}: {prob:.4f} ({pct:.2f}%)")
    print()

    # 3. For each team, top 3 weeks to pick the team and % for those weeks
    print("Top 3 weeks to pick each team (by cumulative probability):")
    teams = set(df[week_cols].values.flatten())
    total_prob = np.exp(df["log_prob"]).sum()
    for team in sorted(teams):
        week_probs = {}
        for week in week_cols:
            mask = df[week] == team
            week_probs[week] = np.exp(df.loc[mask, "log_prob"]).sum()
        top_weeks = sorted(week_probs.items(), key=lambda x: x[1], reverse=True)[:3]
        print(f"{team}:")
        for week, prob in top_weeks:
            pct = 100 * prob / total_prob
            print(f"  {week}: {pct:.2f}%")
    print()


if __name__ == "__main__":
    report_card("results/beam_2025_wk-5_k10000.csv")
