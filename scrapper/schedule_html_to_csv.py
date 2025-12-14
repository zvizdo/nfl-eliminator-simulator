import pandas as pd
from bs4 import BeautifulSoup
import re
from datetime import datetime

# Input/output files
HTML_FILE = "./2025_schedule.html"
CSV_FILE = "../data/nfl_betting_2025.csv"

# Output columns
COLUMNS = [
    "Date",
    "Team",
    "Opponent",
    "Week",
    "Location",
    "Spread",
    "Total (O/U)",
    "Money Line",
    "MOV",
    "ATS Margin",
    "Combined",
    "O/U Margin",
    "Score Team",
    "Score Opponent",
    "Won",
]

# Map HTML team names to CSV team names
TEAM_MAP = {
    "Arizona Cardinals": "Arizona",
    "Atlanta Falcons": "Atlanta",
    "Baltimore Ravens": "Baltimore",
    "Buffalo Bills": "Buffalo",
    "Carolina Panthers": "Carolina",
    "Chicago Bears": "Chicago",
    "Cincinnati Bengals": "Cincinnati",
    "Cleveland Browns": "Cleveland",
    "Dallas Cowboys": "Dallas",
    "Denver Broncos": "Denver",
    "Detroit Lions": "Detroit",
    "Green Bay Packers": "Green Bay",
    "Houston Texans": "Houston",
    "Indianapolis Colts": "Indianapolis",
    "Jacksonville Jaguars": "Jacksonville",
    "Kansas City Chiefs": "Kansas City",
    "Las Vegas Raiders": "Las Vegas",
    "Los Angeles Chargers": "LA Chargers",
    "Los Angeles Rams": "LA Rams",
    "Miami Dolphins": "Miami",
    "Minnesota Vikings": "Minnesota",
    "New England Patriots": "New England",
    "New Orleans Saints": "New Orleans",
    "New York Giants": "NY Giants",
    "New York Jets": "NY Jets",
    "Philadelphia Eagles": "Philadelphia",
    "Pittsburgh Steelers": "Pittsburgh",
    "San Francisco 49ers": "San Francisco",
    "Seattle Seahawks": "Seattle",
    "Tampa Bay Buccaneers": "Tampa Bay",
    "Tennessee Titans": "Tennessee",
    "Washington Commanders": "Washington",
}


def parse_date(date_str):
    # Example: "Thu Sep 4" or "Mon Sep 8" or "Sun Nov 30"
    date_str = date_str.strip("&nbsp;").strip()
    m = re.match(r"(\w{3}) (\w{3}) (\d{1,2})", date_str)
    if m:
        dow, month, day = m.groups()
        # Guess year from file name
        year = 2025
        try:
            dt = datetime.strptime(f"{month} {day} {year}", "%b %d %Y")
            return dt.strftime("%Y-%m-%d")
        except Exception:
            return date_str
    return date_str


def parse_schedule_html(html_file):
    with open(html_file, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f, "html.parser")
    main_table = soup.find("table", width="80%")
    if not main_table:
        raise ValueError("Main schedule table not found.")
    rows = main_table.find_all("tr")
    week = None
    data = []
    last_date = None
    for row in rows:
        # Week header
        hdr = row.find("a")
        if hdr and "Week" in hdr.text:
            week_match = re.search(r"Week (\d+)", hdr.text)
            if week_match:
                week = week_match.group(1)
            continue
        cells = row.find_all("td")
        if len(cells) == 4 and cells[2].get_text(strip=True):
            # Date cell
            date_raw = cells[0].get_text(strip=True)
            if date_raw:
                date = parse_date(date_raw)
                last_date = date
            else:
                date = last_date
            # Team cells
            away = cells[2].get_text(strip=True).replace("&nbsp;", "").strip()
            home = cells[3].get_text(strip=True).replace("&nbsp;", "").strip()
            # Remove any footnote markers, superscripts, or trailing non-alphabetic characters
            away = re.sub(r"[^A-Za-z .&]+$", "", away).strip()
            home = re.sub(r"[^A-Za-z .&]+$", "", home).strip()
            away = away.rstrip("*").strip()
            home = home.rstrip("*").strip()
            away = TEAM_MAP.get(away, away)
            home = TEAM_MAP.get(home, home)
            # Add both perspectives (for betting CSV)
            for team, opp, loc in [(away, home, "Away"), (home, away, "Home")]:
                row_dict = {col: "" for col in COLUMNS}
                row_dict["Date"] = date
                row_dict["Team"] = team
                row_dict["Opponent"] = opp
                row_dict["Week"] = week
                row_dict["Location"] = loc
                data.append(row_dict)
    return pd.DataFrame(data, columns=COLUMNS)


def main():
    df = parse_schedule_html(HTML_FILE)
    df.to_csv(CSV_FILE, index=False)
    print(f"Saved schedule to {CSV_FILE} with {len(df)} rows.")


if __name__ == "__main__":
    main()
