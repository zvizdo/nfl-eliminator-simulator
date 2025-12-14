import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import argparse
import time
from io import StringIO


def scrape_data(url):
    # Use Safari WebDriver (Safari must have 'Allow Remote Automation' enabled in Develop menu)
    driver = webdriver.Safari()

    try:
        driver.get(url)
        time.sleep(2)  # Wait for the page to load

        # Use JavaScript to set the dropdown value and trigger change event
        driver.execute_script(
            """
            var sel = document.querySelector('select[name="custom-filter-table_length"]');
            if (sel) {
                sel.value = '100';
                var event = new Event('change', { bubbles: true });
                sel.dispatchEvent(event);
            }
        """
        )
        time.sleep(1)  # Wait for the table to update

        # Wait until the table has 100 rows or timeout after 10 seconds
        try:
            WebDriverWait(driver, 10).until(
                lambda d: len(
                    d.find_elements(By.CSS_SELECTOR, "#custom-filter-table tbody tr")
                )
                >= 25
            )
        except Exception as e:
            return pd.DataFrame()

        # Locate the table by its ID
        table = driver.find_element(By.ID, "custom-filter-table")
        table_html = table.get_attribute("outerHTML")

        # Use pandas to parse the HTML table, which will include headers automatically
        df = pd.read_html(StringIO(table_html))[0]
        # Remove rows where the 'Date' column is NaN
        if "Date" in df.columns:
            df = df.dropna(subset=["Date"])

        # Enforce data types
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        if "Week" in df.columns:
            df["Week"] = pd.to_numeric(df["Week"], errors="coerce").astype("Int64")
        if "Score" in df.columns:
            score_split = df["Score"].str.split("-", expand=True)
            df["Score Team"] = pd.to_numeric(score_split[0], errors="coerce").astype(
                "Int64"
            )
            df["Score Opponent"] = pd.to_numeric(
                score_split[1], errors="coerce"
            ).astype("Int64")
            df = df.drop(columns=["Score"])
        # Add 'Won' column: 1 if Score Team > Score Opponent, else 0
        if "Score Team" in df.columns and "Score Opponent" in df.columns:
            df["Won"] = (df["Score Team"] > df["Score Opponent"]).astype(int)
        # All other columns to float (except Date, Week, Score Team, Score Opponent)
        exclude_cols = {
            "Date",
            "Week",
            "Score Team",
            "Score Opponent",
            "Team",
            "Opponent",
            "Location",
        }
        for col in df.columns:
            if col not in exclude_cols:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df

    finally:
        driver.quit()


def scrape_season(year):
    """
    Scrape all weeks (1-18) for a given year and merge into a single DataFrame.
    """
    url_template = "https://betiq.teamrankings.com/nfl/betting-trends/custom-trend-tool/?min_season={year}&min_week={week}&max_week={week}&select_game_type=Regular+Season&max_season={year}"
    dfs = []
    for week in range(1, 19):
        url = url_template.format(year=year, week=week)
        print(f"Scraping for year {year}, week {week} --> {url}")
        df = scrape_data(url)
        if not df.empty:
            dfs.append(df)
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    else:
        return pd.DataFrame()


if __name__ == "__main__":
    # for year in range(2000, 2025):
    #     df = scrape_season(year)  # Example for scraping the 2023 season
    #     df.to_csv(f'../data/nfl_betting_{year}.csv', index=False)  # Save the DataFrame to a CSV file

    year = 2025
    df = scrape_season(year)  # Example for scraping the 2023 season
    df.to_csv(f"../data/nfl_betting_{year}.csv", index=False)
