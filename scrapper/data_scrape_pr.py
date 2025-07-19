import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import argparse
import time
from io import StringIO

def scrape_data_pr(url, week=None):
    # Use Safari WebDriver (Safari must have 'Allow Remote Automation' enabled in Develop menu)
    driver = webdriver.Safari()

    try:
        driver.get(url)
        time.sleep(2)  # Wait for the page to load

        # Wait until the table is present or timeout after 10 seconds
        try:
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.ID, 'DataTables_Table_0'))
            )
        except Exception as e:
            return pd.DataFrame()

        # Locate the table by its ID
        table = driver.find_element(By.ID, 'DataTables_Table_0')
        table_html = table.get_attribute('outerHTML')

        # Use pandas to parse the HTML table, which will include headers automatically
        df = pd.read_html(StringIO(table_html))[0]
        # Remove rows where the first column is NaN (if any)
        first_col = df.columns[0]
        df = df.dropna(subset=[first_col])
        # Flatten multi-level columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ['_'.join([str(i) for i in col if str(i) != 'nan']) for col in df.columns.values]
        # Rename columns starting with 'Unnamed' to their level 1 name (if available)
        df.columns = [
            col if not str(col).startswith('Unnamed') else str(col).split('_', 1)[-1] for col in df.columns
        ]
        # Rename columns starting with 'level' by splitting by '_' and taking the last value
        df.columns = [
            str(col).split('_')[-1] if str(col).startswith('level') else col for col in df.columns
        ]
        # Add Week column
        if week is not None:
            df['Week'] = week
        return df

    finally:
        driver.quit()

def scrape_season_pr(year):
    """
    Scrape all weeks (1-18) for a given year and merge into a single DataFrame.
    """
    url_template = "https://betiq.teamrankings.com/nfl/predictions/{year}/?week=week-{week}"
    dfs = []
    for week in range(1, 19):
        url = url_template.format(year=year, week=week)
        print(f"Scraping for year {year}, week {week} --> {url}")
        df = scrape_data_pr(url, week=week)
        if not df.empty:
            dfs.append(df)
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    else:
        return pd.DataFrame()

if __name__ == '__main__':
    for year in range(2000, 2025):
        df = scrape_season_pr(year)
        df.to_csv(f'../data/nfl_rankings_{year}.csv', index=False)
