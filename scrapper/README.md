# Python Data Scraper

This project is a Python data scraper that extracts betting trends data from a specified URL using Selenium and converts it into a pandas DataFrame. The data is sourced from a dynamically generated HTML table.

## Project Structure

```
python-data-scraper
├── data_scrape.py
├── requirements.txt
└── README.md
```

## Requirements

To run this project, you need to have Python installed on your machine. The necessary packages are listed in `requirements.txt`.

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd python-data-scraper
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

To run the data scraper, execute the following command in your terminal:

```
python data_scrape.py
```

This will start the scraper, navigate to the specified URL, and extract the data from the HTML table with the ID `custom-filter-table`. The data will be converted into a pandas DataFrame for further analysis.

## License

This project is licensed under the MIT License.