"""HTML table extraction utility using BeautifulSoup.

Provides functionality to parse HTML files and extract table data.
This is a utility script for data extraction tasks.
"""
from pathlib import Path

from bs4 import BeautifulSoup

file_path = ""

# Check if the file exists
file = Path(file_path)
if file.exists():
    try:
        with file.open(encoding="utf-8") as f:
            content = f.read()

        soup = BeautifulSoup(content, "html.parser")

        # Find all tables
        tables = soup.find_all("table")


        # Process each table
        for _i, table in enumerate(tables):

            # Get all headers (if any)
            headers = [
                th.get_text(strip=True)
                for th in table.find_all(["th", "td"])
                if th.find_parent("tr") == table.find("tr")
            ]

            # Alternative method: look for the first row that seems to contain headers
            rows = table.find_all("tr")
            if rows:
                first_row = rows[0]
                headers = [
                    cell.get_text(strip=True)
                    for cell in first_row.find_all(["th", "td"])
                ]


    except Exception:
        pass
else:
    pass
