import os
import pandas as pd
from bs4 import BeautifulSoup

file_path = ""

# Check if the file exists
if os.path.exists(file_path):
    try:
        print("")
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        soup = BeautifulSoup(content, 'html.parser')
        
        # Find all tables
        tables = soup.find_all('table')
        
        print(f"File: {os.path.basename(file_path)}")
        print(f"Found {len(tables)} table(s)")
        
        # Process each table
        for i, table in enumerate(tables):
            print(f"\nTable {i+1}:")
            
            # Get all headers (if any)
            headers = [th.get_text(strip=True) for th in table.find_all(['th', 'td']) if th.find_parent('tr') == table.find('tr')]
            
            # Alternative method: look for the first row that seems to contain headers
            rows = table.find_all('tr')
            if rows:
                first_row = rows[0]
                headers = [cell.get_text(strip=True) for cell in first_row.find_all(['th', 'td'])]
                
                print(f"Headers (from first row): {headers}")
                print(f"Number of columns: {len(headers)}")
        
    except Exception as e:
        print(f"Failed to process {file_path}: {e}")
else:
    print(f"File does not exist: {file_path}")
