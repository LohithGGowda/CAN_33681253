import os
import requests
import pandas as pd
from bs4 import BeautifulSoup
from pyxlsb import open_workbook

# URL of the webpage to scrape
BASE_URL = "https://www.hfea.gov.uk"
PAGE_URL = "https://www.hfea.gov.uk/about-us/data-research/"

# Create a directory to store downloaded files
os.makedirs("xlsb_files", exist_ok=True)

def get_xlsb_links():
    """Scrape the webpage and return a list of .xlsb file URLs."""
    response = requests.get(PAGE_URL)
    response.raise_for_status()
    
    soup = BeautifulSoup(response.text, 'html.parser')
    links = soup.find_all("a", class_="hfea-v2 teal-white")
    
    file_links = []
    for link in links:
        href = link.get("href")
        if href.endswith(".xlsb"):
            file_links.append(BASE_URL + href if href.startswith("/") else href)
    
    return file_links

def download_xlsb(file_links):
    """Download .xlsb files and save locally."""
    file_paths = []
    for link in file_links:
        filename = os.path.join("xlsb_files", os.path.basename(link))
        response = requests.get(link)
        response.raise_for_status()
        with open(filename, "wb") as f:
            f.write(response.content)
        file_paths.append(filename)
    return file_paths

def read_xlsb(file_path):
    """Read an .xlsb file into a Pandas DataFrame."""
    with open_workbook(file_path) as wb:
        sheets = wb.sheets
        for sheet in sheets:
            df = pd.read_excel(file_path, sheet_name=sheet, engine="pyxlsb")
            return df  # Assuming one sheet per file

def merge_xlsb_to_csv(file_paths, output_csv="merged_data.csv"):
    """Merge all .xlsb files into a single CSV."""
    dfs = [read_xlsb(file) for file in file_paths]
    merged_df = pd.concat(dfs, ignore_index=True)
    merged_df.to_csv(output_csv, index=False)
    print(f"Saved merged data to {output_csv}")

# Execute the steps
xlsb_links = get_xlsb_links()
if xlsb_links:
    file_paths = download_xlsb(xlsb_links)
    merge_xlsb_to_csv(file_paths)
else:
    print("No .xlsb files found on the webpage.")
