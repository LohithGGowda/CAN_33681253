import pandas as pd
import re
import os

# Define the path to your CSV file
file_path = r"C:\Users\nisar\OneDrive\Desktop\fertilizationtreatment\sucessrateprediction\merged_data.csv"

# Load the CSV file into a DataFrame
data = pd.read_csv(file_path, low_memory=False)

# Define the columns you want to keep
columns_to_keep = [
    "Patient Age at Treatment", 
    "Date patient started trying to become pregnant OR date of last pregnancy", 
    "Total Number of Previous cycles, Both IVF and DI", 
    "Total number of previous pregnancies, Both IVF and DI", 
    "Type of Infertility - Female Primary", 
    "Type of Infertility - Female Secondary", 
    "Type of Infertility - Male Primary", 
    "Type of Infertility - Male Secondary",  
    "Cause  of Infertility - Tubal disease", 
    "Cause of Infertility - Ovulatory Disorder", 
    "Cause of Infertility - Male Factor", 
    "Cause of Infertility - Endometriosis", 
    "Cause of Infertility - Cervical factors", 
    "Cause of Infertility - Female Factors", 
    "Egg Donor Age at Registration", 
    "Sperm Donor Age at Registration", 
    "Specific treatment type",
    "Number of Live Births", 
]

# Filter the DataFrame to keep only the specified columns
df = data[columns_to_keep]

# Process the 'Patient Age at Treatment' column to extract the lower bound of the age range
df['Patient Age at Treatment'] = df['Patient Age at Treatment'].apply(
    lambda x: int(x.split('-')[0]) if isinstance(x, str) and '-' in x else x
)

# Function to extract the lower bound of age ranges (e.g., "Between 26 and 30" -> 26)
def extract_age(value):
    if isinstance(value, str):
        # Check if the value contains an age range (e.g., "Between 26 and 30")
        match = re.search(r'Between (\d+)', value)
        if match:
            return int(match.group(1))  # Return the lower bound of the range
    # Return the original value if it's a single number or invalid format
    return value

# Apply the function to clean the 'Egg Donor Age at Registration' and 'Sperm Donor Age at Registration' columns
df['Egg Donor Age at Registration'] = df['Egg Donor Age at Registration'].apply(extract_age)
df['Sperm Donor Age at Registration'] = df['Sperm Donor Age at Registration'].apply(extract_age)

# Function to clean the 'Specific treatment type' column
def clean_treatment_type(value):
    if isinstance(value, str):
        # Use regex to find and retain only IVF, ICSI, or DI
        match = re.search(r'\b(IVF|ICSI|DI)\b', value, re.IGNORECASE)
        if match:
            return match.group(0).upper()  # Keep only the matching term in uppercase
    return value

# Apply the function to clean the treatment type column
df['Specific treatment type'] = df['Specific treatment type'].apply(clean_treatment_type)

# Filter the data to keep only rows with valid treatment types (IVF, ICSI, DI)
df = df[df['Specific treatment type'].isin(['IVF', 'ICSI', 'DI'])]

# Define the directory where the filtered data will be saved
directory = r"C:\Users\nisar\OneDrive\Desktop\fertilizationtreatment\sucessrateprediction"

# Create the directory if it doesn't exist
if not os.path.exists(directory):
    os.makedirs(directory)

# Save the filtered data to a new CSV file
filtered_file_path = os.path.join(directory, 'filtered_merged_data.csv')
df.to_csv(filtered_file_path, index=False)

print(f"Filtered data saved to '{filtered_file_path}'")
