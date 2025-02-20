import pandas as pd
import re

# Define the columns to keep
columns_to_keep = [
    "Patient Age at Treatment",
    "Date patient started trying to become pregnant OR date of last pregnancy",
    "Type of Infertility - Female Primary",
    "Type of Infertility - Female Secondary",
    "Type of Infertility - Male Primary",
    "Type of Infertility - Male Secondary",
    "Cause  of Infertility - Tubal disease",
    "Cause of Infertility - Ovulatory Disorder",
    "Cause of Infertility - Male Factor",
    "Cause of Infertility - Patient Unexplained",
    "Cause of Infertility - Endometriosis",
    "Cause of Infertility - Cervical factors",
    "Cause of Infertility - Female Factors",
    "Cause of Infertility - Partner Sperm Concentration",
    "Cause of Infertility -  Partner Sperm Morphology",
    "Causes of Infertility - Partner Sperm Motility",
    "Cause of Infertility -  Partner Sperm Immunological factors",
    "Specific treatment type"
]

# Load the CSV file
file_path = r"C:\Users\nisar\OneDrive\Desktop\fertilizationtreatment\merged_data.csv"
df = pd.read_csv(file_path, usecols=columns_to_keep)

# Process the 'Patient Age at Treatment' column to extract the lower bound of the age range
df['Patient Age at Treatment'] = df['Patient Age at Treatment'].apply(lambda x: int(x.split('-')[0]) if isinstance(x, str) and '-' in x else x)

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

# Save the filtered data
output_path = r"C:\Users\nisar\OneDrive\Desktop\fertilizationtreatment\filtered_data.csv"
df.to_csv(output_path, index=False)

print(f"Filtered data saved to: {output_path}")
