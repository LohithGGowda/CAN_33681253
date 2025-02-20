import pandas as pd

# Load the filtered CSV file
file_path = r"C:\Users\nisar\OneDrive\Desktop\fertilizationtreatment\sucessrateprediction\filtered_merged_data.csv"
df = pd.read_csv(file_path)

# Filter data for rows with "Number of Live Births" 0, 1, or 2
df_live_births = df[df['Number of Live Births'].isin([0, 1, 2])]

# Count occurrences before balancing
live_birth_counts = df_live_births['Number of Live Births'].value_counts()

print("Live Birth counts before balancing:")
print(live_birth_counts)

# Use the minimum count to balance each group of "Number of Live Births"
min_count = live_birth_counts.min()

# Create the balanced dataset by undersampling or oversampling based on the minimum count
balanced_live_births_df = pd.concat([
    df_live_births[df_live_births['Number of Live Births'] == 0].sample(min_count, replace=True),
    df_live_births[df_live_births['Number of Live Births'] == 1].sample(min_count, replace=True),
    df_live_births[df_live_births['Number of Live Births'] == 2].sample(min_count, replace=True)
])

# Balance the rest of the dataset (if needed) or merge back with other data
# Assuming 'df' still contains other rows, you could merge it back like this:
balanced_df = pd.concat([balanced_live_births_df, df[df['Number of Live Births'] > 2]])

# Fill missing values in the second column based on the first column (forward fill or backfill)
balanced_df['Date patient started trying to become pregnant OR date of last pregnancy'] = balanced_df.groupby('Patient Age at Treatment')['Date patient started trying to become pregnant OR date of last pregnancy'].transform(lambda x: x.ffill().bfill())

# Convert the second column to integer (after filling missing values)
balanced_df['Date patient started trying to become pregnant OR date of last pregnancy'] = balanced_df['Date patient started trying to become pregnant OR date of last pregnancy'].astype('Int64')  # 'Int64' handles NaN as missing

# Fill missing values for "Egg Donor Age at Registration" and "Sperm Donor Age at Registration" using forward or backfill
balanced_df['Egg Donor Age at Registration'] = balanced_df['Egg Donor Age at Registration'].transform(lambda x: x.ffill().bfill())
balanced_df['Sperm Donor Age at Registration'] = balanced_df['Sperm Donor Age at Registration'].transform(lambda x: x.ffill().bfill())

# Count the occurrences after balancing
balanced_counts = balanced_df['Number of Live Births'].value_counts()

print("\nLive Birth counts after balancing:")
print(balanced_counts)

# Save the balanced dataset to a new CSV file
output_path_balanced = r"C:\Users\nisar\OneDrive\Desktop\fertilizationtreatment\sucessrateprediction\balanced_filtered_data.csv"
balanced_df.to_csv(output_path_balanced, index=False)

print(f"\nBalanced data saved to: {output_path_balanced}")
