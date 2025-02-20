import pandas as pd

# Load the filtered CSV file
file_path = r"C:\Users\nisar\OneDrive\Desktop\fertilizationtreatment\filtered_data.csv"
df = pd.read_csv(file_path)

# Count the occurrences of IVF, ICSI, and DI
treatment_counts = df['Specific treatment type'].value_counts()

# Print the count of each treatment type
print("Treatment counts before balancing:")
print(treatment_counts)

# Balance the dataset by undersampling or oversampling (using the minimum count)
min_count = treatment_counts.min()

# Undersample or oversample to balance the dataset
balanced_df = pd.concat([
    df[df['Specific treatment type'] == 'IVF'].sample(min_count, replace=True),
    df[df['Specific treatment type'] == 'ICSI'].sample(min_count, replace=True),
    df[df['Specific treatment type'] == 'DI'].sample(min_count, replace=True)
])

# Fill missing values in the second column based on the first column (forward fill or backfill)
balanced_df['Date patient started trying to become pregnant OR date of last pregnancy'] = balanced_df.groupby('Patient Age at Treatment')['Date patient started trying to become pregnant OR date of last pregnancy'].transform(lambda x: x.ffill().bfill())

# Convert the second column to integer (after filling missing values)
balanced_df['Date patient started trying to become pregnant OR date of last pregnancy'] = balanced_df['Date patient started trying to become pregnant OR date of last pregnancy'].astype('Int64')  # 'Int64' handles NaN as missing

# Count the occurrences after balancing
balanced_counts = balanced_df['Specific treatment type'].value_counts()

# Print the count of each treatment type after balancing
print("\nTreatment counts after balancing:")
print(balanced_counts)

# Save the balanced dataset to a new CSV file
output_path_balanced = r"C:\Users\nisar\OneDrive\Desktop\fertilizationtreatment\balanced_filtered_data.csv"
balanced_df.to_csv(output_path_balanced, index=False)

print(f"\nBalanced data saved to: {output_path_balanced}")
