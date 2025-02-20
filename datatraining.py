import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib

# Load the balanced CSV file
file_path = r"C:\Users\nisar\OneDrive\Desktop\fertilizationtreatment\balanced_filtered_data.csv"
df = pd.read_csv(file_path)

# Strip any leading/trailing spaces from column names
df.columns = df.columns.str.strip()

# List of features to use for training
features = [
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
    "Cause of Infertility -  Partner Sperm Immunological factors"
]

# Check if all feature columns exist
missing_columns = [col for col in features if col not in df.columns]
if missing_columns:
    print(f"Missing columns: {missing_columns}")

# Handle missing values (if any)
df = df.dropna(subset=features + ["Specific treatment type"])

# Encode the target variable (Specific treatment type)
target = "Specific treatment type"
label_encoder = LabelEncoder()
df[target] = label_encoder.fit_transform(df[target])

# Convert features to numeric (if they are not already)
for col in features:
    if df[col].dtype == 'object':
        df[col] = label_encoder.fit_transform(df[col].astype(str))

# Split the dataset into features (X) and target (y)
X = df[features]
y = df[target]

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Decision Tree model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Save the trained model (optional)
model_path = r"C:\Users\nisar\OneDrive\Desktop\fertilizationtreatment\decision_tree_model.pkl"
joblib.dump(model, model_path)
joblib.dump(label_encoder, r"C:\Users\nisar\OneDrive\Desktop\fertilizationtreatment\label_encoder.pkl")

print(f"Trained model saved to: {model_path}")
