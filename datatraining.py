import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
import joblib

# Load the dataset
data = pd.read_csv(r'C:\Users\nisar\OneDrive\Desktop\fertilizationtreatment\sucessrateprediction\balanced_filtered_data.csv')

# Select features and target variable
features = [
    'Patient Age at Treatment',
    'Date patient started trying to become pregnant OR date of last pregnancy',
    'Total Number of Previous cycles, Both IVF and DI',
    'Total number of previous pregnancies, Both IVF and DI',
    'Type of Infertility - Female Primary',
    'Type of Infertility - Female Secondary',
    'Type of Infertility - Male Primary',
    'Type of Infertility - Male Secondary',
    'Cause  of Infertility - Tubal disease',
    'Cause of Infertility - Ovulatory Disorder',
    'Cause of Infertility - Male Factor',
    'Cause of Infertility - Endometriosis',
    'Cause of Infertility - Cervical factors',
    'Cause of Infertility - Female Factors',
    'Egg Donor Age at Registration',
    'Sperm Donor Age at Registration',
    'Specific treatment type'
]

target = 'Number of Live Births'

# Prepare the features and target variables
X = data[features]
y = data[target]

# Convert categorical variables into dummy/indicator variables
X = pd.get_dummies(X, drop_first=True)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Decision Tree model
model = DecisionTreeRegressor()

# Train the model
model.fit(X_train, y_train)

# Predict and evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Save the model
joblib.dump(model, 'decision_tree_model1.pkl')
print('Model saved as decision_tree_model1.pkl')
