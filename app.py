from flask import Flask, request, jsonify
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load the trained Decision Tree model
model_path = r"C:\Users\nisar\OneDrive\Desktop\fertilizationtreatment\decision_tree_model.pkl"
model = joblib.load(model_path)

# Define the feature columns
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

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    input_data = pd.DataFrame([data])

    # Load LabelEncoder (or retrain if needed)
    label_encoder = LabelEncoder()

    # Ensure all features are encoded properly
    for col in features:
        if input_data[col].dtype == 'object':
            input_data[col] = label_encoder.fit_transform(input_data[col].astype(str))

    # Make predictions using the trained model
    prediction = model.predict(input_data[features])

    # Return the prediction as JSON
    return jsonify({'predictedTreatmentType': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
