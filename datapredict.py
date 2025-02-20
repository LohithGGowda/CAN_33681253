import pandas as pd
from flask import Flask, request, jsonify, render_template
import joblib

# Load the trained model
model = joblib.load('decision_tree_model1.pkl')

# Define the feature list that matches the model's expected input
features = [
    'Patient Age at Treatment',
    'Date patient started trying to become pregnant OR date of last pregnancy',
    'Total Number of Previous cycles, Both IVF and DI',
    'Total number of previous pregnancies, Both IVF and DI',
    'Type of Infertility - Female Primary',
    'Type of Infertility - Female Secondary',
    'Type of Infertility - Male Primary',
    'Type of Infertility - Male Secondary',
    'Cause of Infertility - Tubal disease',
    'Cause of Infertility - Ovulatory Disorder',
    'Cause of Infertility - Male Factor',
    'Cause of Infertility - Endometriosis',
    'Cause of Infertility - Cervical factors',
    'Cause of Infertility - Female Factors',
    'Egg Donor Age at Registration',
    'Sperm Donor Age at Registration',
    'Specific treatment type'
]

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    """Render the home page with the form"""
    return render_template('index.html')

@app.route('/datapredict', methods=['POST'])
def predict():
    try:
        # Get data from the form (JSON)
        data = request.get_json()

        # Prepare the data for prediction (convert to list)
        input_data = [
            float(data.get('age', 0)),                              # Age of patient
            float(data.get('pregnancyDate', 0)),                      # Date patient started trying or last pregnancy
            int(data.get('totalPreviousCycles', 0)),                  # Total previous cycles
            int(data.get('totalPreviousPregnancies', 0)),             # Total previous pregnancies
            int(data.get('femalePrimary', 0)),                        # Female Primary Infertility (0/1)
            int(data.get('femaleSecondary', 0)),                      # Female Secondary Infertility (0/1)
            int(data.get('malePrimary', 0)),                          # Male Primary Infertility (0/1)
            int(data.get('maleSecondary', 0)),                        # Male Secondary Infertility (0/1)
            int(data.get('tubalDisease', 0)),                         # Tubal Disease (0/1)
            int(data.get('ovulatoryDisorder', 0)),                    # Ovulatory Disorder (0/1)
            int(data.get('maleFactor', 0)),                           # Male Factor (0/1)
            int(data.get('endometriosis', 0)),                        # Endometriosis (0/1)
            int(data.get('cervicalFactors', 0)),                      # Cervical Factors (0/1)
            int(data.get('femaleFactors', 0)),                        # Female Factors (0/1)
            float(data.get('eggDonorAge', 0)),                        # Egg Donor Age
            float(data.get('spermDonorAge', 0)),                      # Sperm Donor Age
            data.get('specificTreatmentType', 'IVF')                  # Treatment Type (IVF/DI/ICSI)
        ]

        # Create DataFrame with the input data
        input_df = pd.DataFrame([input_data], columns=features)

        # One-hot encode categorical features (Specific Treatment Type)
        input_df = pd.get_dummies(input_df, columns=['Specific treatment type'], drop_first=True)

        # Align the DataFrame with the trained model's feature names (important)
        input_df = input_df.reindex(columns=model.feature_names_in_, fill_value=0)

        # Print out the input dataframe to verify it looks correct
        print("Input DataFrame for Prediction:")
        print(input_df)

        # Make prediction using the model
        prediction = model.predict(input_df)
        
        # Print the raw prediction to debug
        print(f"Model prediction: {prediction[0]}")

        # Modify the logic for prediction output
        if prediction[0] == 0:
            # If prediction is 0 live births, return "NO"
            response = {
                'prediction': "Prediction: NO, no live births It is important to consult with a healthcare provider to understand underlying health issues, maintain a healthy lifestyle with proper diet and regular exercise to improve fertility, monitor menstrual cycles and ovulation to determine the best times for conception, and avoid smoking, excessive alcohol, and other substances that can negatively affect fertility."
            }
        else:
            # If prediction is 1 or more live births, return "YES"
            response = {
                'prediction': f"Prediction: YES, live births possible To support a healthy pregnancy, it is important to take prenatal vitamins with folic acid to aid in the baby’s development, maintain a balanced diet with essential nutrients, get regular prenatal check-ups to monitor health progress, and engage in light physical activities as recommended by your healthcare provider."
            }

        return jsonify(response)

    except Exception as e:
        # Handle any errors during prediction
        return jsonify({'error': str(e)})

# Run the app
if __name__ == '__main__':
    app.run(debug=True, port=5001)
