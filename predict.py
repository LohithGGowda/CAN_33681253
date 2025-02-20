from flask import Flask, request, jsonify, render_template_string
import joblib
import numpy as np

# Load the trained model and LabelEncoder
model_path = r"C:\Users\nisar\OneDrive\Desktop\fertilizationtreatment\decision_tree_model.pkl"
label_encoder_path = r"C:\Users\nisar\OneDrive\Desktop\fertilizationtreatment\label_encoder.pkl"

model = joblib.load(model_path)
label_encoder = joblib.load(label_encoder_path)

# Initialize Flask app
app = Flask(__name__)
treatment_info = {
    "IVF": "In Vitro Fertilization (IVF) is a process where eggs are retrieved and fertilized outside the body, then transferred back into the uterus.",
    "ICSI": "Intracytoplasmic Sperm Injection (ICSI) is an advanced IVF technique where a single sperm is injected directly into an egg.",
    "DI": "Donor Insemination (DI) is a fertility treatment using donor sperm to achieve pregnancy through artificial insemination.",
    "Unknown": "No specific treatment information available for this prediction. Please consult your fertility specialist."
}

# Define the features based on the original model
features = [
    "Patient Age at Treatment",
    "Date patient started trying to become pregnant OR date of last pregnancy",
    "Type of Infertility - Female Primary",
    "Type of Infertility - Female Secondary",
    "Type of Infertility - Male Primary",
    "Type of Infertility - Male Secondary",
    "Cause of Infertility - Tubal disease",
    "Cause of Infertility - Ovulatory Disorder",
    "Cause of Infertility - Male Factor",
    "Cause of Infertility - Patient Unexplained",
    "Cause of Infertility - Endometriosis",
    "Cause of Infertility - Cervical factors",
    "Cause of Infertility - Female Factors",
    "Cause of Infertility - Partner Sperm Concentration",
    "Cause of Infertility - Partner Sperm Morphology",
    "Causes of Infertility - Partner Sperm Motility",
    "Cause of Infertility - Partner Sperm Immunological factors"
]

# HTML content for the form and prediction output
html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Treatment Type Prediction</title>
     <style>
        body {
            font-family: 'Arial', sans-serif;
            margin: 0;
            padding: 0;
            background: url('https://static.vecteezy.com/system/resources/previews/008/029/095/large_2x/sperm-and-egg-cell-natural-fertilization-3d-illustration-on-blue-background-photo.jpg') no-repeat center center/cover;
            color: #333;
        }

        header {
            background-color: #004080;
            color: white;
            padding: 20px;
            text-align: center;
            font-size: 28px;
            border-bottom: 3px solid #fff;
        }

        main {
            width: 60%;
            margin: 30px auto;
            background: #fff;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 8px 15px rgba(0, 0, 0, 0.1);
        }

        h2 {
            color: #ffffff;
            font-size: 28px;
            margin: 0;
            padding: 10px 0;
            text-align: center;
        }

        .form-group {
            display: flex;
            justify-content: space-between;
            margin-bottom: 20px;
            align-items: center;
        }

        label {
            font-size: 16px;
            width: 45%;
            font-weight: 600;
            color: #444;
        }

        input[type="number"], input[type="radio"], input[type="checkbox"] {
            width: 50%;
            padding: 8px;
            border: 1px solid #ccc;
            border-radius: 8px;
            font-size: 16px;
            color: #444;
        }

        input[type="radio"], input[type="checkbox"] {
            width: auto;
            margin-top: 5px;
        }

        .form-group .radio-label {
            display: inline-block;
            margin-right: 15px;
            font-size: 14px;
        }

        button {
            padding: 12px;
            background-color: #008CBA;
            color: white;
            font-size: 18px;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            width: 100%;
            transition: background-color 0.3s ease;
        }

        button:hover {
            background-color: #005f73;
        }

        #result {
            margin-top: 20px;
            text-align: center;
            font-size: 20px;
            font-weight: 600;
        }

        .flex-row {
            display: flex;
            flex-wrap: wrap;
            justify-content: space-between;
        }

        .flex-row .form-group {
            width: 48%;
        }

        footer {
            background-color: #004080;
            color: white;
            padding: 10px;
            text-align: center;
            font-size: 14px;
            position: absolute;
            bottom: 0;
            width: 100%;
        }
    </style>
</head>
<body>
    <header>
        <h2>Treatment Type Prediction Form</h2>
    </header>
    <main>
        <form id="predictionForm">
           <!-- Input fields for user data -->
           <div class="form-group">
                <label for="age">Patient Age at Treatment (18-50):</label>
                <input type="number" id="age" name="age" min="18" max="50" required>
           </div>

           <div class="form-group">
                <label for="pregnancyDate">Date patient started trying to become pregnant OR date of last pregnancy (1-40):</label>
                <input type="number" id="pregnancyDate" name="pregnancyDate" min="1" max="40" required>
           </div>

           <!-- Female infertility types -->
           <div class="form-group">
                <label>Female Primary Infertility:</label>
                <input type="radio" id="femalePrimaryYes" name="femalePrimary" value="1"> Yes
                <input type="radio" id="femalePrimaryNo" name="femalePrimary" value="0" checked> No
           </div>

           <div class="form-group">
                <label>Female Secondary Infertility:</label>
                <input type="radio" id="femaleSecondaryYes" name="femaleSecondary" value="1"> Yes
                <input type="radio" id="femaleSecondaryNo" name="femaleSecondary" value="0" checked> No
           </div>

           <!-- Male infertility types -->
           <div class="form-group">
                <label>Male Primary Infertility:</label>
                <input type="radio" id="malePrimaryYes" name="malePrimary" value="1"> Yes
                <input type="radio" id="malePrimaryNo" name="malePrimary" value="0" checked> No
           </div>

           <div class="form-group">
                <label>Male Secondary Infertility:</label>
                <input type="radio" id="maleSecondaryYes" name="maleSecondary" value="1"> Yes
                <input type="radio" id="maleSecondaryNo" name="maleSecondary" value="0" checked> No
           </div>

            <!-- Causes of Infertility -->
            <div class="flex-row">
                <div class="form-group">
                    <label for="tubalDisease">Cause of Infertility - Tubal disease:</label>
                    <input type="checkbox" id="tubalDisease" name="tubalDisease" value="1">
                </div>
                <div class="form-group">
                    <label for="ovulatoryDisorder">Cause of Infertility - Ovulatory Disorder:</label>
                    <input type="checkbox" id="ovulatoryDisorder" name="ovulatoryDisorder" value="1">
                </div>
                <div class="form-group">
                    <label for="maleFactor">Cause of Infertility - Male Factor:</label>
                    <input type="checkbox" id="maleFactor" name="maleFactor" value="1">
                </div>
                <div class="form-group">
                    <label for="unexplained">Cause of Infertility - Patient Unexplained:</label>
                    <input type="checkbox" id="unexplained" name="unexplained" value="1">
                </div>
                <div class="form-group">
                    <label for="endometriosis">Cause of Infertility - Endometriosis:</label>
                    <input type="checkbox" id="endometriosis" name="endometriosis" value="1">
                </div>
                <div class="form-group">
                    <label for="cervicalFactors">Cause of Infertility - Cervical factors:</label>
                    <input type="checkbox" id="cervicalFactors" name="cervicalFactors" value="1">
                </div>
            </div>

            <div class="flex-row">
                <div class="form-group">
                    <label for="femaleFactors">Cause of Infertility - Female Factors:</label>
                    <input type="checkbox" id="femaleFactors" name="femaleFactors" value="1">
                </div>
                <div class="form-group">
                    <label for="partnerSpermConc">Cause of Infertility - Partner Sperm Concentration:</label>
                    <input type="checkbox" id="partnerSpermConc" name="partnerSpermConc" value="1">
                </div>
                <div class="form-group">
                    <label for="partnerSpermMorph">Cause of Infertility - Partner Sperm Morphology:</label>
                    <input type="checkbox" id="partnerSpermMorph" name="partnerSpermMorph" value="1">
                </div>
                <div class="form-group">
                    <label for="partnerSpermMotility">Cause of Infertility - Partner Sperm Motility:</label>
                    <input type="checkbox" id="partnerSpermMotility" name="partnerSpermMotility" value="1">
                </div>
                <div class="form-group">
                    <label for="partnerSpermImmuno">Cause of Infertility - Partner Sperm Immunological Factors:</label>
                    <input type="checkbox" id="partnerSpermImmuno" name="partnerSpermImmuno" value="1">
                </div>
            </div>

           <!-- Submit button -->
           <div class="form-group">
                <button type="submit">Predict Treatment Type</button>
           </div>

       </form>

       <!-- Result display -->
       <div id="result">
           <h3>Predicted Treatment Type:<span id="predictionResult"></span></h3>
       </div>

   </main>

   <!-- JavaScript to handle form submission -->
   <script>
       document.getElementById('predictionForm').addEventListener('submit', async function(event) {
           event.preventDefault(); // Prevent page reload

           // Collect input data
           const formData = {
               age: document.getElementById('age').value,
               pregnancyDate: document.getElementById('pregnancyDate').value,
               femalePrimary: document.querySelector('input[name=femalePrimary]:checked').value,
               femaleSecondary: document.querySelector('input[name=femaleSecondary]:checked').value,
               malePrimary: document.querySelector('input[name=malePrimary]:checked').value,
               maleSecondary: document.querySelector('input[name=maleSecondary]:checked').value,
               tubalDisease : document.getElementById('tubalDisease').checked ? '1':'0',
               ovulatoryDisorder : document.getElementById('ovulatoryDisorder').checked ? '1':'0',
               maleFactor : document.getElementById('maleFactor').checked ? '1':'0',
               unexplained : document.getElementById('unexplained').checked ? '1':'0',
               endometriosis : document.getElementById('endometriosis').checked ? '1':'0',
               cervicalFactors : document.getElementById('cervicalFactors').checked ? '1':'0',
               femaleFactors : document.getElementById('femaleFactors').checked ? '1':'0',
               partnerSpermConc : document.getElementById('partnerSpermConc').checked ? '1':'0',
               partnerSpermMorph : document.getElementById('partnerSpermMorph').checked ? '1':'0',
               partnerSpermMotility : document.getElementById('partnerSpermMotility').checked ? '1':'0',
               partnerSpermImmuno : document.getElementById('partnerSpermImmuno').checked ? '1':'0'
           };

           // Send data to the server for prediction
           const response = await fetch('/predict', {
               method:'POST',
               headers:{
                   'Content-Type': 'application/json'
               },
               body : JSON.stringify(formData)
           });

           const result = await response.json();
           if (result.error) {
              alert(result.error);
              return; 
          }
           
          // Display prediction result
          document.getElementById('predictionResult').textContent = result.prediction + ": " + result.description;

       });
   </script>

</body>
</html>
"""

@app.route('/predict', methods=['POST'])
def predict():
   data = request.get_json()
   print("Received data:", data) 

   if not data:
       return jsonify({"error": "No data received!"}),400

   try:
       input_data = [
          int(data['age']),
          int(data['pregnancyDate']),
          int(data['femalePrimary']),
          int(data['femaleSecondary']),
          int(data['malePrimary']),
          int(data['maleSecondary']),
          int(data.get('tubalDisease',0)),
          int(data.get('ovulatoryDisorder',0)),
          int(data.get('maleFactor',0)),
          int(data.get('unexplained',0)),
          int(data.get('endometriosis',0)),
          int(data.get('cervicalFactors',0)),
          int(data.get('femaleFactors',0)),
          int(data.get('partnerSpermConc',0)),
          int(data.get('partnerSpermMorph',0)),
          int(data.get('partnerSpermMotility',0)),
          int(data.get('partnerSpermImmuno',0))
       ]

       input_data = np.array(input_data).reshape(1,-1)
       prediction = model.predict(input_data)
       prediction_label = label_encoder.inverse_transform(prediction)[0]
       treatment_description = treatment_info.get(prediction_label,treatment_info["Unknown"])

       return jsonify({'prediction': prediction_label,'description': treatment_description})
   except Exception as e:
       return jsonify({"error": f"Error processing input data:{str(e)}"}),400

# Define the route to render the HTML form
@app.route('/')
def index():
   return render_template_string(html_content)

if __name__ == '__main__':
   app.run(debug=True)
