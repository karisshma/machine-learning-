from flask import Flask, request, jsonify
import joblib
import pandas as pd # Import pandas for DataFrame conversion

# Initialize Flask application
app = Flask(__name__)

# Load the pre-trained model
model = joblib.load('logistic_regression_model.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    # 1. Retrieve the JSON data sent with the request
    data = request.get_json(force=True, silent=True)

    # 3. Add error handling to check if the data is present and in the expected format.
    if data is None:
        return jsonify({"error": "No JSON data provided"}), 400

    # Ensure data is a dictionary or a list containing a single dictionary
    if not isinstance(data, (dict, list)):
        return jsonify({"error": "Invalid JSON format. Expected a dictionary or a list of dictionaries."}), 400

    # Convert the data into a pandas DataFrame
    try:
        # If single instance prediction (dict)
        if isinstance(data, dict):
            df_input = pd.DataFrame([data])
        # If multiple instances prediction (list of dicts)
        elif isinstance(data, list):
            df_input = pd.DataFrame(data)

        # Ensure the column names match the features used during training
        expected_features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
        if not all(feature in df_input.columns for feature in expected_features):
            missing_features = [feature for feature in expected_features if feature not in df_input.columns]
            return jsonify({"error": f"Missing features in input data: {', '.join(missing_features)}"}), 400

        # Reorder columns to match model's expected input order if necessary (optional but good practice)
        df_input = df_input[expected_features]

    except Exception as e:
        return jsonify({"error": f"Error processing input data: {str(e)}"}), 400

    # Make predictions
    predictions = model.predict(df_input)
    probabilities = model.predict_proba(df_input)

    # Format the predictions into a list of dictionaries
    results = []
    for i in range(len(predictions)):
        result = {
            "prediction": int(predictions[i]),
            "probability_of_diabetes": float(probabilities[i][1]) # Probability of class 1 (diabetes)
        }
        results.append(result)

    return jsonify(results)

print("Flask /predict endpoint updated to make predictions and return results.")
