from flask import Flask, request, jsonify
import pickle
import numpy as np
import os
import sys

# Add root to path to allow script imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.preprocess import load_and_preprocess

app = Flask(__name__)

# Load the upgraded model (XGBoost or best available)
model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'xgb_model.pkl'))
with open(model_path, 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return "✅ Credit Card Fraud Detection API is live with XGBoost!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json

        # Extract features V1-V28 + Amount_Norm
        features = [data.get(f"V{i}") for i in range(1, 29)] + [data.get("Amount_Norm")]

        # Validate input
        if None in features:
            return jsonify({
                "error": "Missing one or more required fields (V1 to V28 + Amount_Norm)"
            }), 400

        # Convert to float array
        features = np.array(features).reshape(1, -1)

        # Make prediction
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0][1]  # Probability of fraud (class 1)

        return jsonify({
            "prediction": int(prediction),
            "fraud_probability": round(float(probability), 4),
            "label": "Fraud" if prediction == 1 else "Legit"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
