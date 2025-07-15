from flask import Flask, request, jsonify
import pickle
import numpy as np
import os
import sys
import requests
import io

# Add root to path to allow script imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.preprocess import load_and_preprocess  # Optional, in case used later

app = Flask(__name__)

# 🔗 Load model from Google Drive
model_url = "https://drive.google.com/uc?export=download&id=1GiarLLxigYgnZSdG4carSthLbN4iMS_A"
try:
    print("📥 Downloading model from Google Drive...")
    response = requests.get(model_url)
    response.raise_for_status()  # Raise error if download fails
    model = pickle.load(io.BytesIO(response.content))
    print("✅ Model loaded successfully.")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    model = None

@app.route('/')
def home():
    return "✅ Credit Card Fraud Detection API is live with XGBoost!"

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

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
