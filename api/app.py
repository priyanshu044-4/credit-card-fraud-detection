from flask import Flask, request, jsonify
import pickle
import numpy as np
import os
import sys
import gdown

# Add root to path to allow script imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

app = Flask(__name__)

# 🔗 Google Drive direct file ID for the model
gdown_url = "https://drive.google.com/uc?id=1GiarLLxigYgnZSdG4carSthLbN4iMS_A"
model_path = "models/xgb_model.pkl"
os.makedirs("models", exist_ok=True)

# 🔁 Download model if not exists
try:
    if not os.path.exists(model_path):
        print("📥 Downloading model from Google Drive...")
        gdown.download(gdown_url, model_path, quiet=False)
    else:
        print("✅ Model already exists. Skipping download.")

    with open(model_path, 'rb') as f:
        model = pickle.load(f)
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
