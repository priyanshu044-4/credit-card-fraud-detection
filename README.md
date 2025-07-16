# 💳 Credit Card Fraud Detection API & Frontend App

A powerful machine learning application to detect fraudulent credit card transactions using an XGBoost model.  
🚀 Live and accessible via both REST API and an intuitive Streamlit frontend.

---

## 🔗 Live Demo

- **🧠 Backend API:** [https://credit-card-fraud-api-0vy3.onrender.com](https://credit-card-fraud-api-0vy3.onrender.com)
- **🎨 Frontend App:** [https://credit-card-fraud-detection-dejcyumsajbhayrnghk9xs.streamlit.app/](https://credit-card-fraud-detection-dejcyumsajbhayrnghk9xs.streamlit.app/)

---

## 📌 Features

- ✅ Real-time fraud prediction via API
- 📊 Probabilistic confidence score
- 🌐 Streamlit web interface for end-users
- 🔒 Model securely hosted via Google Drive
- 🧠 Trained using XGBoost for high accuracy

---

## 🧠 Model Information

- **Algorithm:** XGBoost Classifier
- **Dataset:** [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- **Input Features:**  
  - V1 to V28 (PCA-transformed)
  - `Amount_Norm` (normalized transaction amount)

---

## 📦 Project Structure

📁 CreditCard
├── api/
│ ├── app.py # Flask backend
│ └── models/xgb_model.pkl # Downloaded model (ignored in Git)
├── scripts/
│ └── preprocess.py # Dataset preprocessing
├── data/
│ └── creditcard.csv # CSV (linked via Google Drive)
├── frontend/
│ └── app.py # Streamlit frontend
└── requirements.txt # All dependencies


## 🚀 How to Use

### ▶️ 1. Predict via API

**Endpoint:** `POST /predict`  
**URL:** `https://credit-card-fraud-api-0vy3.onrender.com/predict`

#### 🔄 Sample Request (Python)

```python
import requests

url = "https://credit-card-fraud-api-0vy3.onrender.com/predict"
payload = {
    "V1": -1.359807, "V2": -0.072781, ..., "V28": 0.02187,
    "Amount_Norm": 0.244
}

res = requests.post(url, json=payload)
print(res.json())
✅ Sample Response
json
Copy
Edit
{
  "prediction": 0,
  "fraud_probability": 0.0342,
  "label": "Legit"
}
🖥️ 2. Use Streamlit Web App
Visit the live Streamlit frontend:
📍 https://credit-card-fraud-detection-dejcyumsajbhayrnghk9xs.streamlit.app/

Fill the form and get prediction results instantly!

🔧 Tech Stack
Layer	Tools Used
Frontend	Streamlit
Backend	Flask + Gunicorn
Model	XGBoost
Hosting	Render.com + Google Drive (model)
Language	Python

📂 Setup Locally
bash
Copy
Edit
git clone https://github.com/yourusername/credit-card-fraud-detection.git
cd credit-card-fraud-detection

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

pip install -r requirements.txt

# Run API
cd api
python app.py

# Run frontend
cd ../frontend
streamlit run app.py
⚠️ Notes
creditcard.csv is not pushed to GitHub (size >100MB).
Access it via this Drive link.

XGBoost model is downloaded at runtime from Google Drive.

🤝 Contributing
Pull requests are welcome!
For major changes, please open an issue first to discuss what you would like to change.

📜 License
This project is licensed under the MIT License.

🙋‍♂️ Author
Priyanshu Kumar
🔗 LinkedIn
📬 priyanshukumar97908@gmail.com
