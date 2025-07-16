import os
import gdown

def download_model():
    url = "https://drive.google.com/uc?id=1GiarLLxigYgnZSdG4carSthLbN4iMS_A"
    output_path = "models/xgb_model.pkl"
    os.makedirs("models", exist_ok=True)

    if not os.path.exists(output_path):
        print("📥 Downloading model from Google Drive...")
        gdown.download(url, output_path, quiet=False)
    else:
        print("✅ Model already exists.")
