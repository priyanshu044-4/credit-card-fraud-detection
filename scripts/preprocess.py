# scripts/preprocess.py
import os
import gdown

def download_dataset():
    url = "https://drive.google.com/uc?id=1GiarLLxigYgnZSdG4carSthLbN4iMS_A"
    output_path = "data/creditcard.csv"
    os.makedirs("data", exist_ok=True)

    if not os.path.exists(output_path):
        print("📥 Downloading dataset from Google Drive...")
        gdown.download(url, output_path, quiet=False)
    else:
        print("✅ Dataset already exists.")

def load_and_preprocess():
    download_dataset()
    import pandas as pd
    df = pd.read_csv("data/creditcard.csv")
    # Add your preprocessing here
    return df
