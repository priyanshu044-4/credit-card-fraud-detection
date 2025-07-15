import os
import gdown
import pandas as pd

def load_data():
    url = "https://drive.google.com/uc?export=download&id=1GiarLLxigYgnZSdG4carSthLbN4iMS_A"
    csv_path = "data/creditcard.csv"

    if not os.path.exists(csv_path):
        os.makedirs("data", exist_ok=True)
        print("📥 Downloading dataset from Google Drive...")
        gdown.download(url, csv_path, quiet=False)
    else:
        print("✅ Dataset already exists.")

    df = pd.read_csv(csv_path)
    return df
