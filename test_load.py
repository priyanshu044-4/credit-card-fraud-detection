print("✅ test_load.py started")

try:
    import pandas as pd
    print("📦 pandas imported")
    
    print("📂 Attempting to read first 100 rows of CSV...")
    df = pd.read_csv("data/creditcard.csv", nrows=100)
    print(f"✅ Read complete: {df.shape[0]} rows, {df.shape[1]} columns")

except Exception as e:
    print("❌ Error occurred:", e)
