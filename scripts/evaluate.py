import sys
import os

# ✅ Add the project root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.preprocess import load_and_preprocess

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import pickle
import pandas as pd

print("📊 Loading data...")
X, y = load_and_preprocess()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

print("📦 Loading model...")
with open("models/rf_model.pkl", "rb") as f:
    model = pickle.load(f)

print("🔍 Making predictions...")
y_pred = model.predict(X_test)

print("\n📈 Classification Report:")
print(classification_report(y_test, y_pred))

print("\n🔲 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
