import sys
import os

# Add root path for module import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.preprocess import load_and_preprocess
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import pickle

print("🔄 Loading and preprocessing data...")
X, y = load_and_preprocess()
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

print("🧠 Training XGBoost model...")
scale_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
model = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_weight,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)
model.fit(X_train, y_train)

model_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
os.makedirs(model_dir, exist_ok=True)

model_path = os.path.join(model_dir, 'xgb_model.pkl')
with open(model_path, 'wb') as f:
    pickle.dump(model, f)

print(f"✅ XGBoost model trained and saved to {model_path}")
