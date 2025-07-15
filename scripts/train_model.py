print("✅ train_model.py started")

# STEP 1: Import libraries
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    import pickle
    import os
    from preprocess import load_and_preprocess

    print("✅ All libraries imported successfully")
except Exception as e:
    print("❌ Import failed:", e)
    exit()

# STEP 2: Load and preprocess
try:
    print("🔄 Loading and preprocessing data...")
    X, y = load_and_preprocess()
    print(f"✅ Data loaded: {X.shape[0]} rows, {X.shape[1]} features")
except Exception as e:
    print("❌ Failed during preprocessing:", e)
    exit()

# STEP 3: Split data
try:
    print("📊 Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
    print("✅ Data split complete")
except Exception as e:
    print("❌ Split failed:", e)
    exit()

# STEP 4: Train model
try:
    print("🧠 Training Random Forest model...")
    model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)
    print("✅ Model training complete")
except Exception as e:
    print("❌ Training failed:", e)
    exit()

# STEP 5: Save model
try:
    print("💾 Saving model...")
    os.makedirs('models', exist_ok=True)
    with open('models/rf_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("✅ Model saved to models/rf_model.pkl")
except Exception as e:
    print("❌ Saving model failed:", e)
    exit()
