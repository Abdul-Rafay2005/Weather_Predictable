# backend/src/services/ml_predict.py
import joblib
import os
import pandas as pd

MODEL_DIR = "backend/models"

# Load models once
models = {}
for label in ["very_hot", "very_cold", "very_wet", "very_windy", "very_uncomfortable"]:
    path = os.path.join(MODEL_DIR, f"{label}_model.joblib")
    if os.path.exists(path):
        models[label] = joblib.load(path)

def predict_conditions(features: dict):
    """
    features = { "tmax": .., "tmin": .., "precip": .., "wind": .., "rh": .., "doy": .. }
    Returns probabilities for each label.
    """
    X = pd.DataFrame([features])
    results = {}
    for label, model in models.items():
        proba = model.predict_proba(X)[0][1]  # probability of class 1
        results[f"{label}_percent"] = round(proba * 100, 1)
    return results
