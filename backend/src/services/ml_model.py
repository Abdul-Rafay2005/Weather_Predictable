import joblib
import numpy as np
from src.services import seasonal_data

# Load trained models into a dictionary
models = {}
for target in ["very_hot", "very_cold", "very_wet", "very_windy", "very_uncomfortable"]:
    try:
        models[target] = joblib.load(f"backend/models/{target}_model.joblib")
    except Exception as e:
        print(f"⚠️ Could not load model {target}: {e}")

def predict_probabilities(lat, lon, date_str):
    # 1️⃣ Try fetching seasonal features
    features = seasonal_data.get_seasonal_estimates(lat, lon, date_str)
    
    # 2️⃣ Fallback if features missing
    if features is None or "error" in features:
        features = seasonal_data.get_default_features(lat, lon, date_str)
        if features is None:
            # If still None, return all nulls
            return {
                key: None for key in ["very_hot_percent", "very_cold_percent", 
                                      "very_wet_percent", "very_windy_percent", 
                                      "very_uncomfortable_percent"]
            } | {"notes": "ML model prediction failed: no seasonal features available"}

    X = np.array([features])  # ensure 2D array

    try:
        probs = {}
        for target in models:
            model = models[target]
            # predict probability for class '1' (event happens)
            probs[f"{target}_percent"] = model.predict_proba(X)[0][1] * 100
        probs["notes"] = "ML model probabilities"
        return probs
    except Exception as e:
        return {
            key: None for key in ["very_hot_percent", "very_cold_percent", 
                                  "very_wet_percent", "very_windy_percent", 
                                  "very_uncomfortable_percent"]
        } | {"notes": f"ML model prediction failed: {e}"}
