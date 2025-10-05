from fastapi import FastAPI, Query
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    accuracy_score, precision_score, recall_score, f1_score
)
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware

# ============================================================
# 1️⃣ Initialize FastAPI App
# ============================================================
app = FastAPI(title="Weather Prediction API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or specify your frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# 2️⃣ Load Dataset and Train Models
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "nyc_training_data.csv"

print("🔄 Loading dataset...")
df = pd.read_csv(DATA_PATH)
df['date'] = pd.to_datetime(df['date'])
df['month'] = df['date'].dt.month
df['doy'] = df['date'].dt.dayofyear
df['year'] = df['date'].dt.year

def get_season(month: int) -> str:
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Autumn'

df['season'] = df['month'].apply(get_season)
season_map = {'Winter': 1, 'Spring': 2, 'Summer': 3, 'Autumn': 4}
df['season_code'] = df['season'].map(season_map)
df['rain_flag'] = (df['precip'] > 0).astype(int)

# --- Train Rain Classifier ---
features = ['tmax', 'tmin', 'wind', 'rh', 'doy', 'season_code', 'year']
X = df[features]
y = df['rain_flag']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

clf = RandomForestClassifier(n_estimators=200, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("\n--- Rain Prediction Metrics ---")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))

# --- Train Regressors for tmax, tmin, rh ---
weather_models = {}
weather_results = {}
targets = ['tmax', 'tmin', 'rh']
features_for_weather = ['doy', 'season_code', 'rain_flag', 'wind', 'year']

for target in targets:
    Xw = df[[f for f in features_for_weather if f != target]]
    yw = df[target]
    Xw_train, Xw_test, yw_train, yw_test = train_test_split(Xw, yw, test_size=0.2, shuffle=False)
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(Xw_train, yw_train)
    weather_models[target] = model
    weather_results[target] = {
        'R²': r2_score(yw_test, model.predict(Xw_test)),
        'RMSE': np.sqrt(mean_squared_error(yw_test, model.predict(Xw_test))),
        'MAE': mean_absolute_error(yw_test, model.predict(Xw_test))
    }

print("\n--- MODEL PERFORMANCE SUMMARY ---")
for k, v in weather_results.items():
    print(f"{k.upper()} → R²: {v['R²']:.3f}, RMSE: {v['RMSE']:.3f}, MAE: {v['MAE']:.3f}")

# ============================================================
# 3️⃣ Pydantic Model for API Response
# ============================================================
class WeatherPrediction(BaseModel):
    date: str
    season: str
    tmax: float
    tmin: float
    rh: float
    rain_probability: float
    weather: str

# ============================================================
# 4️⃣ Prediction Function
# ============================================================
def predict_future_weather(date_str: str):
    date = pd.to_datetime(date_str)
    month = date.month
    doy = date.dayofyear
    year = date.year
    season_code = season_map[get_season(month)]

    # Use median seasonal wind value
    wind = df.groupby("season_code")["wind"].median().loc[season_code]

    # Prepare input for regressors
    X_input = pd.DataFrame([[doy, season_code, 0, wind, year]], columns=['doy', 'season_code', 'rain_flag', 'wind', 'year'])

    # Predict tmax, tmin, rh
    tmax = weather_models['tmax'].predict(X_input)[0]
    tmin = weather_models['tmin'].predict(X_input)[0]
    rh = weather_models['rh'].predict(X_input)[0]

    # Predict rain probability
    rain_features = pd.DataFrame([[tmax, tmin, wind, rh, doy, season_code, year]],
                                 columns=['tmax', 'tmin', 'wind', 'rh', 'doy', 'season_code', 'year'])
    rain_prob = clf.predict_proba(rain_features)[0][1] * 100

    # Describe weather
    if rain_prob >= 80:
        weather_desc = "High chance of rain 🌧"
    elif 60 <= rain_prob < 80:
        weather_desc = "Low chance of rain 🌦"
    else:
        weather_desc = "Likely sunny ☀"

    return {
        "date": date_str,
        "season": get_season(month),
        "tmax": round(float(tmax), 2),
        "tmin": round(float(tmin), 2),
        "rh": round(float(rh), 2),
        "rain_probability": round(float(rain_prob), 2),
        "weather": weather_desc
    }

# ============================================================
# 5️⃣ FastAPI Routes
# ============================================================
@app.get("/")
def home():
    return {"message": "Weather Prediction API running successfully 🌦️"}

@app.get("/predict", response_model=WeatherPrediction)
def get_weather_prediction(date: str = Query(..., description="Date in YYYY-MM-DD format")):
    """Predict future weather based on the given date."""
    try:
        return predict_future_weather(date)
    except Exception as e:
        return {"error": str(e)}

# ============================================================
# 6️⃣ Run App
# ============================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)
