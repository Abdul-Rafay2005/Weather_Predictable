# src/routes/forecast.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime
from src.services import nasa_data, process, forecast_api, seasonal_data, ml_model

router = APIRouter()

class ForecastRequest(BaseModel):
    lat: Optional[float] = Field(None, description="Latitude (decimal degrees)")
    lon: Optional[float] = Field(None, description="Longitude (decimal degrees)")
    date: Optional[str] = Field(None, description="Target date in YYYY-MM-DD format")
    year_start: Optional[int] = Field(1995, description="Start year for historical climatology")
    year_end: Optional[int] = Field(2024, description="End year for historical climatology")

def _is_forecast_invalid(probs: dict) -> bool:
    keys = ["very_wet_percent","very_hot_percent","very_cold_percent","very_windy_percent","very_uncomfortable_percent"]
    return probs is None or all(probs.get(k) is None for k in keys)

@router.post("/")
def get_forecast(req: ForecastRequest):
    if req.lat is None or req.lon is None:
        raise HTTPException(status_code=422, detail="lat and lon are required (decimal degrees).")

    # parse or default date
    target_date = datetime.strptime(req.date, "%Y-%m-%d").date() if req.date else datetime.utcnow().date()
    month, day = target_date.month, target_date.day

    # 1️⃣ Climatology
    hist = nasa_data.fetch_historical_for_doy(req.lat, req.lon, req.year_start, req.year_end, month, day)
    clim_probs = process.compute_probabilities(hist)

    # 2️⃣ Forecast (Open-Meteo)
    forecast_probs = forecast_api.forecast_probabilities_for_date(req.lat, req.lon, str(target_date))

    # 3️⃣ If forecast fails -> ML model fallback
    if _is_forecast_invalid(forecast_probs):
        forecast_probs = ml_model.predict_probabilities(req.lat, req.lon, str(target_date))

    # 4️⃣ Blend climatology + forecast/ML
    blend = process.blend_probabilities(climatology=clim_probs, forecast=forecast_probs, target_date_str=str(target_date))

    response = {
        "location": {"lat": req.lat, "lon": req.lon},
        "date": str(target_date),
        "historical_years_used": clim_probs.get("available_years"),
        "probabilities_climatology": {
            "very_wet_percent": clim_probs.get("very_wet"),
            "very_hot_percent": clim_probs.get("very_hot"),
            "very_cold_percent": clim_probs.get("very_cold"),
            "very_windy_percent": clim_probs.get("very_windy"),
            "very_uncomfortable_percent": clim_probs.get("very_uncomfortable")
        },
        "probabilities_forecast": forecast_probs,
        "blend": blend,
        "notes": "Probabilities: climatology (historical) + forecast (Open-Meteo) or ML model (NASA POWER features) blended."
    }
    return response
