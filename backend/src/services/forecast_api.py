# src/services/forecast_api.py
import requests
from datetime import datetime, date as date_cls
from typing import Dict, Any, Optional

# Open-Meteo daily parameters we will request
# Docs: https://open-meteo.com/
OM_DAILY = "temperature_2m_max,temperature_2m_min,precipitation_sum,windspeed_10m_max"

def fetch_open_meteo_daily(lat: float, lon: float, start_date: str, end_date: str, timezone: str = "UTC"):
    """
    Fetch daily forecast data from Open-Meteo between start_date and end_date (YYYY-MM-DD).
    Returns parsed JSON dict with daily arrays.
    """
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": OM_DAILY,
        "start_date": start_date,
        "end_date": end_date,
        "timezone": timezone
    }
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    return r.json()

def forecast_probabilities_for_date(lat: float, lon: float, target_date: str) -> Dict[str, Optional[float]]:
    """
    Compute simple forecast-based probability estimates for target_date using Open-Meteo daily values.
    Returns percentages for very_wet, very_hot, very_cold, very_windy, very_uncomfortable.
    NOTE: Open-Meteo returns deterministic forecast values; here we convert those values into 0/100% probabilities
    (or simple heuristics) because we don't have ensemble spread. This is a simple approximation:
      - if forecast value crosses threshold -> 100% (very likely), else 0% (not likely).
    You can refine by using ensembles or probabilistic forecasts later.
    """
    # request the one-day window
    # if target_date is today or future, Open-Meteo will return forecast (if available)
    try:
        resp = fetch_open_meteo_daily(lat, lon, target_date, target_date)
    except Exception as e:
        # return None-probabilities on failure
        return {
            "very_wet_percent": None,
            "very_hot_percent": None,
            "very_cold_percent": None,
            "very_windy_percent": None,
            "very_uncomfortable_percent": None,
            "notes": f"Open-Meteo fetch failed: {e}"
        }

    daily = resp.get("daily", {})
    # parse values (lists with single element if successful)
    try:
        tmax = _safe_get_first(daily.get("temperature_2m_max"))
        tmin = _safe_get_first(daily.get("temperature_2m_min"))
        prcp = _safe_get_first(daily.get("precipitation_sum"))
        ws = _safe_get_first(daily.get("windspeed_10m_max"))
    except Exception:
        tmax = tmin = prcp = ws = None

    # thresholds (should match climatology thresholds)
    TH = {
        "very_wet_mm": 2.0,
        "very_hot_c": 30.0,
        "very_cold_c": 0.0,
        "very_windy_ms": 10.0
    }

    is_wet = (prcp is not None) and (prcp >= TH["very_wet_mm"])
    is_hot = (tmax is not None) and (tmax >= TH["very_hot_c"])
    is_cold = (tmin is not None) and (tmin <= TH["very_cold_c"])
    is_windy = (ws is not None) and (ws >= TH["very_windy_ms"])

    uncomfortable = False
    if is_hot and (is_windy or is_wet):
        uncomfortable = True
    if is_cold and is_windy:
        uncomfortable = True

    # Simplistic probability mapping:
    # - If forecast predicts condition -> 90% (high confidence)
    # - If forecast near threshold (e.g., within small margin) -> 50% (medium)
    # - Else -> 5% (low)
    def map_prob(flag, value, thresh):
        if value is None:
            return None
        margin = 1.0  # 1 unit margin (degree or mm or m/s)
        if flag:
            return 90.0
        # near threshold?
        if abs(value - thresh) <= margin:
            return 50.0
        return 5.0

    probs = {
        "very_wet_percent": map_prob(is_wet, prcp, TH["very_wet_mm"]),
        "very_hot_percent": map_prob(is_hot, tmax, TH["very_hot_c"]),
        "very_cold_percent": map_prob(is_cold, tmin, TH["very_cold_c"]),
        "very_windy_percent": map_prob(is_windy, ws, TH["very_windy_ms"]),
        "very_uncomfortable_percent": 90.0 if uncomfortable else (50.0 if (is_hot or is_cold) else 5.0)
    }

    # attach raw forecast values for debugging if needed
    probs["forecast_values"] = {
        "tmax": tmax, "tmin": tmin, "precipitation": prcp, "wind_max": ws
    }
    probs["notes"] = "Forecast-derived probabilities from Open-Meteo (simple heuristic mapping)."

    return probs

def _safe_get_first(arr):
    if arr is None:
        return None
    if isinstance(arr, list) and len(arr) > 0:
        v = arr[0]
        # sometimes values can be None
        return v if v is not None else None
    return None
