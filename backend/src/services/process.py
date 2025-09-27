# src/services/process.py
from datetime import datetime

def compute_probabilities(hist_stats: dict) -> dict:
    """
    Turn historical statistics into event probabilities.
    hist_stats should have *_count and available_years keys.
    """
    years = hist_stats.get("available_years", 30)
    def pct(key):
        return round((hist_stats.get(f"{key}_count", 0) / years) * 100, 2)

    return {
        "very_hot": pct("very_hot"),
        "very_cold": pct("very_cold"),
        "very_wet": pct("very_wet"),
        "very_windy": pct("very_windy"),
        "very_uncomfortable": pct("very_uncomfortable"),
        "available_years": years,
    }


def forecast_to_probabilities(forecast_data: dict) -> dict:
    """
    Convert forecast (from Open-Meteo or NASA seasonal fallback) into % probabilities.
    """
    if not forecast_data or "error" in forecast_data:
        return {
            "very_wet_percent": None,
            "very_hot_percent": None,
            "very_cold_percent": None,
            "very_windy_percent": None,
            "very_uncomfortable_percent": None,
            "notes": forecast_data.get("error", "No forecast data"),
        }

    return {
        "very_wet_percent": forecast_data.get("very_wet", 0),
        "very_hot_percent": forecast_data.get("very_hot", 0),
        "very_cold_percent": forecast_data.get("very_cold", 0),
        "very_windy_percent": forecast_data.get("very_windy", 0),
        "very_uncomfortable_percent": forecast_data.get("very_uncomfortable", 0),
    }


def blend_probabilities(climatology: dict, forecast: dict, target_date_str: str, ml: dict = None) -> dict:
    """
    Blend climatology, forecast, and optional ML predictions.
    Weighting:
      - Forecast has more weight if target_date is close.
      - Climatology dominates for long-range.
      - ML is included if available (averaged in).
    """
    today = datetime.utcnow().date()
    target_date = datetime.strptime(target_date_str, "%Y-%m-%d").date()
    days_diff = abs((target_date - today).days)

    # weights: closer → rely more on forecast, far → rely on climatology
    weight_forecast = max(0.0, 1.0 - (days_diff / 90))  # fade forecast after 90 days
    weight_clim = 1.0 - weight_forecast

    # if no forecast, fall back to climatology only
    if forecast is None or all(forecast.get(k) is None for k in [
        "very_wet_percent", "very_hot_percent", "very_cold_percent",
        "very_windy_percent", "very_uncomfortable_percent"
    ]):
        weight_clim, weight_forecast = 1.0, 0.0

    results = {}
    for key in ["very_wet", "very_hot", "very_cold", "very_windy", "very_uncomfortable"]:
        pc = climatology.get(key, 0) or 0
        pf = forecast.get(f"{key}_percent", None)
        pm = ml.get(f"{key}_percent", None) if ml else None

        # blending
        values = []
        weights = []

        if pc is not None:
            values.append(pc)
            weights.append(weight_clim)

        if pf is not None:
            values.append(pf)
            weights.append(weight_forecast)

        if pm is not None:
            values.append(pm)
            weights.append(0.5)  # ML acts like an extra "voice"

        if values and weights:
            weighted_sum = sum(v * w for v, w in zip(values, weights))
            norm = sum(weights)
            p_blend = round(weighted_sum / norm, 2) if norm > 0 else round(values[0], 2)
        else:
            p_blend = 0

        results[key] = {
            "p_climatology": pc,
            "p_forecast": pf,
            "p_ml": pm,
            "p_blend_percent": p_blend,
        }

    results["blend_weight_forecast"] = round(weight_forecast, 2)
    results["days_difference"] = days_diff

    return results
