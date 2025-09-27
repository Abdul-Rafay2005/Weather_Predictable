import requests
import datetime

NASA_POWER_API = "https://power.larc.nasa.gov/api/temporal/climatology/point"

def get_seasonal_estimates(lat: float, lon: float, date: str):
    """
    Fetch seasonal climatology averages from NASA POWER for given lat/lon and date.
    Used when forecast is unavailable (long-term future).
    """
    dt = datetime.datetime.fromisoformat(date)
    month = dt.month

    params = {
        "parameters": "T2M,T2M_MIN,T2M_MAX,PRECTOTCORR,WS2M",
        "community": "RE",
        "longitude": lon,
        "latitude": lat,
        "format": "JSON"
    }

    try:
        resp = requests.get(NASA_POWER_API, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()["properties"]["parameter"]

        return {
            "tmax": data["T2M_MAX"][str(month)],
            "tmin": data["T2M_MIN"][str(month)],
            "precipitation": data["PRECTOTCORR"][str(month)],
            "wind_max": data["WS2M"][str(month)]
        }
    except Exception as e:
        return {"error": f"NASA seasonal data fetch failed: {str(e)}"}
