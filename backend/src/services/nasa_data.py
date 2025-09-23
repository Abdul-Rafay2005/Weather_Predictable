# src/services/nasa_data.py
import requests
from datetime import datetime

POWER_BASE = "https://power.larc.nasa.gov/api/temporal/daily/point"

def fetch_power_daily(lat: float, lon: float, start: str, end: str, params: str = "T2M_MAX,T2M_MIN,PRECTOT,WS50M"):
    """
    Fetch NASA POWER daily data for a lat/lon between start and end (YYYYMMDD format).
    Returns JSON dict with 'properties' -> 'parameter' dict where keys are parameter names.
    params: comma separated parameter codes (defaults include: T2M_MAX,T2M_MIN,PRECTOT,WS50M)
    """
    url = POWER_BASE
    payload = {
        "start": start,
        "end": end,
        "latitude": lat,
        "longitude": lon,
        "parameters": params,
        "community": "AG",
        "format": "JSON",
        "user": "anonymous"
    }
    r = requests.get(url, params=payload, timeout=30)
    r.raise_for_status()
    return r.json()

def fetch_historical_for_doy(lat: float, lon: float, year_start: int, year_end: int, month:int, day:int):
    """
    Fetches historical daily data for the same month/day across a span of years.
    Returns dict:
      { year: {param: value, ...}, ... }
    Note: NASA POWER allows a date-range query; we'll request from year_start-01-01 to year_end-12-31
    and then pick out only the month/day entries.
    """
    # NASA POWER wants start/end as YYYYMMDD strings
    start = f"{year_start}0101"
    end = f"{year_end}1231"
    resp = fetch_power_daily(lat, lon, start, end)
    # parameters structure: resp['properties']['parameter'][PARAM_NAME] -> { 'YYYYMMDD': value, ... }
    params = resp.get("properties", {}).get("parameter", {})
    # build per-year entries for requested month/day
    result = {}
    for year in range(year_start, year_end + 1):
        date_key = f"{year}{month:02d}{day:02d}"
        values = {}
        for p, series in params.items():
            # POWER may return 'null' for missing; handle that
            values[p] = series.get(date_key, None)
        result[year] = values
    return result
