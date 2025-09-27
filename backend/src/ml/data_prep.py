import requests
import pandas as pd
import numpy as np
from datetime import datetime

def fetch_nasa_power_daily(lat, lon, year_start=1995, year_end=2024):
    """
    Fetch daily NASA POWER data for given location & years.
    Variables: Tmax, Tmin, Precipitation, Windspeed, Humidity.
    """
    base_url = "https://power.larc.nasa.gov/api/temporal/daily/point"
    params = {
        "parameters": "T2M_MAX,T2M_MIN,PRECTOTCORR,WS2M,RH2M",
        "community": "RE",
        "longitude": lon,
        "latitude": lat,
        "start": f"{year_start}0101",
        "end": f"{year_end}1231",
        "format": "JSON"
    }

    print(f"Fetching NASA POWER data for {lat}, {lon} ...")
    resp = requests.get(base_url, params=params)
    resp.raise_for_status()
    data = resp.json()["properties"]["parameter"]

    # Convert JSON to DataFrame
    df = pd.DataFrame({
        "date": list(data["T2M_MAX"].keys()),
        "tmax": list(data["T2M_MAX"].values()),
        "tmin": list(data["T2M_MIN"].values()),
        "precip": list(data["PRECTOTCORR"].values()),
        "wind": list(data["WS2M"].values()),
        "rh": list(data["RH2M"].values())
    })

    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    df["doy"] = df["date"].dt.dayofyear
    return df


def label_conditions(df):
    """
    Create binary labels for extreme conditions.
    Thresholds are adjustable.
    """
    df["very_hot"] = (df["tmax"] > 35).astype(int)
    df["very_cold"] = (df["tmin"] < 0).astype(int)
    df["very_wet"] = (df["precip"] > 10).astype(int)
    df["very_windy"] = (df["wind"] > 10).astype(int)
    df["very_uncomfortable"] = ((df["tmax"] > 35) & (df["rh"] > 60)).astype(int)
    return df


def build_dataset(lat, lon, year_start=1995, year_end=2024, save_csv="training_data.csv"):
    """
    Build labeled dataset for ML training.
    """
    df = fetch_nasa_power_daily(lat, lon, year_start, year_end)
    df = label_conditions(df)

    # Save CSV
    df.to_csv(save_csv, index=False)
    print(f"✅ Dataset saved to {save_csv}, {len(df)} rows")
    return df


if __name__ == "__main__":
    # Example: New York City
    dataset = build_dataset(lat=40.7128, lon=-74.006, save_csv="nyc_training_data.csv")
    print(dataset.head())
