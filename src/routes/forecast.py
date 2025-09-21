from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()

class ForecastRequest(BaseModel):
    location: str  # city name or lat,long
    date: str      # YYYY-MM-DD

@router.post("/")
def get_forecast(req: ForecastRequest):
    # TODO: fetch from NASA/NOAA API
    return {
        "location": req.location,
        "date": req.date,
        "rain_probability": 42,
        "heat_probability": 15,
        "storm_probability": 5
    }
