from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()

class ClimatologyRequest(BaseModel):
    location: str
    month: int   # 1-12

@router.post("/")
def get_climatology(req: ClimatologyRequest):
    # TODO: fetch historical averages from NASA datasets
    return {
        "location": req.location,
        "month": req.month,
        "avg_rain_probability": 30,
        "avg_heat_probability": 10,
        "avg_storm_probability": 2
    }
