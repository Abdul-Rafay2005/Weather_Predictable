from fastapi import FastAPI
from src.routes import forecast, climatology

app = FastAPI(title="Will It Rain On My Parade – Backend")

# include routes
app.include_router(forecast.router, prefix="/forecast", tags=["Forecast"])
app.include_router(climatology.router, prefix="/climatology", tags=["Climatology"])

@app.get("/")
def root():
    return {"message": "Backend API is running 🚀"}
