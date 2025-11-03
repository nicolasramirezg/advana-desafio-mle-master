from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
import pandas as pd
from challenge.model import DelayModel

app = FastAPI(title="Flight Delay Prediction API")

model = DelayModel()


@app.get("/health", status_code=200)
async def get_health() -> dict:
    return {
        "status": "OK"
    }


class FlightsInput(BaseModel):
    flights: List[Dict]


@app.post("/predict", status_code=200)
def predict(data: FlightsInput):
    try:
        flights_df = pd.DataFrame(data.flights)

        #Validation 1
        if not all(col in flights_df.columns for col in ["OPERA", "TIPOVUELO", "MES"]):
            raise HTTPException(status_code=400, detail="Missing required columns.")

        #MES must be integer between 1 and 12
        if not flights_df["MES"].apply(lambda x: isinstance(x, (int, float)) and 1 <= int(x) <= 12).all():
            raise HTTPException(status_code=400, detail="Invalid MES value.")

        # TIPOVUELO must be 'N' or 'I'
        if not flights_df["TIPOVUELO"].isin(["N", "I"]).all():
            raise HTTPException(status_code=400, detail="Invalid TIPOVUELO value.")

        # Validate OPERA exists in training set /safeguard
        allowed_operas = [
            "Aerolineas Argentinas",
            "Grupo LATAM",
            "Sky Airline",
            "Copa Air",
            "Latin American Wings",
            "Avianca",
            "JetSMART SPA",
            "American Airlines",
            "Air France",
            "Qantas Airways",
            "Gol Trans",
            "United Airlines",
            "Iberia"
        ]
        if not flights_df["OPERA"].isin(allowed_operas).all():
            raise HTTPException(status_code=400, detail="Invalid OPERA value.")

        #Preprocess and predict
        features = model.preprocess(flights_df)
        preds = model.predict(features)

        return {"predict": preds}

    except HTTPException:
        # Propagate controlled errors
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")
