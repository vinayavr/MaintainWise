
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "artifacts" / "model.pkl"

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

model = joblib.load(MODEL_PATH)

app = FastAPI(
    title="MaintainWise Prediction API",
    description="Predictive maintenance using the trained Random Forest model.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class SensorInput(BaseModel):
    machine_type: str = Field(pattern="^[LMH]$")
    air_temperature: float = Field(ge=295, le=305)
    process_temperature: float = Field(ge=305, le=315)
    rotational_speed: float = Field(ge=1000, le=3000)
    torque: float = Field(ge=3, le=80)
    tool_wear: float = Field(ge=0, le=260)


@app.get("/")
def health():
    return {"status": "ok", "service": "MaintainWise Prediction API"}


@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": model is not None}


@app.post("/predict")
def predict(data: SensorInput):
    if data.process_temperature <= data.air_temperature:
        raise HTTPException(
            status_code=422,
            detail="Process temperature must be greater than air temperature.",
        )

    temperature_difference = (
        data.process_temperature - data.air_temperature
    )
    mechanical_power = (
        data.torque * data.rotational_speed * (2 * np.pi / 60)
    )

    # Preserve the feature order expected by the trained model.
    features = pd.DataFrame([{
        "Type": data.machine_type,
        "Air temperature [K]": data.air_temperature,
        "Process temperature [K]": data.process_temperature,
        "Rotational speed [rpm]": data.rotational_speed,
        "Torque [Nm]": data.torque,
        "Tool wear [min]": data.tool_wear,
        "Temperature difference [K]": temperature_difference,
        "Estimated mechanical power [W]": mechanical_power,
    }])

    try:
        prediction = int(model.predict(features)[0])
        probabilities = model.predict_proba(features)[0]
        classes = list(model.classes_)
        failure_index = classes.index(1)
        failure_probability = float(probabilities[failure_index])
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {type(exc).__name__}: {exc}",
        ) from exc

    return {
        "prediction": prediction,
        "status": "Failure" if prediction == 1 else "No failure",
        "failure_probability": failure_probability,
        "model": "Random Forest",
    }

