"""
FastAPI backend for the Stroke Risk Prediction demo.

Endpoints
---------
GET  /health                -> liveness check
GET  /meta/models           -> list of available model keys + display names
GET  /meta/options          -> dropdown lists + numeric input ranges
GET  /meta/statistics       -> show statistics and analytics Stroke rate 
POST /predict               -> run a prediction with the chosen model

"""
import json
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
LIST_DIR = BASE_DIR / "list"
THRESHOLD_DIR = BASE_DIR / "Threshold"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
STATISTICS_DIR = BASE_DIR / "Statistics_data"

app = FastAPI(
    title="Stroke Risk Prediction API",
    description = 'Predict Stroke Disease using ML models',
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model 
MODEL_FILES = {
    "logistic_regression": "logistic_regression.pkl",
    "random_forest": "random_forest.pkl",
    "xgboost": "xgboost.pkl",
    "lightgbm": "lightgbm.pkl",
}

MODEL_DISPLAY_NAMES = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
}

_models = {}
for key, filename in MODEL_FILES.items():
    path = MODELS_DIR / filename
    if path.exists():
        _models[key] = joblib.load(path)

# Load threshold 
with open (THRESHOLD_DIR / "best_thresholds.pkl") as f:
    _best_thresholds_by_display_name = json.load(f)
    _thresholds = {
        key: _best_thresholds_by_display_name.get(MODEL_DISPLAY_NAMES[key], 0.5)
        for key in MODEL_FILES
    }

# Load List
_ever_married_lists = joblib.load(LIST_DIR/ "ever_married_list.pkl")
_smoking_status_lists = joblib.load(LIST_DIR/ "smoking_status_list.pkl")
_work_type_lists = joblib.load(LIST_DIR/ "work_type_list.pkl")

# Load artifacts
with open (ARTIFACTS_DIR / "bmi_bounds.json") as f:
    _bmi_bounds = json.load(f)
    
with open(ARTIFACTS_DIR / "feature_ranges.json") as f:
    _feature_ranges = json.load(f)

# Load Statistics data

with open(STATISTICS_DIR / "statistics.json") as f:
    _statistics_data = json.load(f)

# Schema input

ModelKey = Literal["logistic_regression", "random_forest", "xgboost", "lightgbm"]


class PredictRequest(BaseModel):
    model: ModelKey = Field(..., description="Which trained model to use")
    age: float = Field(..., ge=0, le=120)
    hypertension: int = Field(..., ge=0, le=1)
    heart_disease: int = Field(..., ge=0, le=1)
    ever_married: str
    work_type: str
    avg_glucose_level: float = Field(..., gt=0)
    bmi: float = Field(..., gt=0)
    smoking_status: str

# Feature engineering 
def age_group(age: float) -> str:
    if age < 18:
        return "Child"
    elif age < 40:
        return "Young Adult"
    elif age < 60:
        return "Middle Aged"
    else:
        return "Senior"


def bmi_category(bmi: float) -> str:
    if bmi < 18.5:
        return "Underweight"
    elif bmi < 25:
        return "Normal"
    elif bmi < 30:
        return "Overweight"
    else:
        return "Obese"


def glucose_category(glucose: float) -> str:
    if glucose < 100:
        return "Normal"
    elif glucose < 126:
        return "Prediabetes"
    else: 
        return "Diabetes"


def build_feature_row(payload: "PredictRequest") -> pd.DataFrame:
    bmi = payload.bmi
    # Same winsorization bound used at training time, applied to user input too
    bmi_capped = min(max(_bmi_bounds["min"],bmi), _bmi_bounds["max"])

    row = {
        "Age": payload.age,
        "Hypertension": payload.hypertension,
        "Heart Disease": payload.heart_disease,
        "Ever Married": payload.ever_married,
        "Work Type": payload.work_type,
        "Avg Glucose Level": payload.avg_glucose_level,
        "Bmi": bmi_capped,
        "Smoking Status": payload.smoking_status,
        "Age Group": age_group(payload.age),
        "Bmi Category": bmi_category(bmi_capped),
        "Glucose Category": glucose_category(payload.avg_glucose_level),
        "Comorbidity Score": payload.hypertension + payload.heart_disease,
        "Is Smoker": 1 if payload.smoking_status in ["smokes", "formerly smoked"] else 0,
        "Age Bmi Interaction": payload.age * bmi_capped,
        "Age Glucose Interaction": payload.age * payload.avg_glucose_level,
    }
    return pd.DataFrame([row])

class PredictResponse(BaseModel):
    model: str
    model_display_name: str
    probability: float
    prediction: int
    risk_label: str
    message: str


# API router

@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": list(_models.keys())}


@app.get("/meta/models")
def list_models():
    return [
        {"key": key, "display_name": MODEL_DISPLAY_NAMES[key],
         "threshold": _thresholds[key]}
        for key in MODEL_FILES
    ]


@app.get("/meta/options")
def get_options():
    return {
        "ever_married": _ever_married_lists,
        "work_type": _work_type_lists,
        "smoking_status": _smoking_status_lists,
        "ranges": _feature_ranges,
    }


@app.get("/meta/statistics")
def get_statistics():
    return _statistics_data


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest):
    if payload.model not in _models:
        raise HTTPException(status_code=503, detail=f"Model '{payload.model}' is not loaded on the server.")
    if payload.ever_married not in _ever_married_lists:
        raise HTTPException(status_code=422, detail="Invalid 'ever_married' value.")
    if payload.work_type not in  _work_type_lists:
        raise HTTPException(status_code=422, detail="Invalid 'work_type' value.")
    if payload.smoking_status not in _smoking_status_lists:
        raise HTTPException(status_code=422, detail="Invalid 'smoking_status' value.")

    pipeline = _models[payload.model]
    row = build_feature_row(payload)
    probality = float(pipeline.predict_proba(row)[0, 1])
    threshold = _thresholds[payload.model]
    predict = int(probality >= threshold)

    if probality < 0.2:
        risk_label = "Low"
    elif probality < threshold:
        risk_label = "Moderate"
    else:
        risk_label = "High"

    message = "Stroke" if predict == 1 else 'No Stroke'
    
    return PredictResponse(
        model = payload.model,
        model_display_name = MODEL_DISPLAY_NAMES[payload.model],
        probability = round(probality, 4),
        prediction = predict,
        risk_label = risk_label,
        message = message
    )
