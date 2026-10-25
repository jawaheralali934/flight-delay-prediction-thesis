"""
FastAPI back-end for the flight-delay-risk dashboard prototype.
Loads the trained XGBoost model produced by train_models.py and serves
risk-tier predictions with per-flight SHAP explanations.
"""
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
import pandas as pd
import shap
import time
import os

app = FastAPI(title="Flight Delay Risk Dashboard API")

# models.pkl and X_train.pkl are written by train_models.py in the parent folder
BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
with open(os.path.join(BASE, "models.pkl"), "rb") as f:
    models = pickle.load(f)
xgbc = models["xgb"]
X_train = pd.read_pickle(os.path.join(BASE, "X_train.pkl"))
FEATURE_COLS = list(X_train.columns)
explainer = shap.TreeExplainer(xgbc)

class FlightInput(BaseModel):
    month: int
    day: int
    hour: int
    distance: float
    prev_arr_delay: float
    is_weekend: int
    temp: float
    wind_speed: float
    wind_gust: float
    precip: float
    visib: float
    carrier: str
    origin: str

def build_feature_row(inp: FlightInput):
    row = {c: 0 for c in FEATURE_COLS}
    for k in ["month","day","hour","distance","prev_arr_delay","is_weekend",
              "temp","wind_speed","wind_gust","precip","visib"]:
        row[k] = getattr(inp, k)
    carrier_col = f"carrier_{inp.carrier}"
    if carrier_col in row: row[carrier_col] = 1
    origin_col = f"origin_{inp.origin}"
    if origin_col in row: row[origin_col] = 1
    return pd.DataFrame([row])[FEATURE_COLS]

@app.post("/predict")
def predict(inp: FlightInput):
    t0 = time.time()
    X_row = build_feature_row(inp)
    proba = float(xgbc.predict_proba(X_row)[0,1])
    shap_vals = explainer.shap_values(X_row)[0]
    top_idx = np.argsort(-np.abs(shap_vals))[:5]
    top_factors = [{"feature": FEATURE_COLS[i], "shap_value": float(shap_vals[i])} for i in top_idx]
    tier = "Green" if proba < 0.3 else ("Amber" if proba < 0.6 else "Red")
    latency_ms = (time.time() - t0) * 1000
    return {"delay_probability": proba, "risk_tier": tier, "top_factors": top_factors, "latency_ms": latency_ms}

@app.get("/health")
def health():
    return {"status": "ok"}
