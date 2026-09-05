"""
Vantara Customer Intelligence Platform - REST API
Run with: uvicorn api.main:app --reload
Docs at:  http://127.0.0.1:8000/docs
"""
from fastapi import FastAPI, UploadFile, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib 
import io

app = FastAPI(
    title="Vantara Customer Intelligence Platform",
    description="Predicts customer churn, estimates lifetime value, and returns segment for a customer.",
    version="1.0.0",
)

# ---- load real trained artifacts (produced by src/models/*.py) ----
churn_pre = joblib.load("models_artifacts/preprocessor.pkl")
churn_model = joblib.load("models_artifacts/churn_model.pkl")
clv_pre = joblib.load("models_artifacts/clv_preprocessor.pkl")
clv_model = joblib.load("models_artifacts/clv_model.pkl")
segment_scaler = joblib.load("models_artifacts/segment_scaler.pkl")
segment_model = joblib.load("models_artifacts/segment_kmeans.pkl")

CHURN_NUM = ["Recency", "Frequency", "Monetary", "AvgOrderValue", "TenureDays",
             "AvgBasketSize", "PurchaseGapStd", "EngagementScore", "ReturnRate"]
CLV_NUM = ["Recency", "Frequency", "AvgOrderValue", "TenureDays", "AvgBasketSize", "PurchaseGapStd", "ReturnRate"]


class CustomerFeatures(BaseModel):
    Recency: float
    Frequency: float
    Monetary: float
    AvgOrderValue: float
    TenureDays: float
    AvgBasketSize: float
    PurchaseGapStd: float
    EngagementScore: float
    ReturnRate: float
    Country: str = "United Kingdom"


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model/metadata")
def metadata():
    return {
        "churn_model": type(churn_model).__name__,
        "clv_model": type(clv_model).__name__,
        "segmentation_model": "KMeans",
        "version": "1.0.0",
    }


@app.post("/predict/churn")
def predict_churn(payload: CustomerFeatures):
    row = pd.DataFrame([payload.dict()])
    X = churn_pre.transform(row[CHURN_NUM + ["Country"]])
    proba = float(churn_model.predict_proba(X)[0][1])
    return {"churn_probability": round(proba, 4), "churn_label": int(proba >= 0.5)}


@app.post("/predict/clv")
def predict_clv(payload: CustomerFeatures):
    row = pd.DataFrame([payload.dict()])
    X = clv_pre.transform(row[CLV_NUM + ["Country"]])
    pred_log = clv_model.predict(X)[0]
    clv = float(np.expm1(pred_log))
    return {"predicted_clv": round(clv, 2)}


@app.post("/predict/segment")
def predict_segment(payload: CustomerFeatures):
    row = pd.DataFrame([payload.dict()])
    row["Monetary"] = np.log1p(row["Monetary"])
    X = segment_scaler.transform(row[["Recency", "Frequency", "Monetary"]])
    cluster = int(segment_model.predict(X)[0])
    return {"cluster": cluster}


@app.post("/predict/batch")
async def predict_batch(file: UploadFile):
    if not file.filename.endswith(".csv"):
        raise HTTPException(400, "Please upload a CSV file")
    content = await file.read()
    df = pd.read_csv(io.BytesIO(content))
    missing = set(CHURN_NUM + ["Country"]) - set(df.columns)
    if missing:
        raise HTTPException(400, f"Missing columns: {missing}")
    X = churn_pre.transform(df[CHURN_NUM + ["Country"]])
    df["churn_probability"] = churn_model.predict_proba(X)[:, 1]
    return df.to_dict(orient="records")
