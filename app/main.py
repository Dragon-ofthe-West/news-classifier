from pathlib import Path
import json

import joblib
from fastapi import FastAPI, HTTPException

from app.schemas import (
    HealthResponse,
    ModelInfoResponse,
    PredictRequest,
    PredictResponse,
)

ARTIFACTS = Path("artifacts")
PIPELINE_PATH = ARTIFACTS / "pipeline.joblib"
META_PATH = ARTIFACTS / "meta.json"

app = FastAPI(title="News Classifier", version="1.0.0")

if not PIPELINE_PATH.exists():
    raise RuntimeError("pipeline.joblib не найден. Запустите python train.py")

pipeline = joblib.load(PIPELINE_PATH)
meta = json.loads(META_PATH.read_text())
LABELS = meta["labels"]


@app.get("/health", response_model=HealthResponse)
def health():
    return {"status": "ok", "model_version": meta["model_version"]}


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info():
    return {
        "model_version": meta["model_version"],
        "dataset": meta["dataset"],
        "metrics": meta["metrics"],
        "labels": LABELS,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if not req.text.strip():
        raise HTTPException(status_code=422, detail="Пустой текст")
    proba = pipeline.predict_proba([req.text])[0]
    idx = int(proba.argmax())
    return {
        "label": idx,
        "label_name": LABELS[idx],
        "probability": round(float(proba[idx]), 4),
    }
