import os
import subprocess
from pathlib import Path

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session

from app.database import create_tables, get_db
from app.models import ModelVersion, Prediction
from app.predict import (
    get_model_version,
    is_model_loaded,
    load_model,
    predict as run_predict,
)
from app.schemas import HealthResponse, PredictRequest, PredictResponse, RetrainResponse

load_dotenv()

RETRAIN_API_KEY = os.getenv("RETRAIN_API_KEY", "change-me-in-production")
APP_VERSION = "1.0.0-KE"

app = FastAPI(
    title="AfyaPredict KE",
    description=(
        "Kenyan Clinical Laboratory Test Result Predictor. "
        "Predicts Normal / Abnormal / Inconclusive outcomes from patient admission records. "
        "Built for Kenya's healthcare ecosystem – NHIF, local conditions, KES billing."
    ),
    version=APP_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
)


@app.on_event("startup")
def startup_event() -> None:
    create_tables()
    loaded = load_model()
    if not loaded:
        print("⚠️  Model not found. Run python ml/train.py to train the model.")


@app.get("/health", response_model=HealthResponse, tags=["System"])
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_loaded=is_model_loaded(),
        dataset_version="KE",
        version=APP_VERSION,
    )


@app.post("/predict", response_model=PredictResponse, tags=["Inference"])
def predict(request: PredictRequest, db: Session = Depends(get_db)) -> PredictResponse:
    result = run_predict(request.model_dump())

    pred_record = Prediction(
        predicted_result=result["prediction"],
        probability_normal=result["probabilities"].get("Normal", 0.0),
        probability_abnormal=result["probabilities"].get("Abnormal", 0.0),
        probability_inconclusive=result["probabilities"].get("Inconclusive", 0.0),
        model_version=result["model_version"],
    )
    db.add(pred_record)
    db.commit()

    return PredictResponse(**result)


@app.post("/retrain", response_model=RetrainResponse, tags=["MLOps"])
def retrain(x_api_key: str = Header(...)) -> RetrainResponse:
    if x_api_key != RETRAIN_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key.")

    pipeline = subprocess.run(
        ["python", "scripts/kenyanize.py"],
        capture_output=True,
        text=True,
    )
    if pipeline.returncode != 0:
        raise HTTPException(
            status_code=500,
            detail=f"Kenyanize step failed: {pipeline.stderr}",
        )

    train_proc = subprocess.run(
        ["python", "ml/train.py"],
        capture_output=True,
        text=True,
    )
    if train_proc.returncode != 0:
        raise HTTPException(
            status_code=500,
            detail=f"Training failed: {train_proc.stderr}",
        )

    load_model()

    return RetrainResponse(
        status="success",
        algorithm="RandomForest",
        macro_f1=0.0,
        message="Model retrained successfully on Kenyan dataset. Artifacts reloaded.",
    )


if Path("frontend").exists():
    app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")