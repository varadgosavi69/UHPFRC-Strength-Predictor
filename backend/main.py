import sys
from pathlib import Path

import joblib
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
MODEL_PATH = PROJECT_ROOT / "models" / "best_model.pkl"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from explain_model import explain_single_prediction, get_model_context  # noqa: E402


class MixDesignInput(BaseModel):
    cement: float = Field(..., ge=0)
    silica_fume: float = Field(..., ge=0)
    fly_ash: float = Field(..., ge=0)
    sand: float = Field(..., ge=0)
    coarse_aggregate: float = Field(..., ge=0)
    water: float = Field(..., ge=0)
    superplasticizer: float = Field(..., ge=0)
    fiber_type: str = Field(..., min_length=1)
    fiber_content_percent: float = Field(..., ge=0, le=5)
    water_binder_ratio: float = Field(..., gt=0)
    curing_age_days: int = Field(..., gt=0)
    curing_temp_celsius: float
    specimen_type: str = Field(..., min_length=1)

    class Config:
        extra = "forbid"


class PredictionResponse(BaseModel):
    predicted_strength_MPa: float
    shap_contributions: dict[str, float]


app = FastAPI(title="UHPFRC Strength Predictor API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def load_model_on_startup() -> None:
    joblib.load(MODEL_PATH)
    get_model_context()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(mix_design: MixDesignInput) -> PredictionResponse:
    explanation = explain_single_prediction(mix_design.dict())
    return PredictionResponse(
        predicted_strength_MPa=explanation["predicted_compressive_strength_MPa"],
        shap_contributions=explanation["shap_contributions"],
    )
