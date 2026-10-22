"""FastAPI service serving the trained Abalone regression pipeline."""

import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException

from src.web_service.lib.inference import run_inference
from src.web_service.lib.models import AbaloneInput, AbaloneOutput
from src.web_service.utils import load_object

DEFAULT_MODEL_PATH = Path(__file__).parent / "local_objects" / "abalone_pipeline.pkl"
MODEL_PATH = Path(os.environ.get("MODEL_PATH", DEFAULT_MODEL_PATH))

app = FastAPI(
    title="Abalone age prediction",
    description="Predict the number of rings (age - 1.5) of an abalone from its physical measurements.",
)
artifact: dict[str, Any] = {}


@app.on_event("startup")
def load_model() -> None:
    """Load the trained pipeline once at startup, if it exists."""
    if MODEL_PATH.is_file():
        artifact.update(load_object(MODEL_PATH))


@app.get("/")
def home() -> dict:
    """Health check."""
    return {"health_check": "App up and running!"}


@app.post("/predict", response_model=AbaloneOutput, status_code=201)
def predict(payload: AbaloneInput) -> AbaloneOutput:
    """Predict the number of rings for one abalone."""
    if "pipeline" not in artifact:
        raise HTTPException(status_code=503, detail=f"Model not available at {MODEL_PATH}. Run the training flow first.")
    return AbaloneOutput(prediction=run_inference(artifact["pipeline"], payload))
