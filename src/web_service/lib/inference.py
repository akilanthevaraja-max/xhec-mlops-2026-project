"""Inference logic for the web service."""

from sklearn.pipeline import Pipeline

from modelling.predicting import predict
from src.web_service.lib.models import AbaloneInput


def run_inference(pipeline: Pipeline, payload: AbaloneInput) -> float:
    """Predict the number of rings for one validated observation.

    The payload is converted back to the dataset's original column names (for example
    "Whole weight"), which is what the modelling code expects.
    """
    observation = payload.dict(by_alias=True)
    return float(predict(pipeline, observation)[0])
