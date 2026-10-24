"""HTTP interface to the separately running FastAPI prediction service."""

import math
from typing import Any

import httpx


class PredictionServiceError(Exception):
    """An actionable error returned by the prediction service client."""


def validate_base_url(base_url: str) -> None:
    """Reject API URLs that are not absolute HTTP or HTTPS addresses."""
    try:
        parsed_url = httpx.URL(base_url)
    except httpx.InvalidURL as exc:
        raise PredictionServiceError("Enter a valid HTTP or HTTPS API URL.") from exc
    if parsed_url.scheme not in {"http", "https"} or not parsed_url.host:
        raise PredictionServiceError("Enter a valid HTTP or HTTPS API URL.")


def request_model_info(base_url: str) -> dict[str, Any]:
    """Fetch the metadata of the model currently loaded by FastAPI."""
    validate_base_url(base_url)
    try:
        response = httpx.get(f"{base_url.rstrip('/')}/model-info", timeout=5.0)
    except httpx.RequestError as exc:
        raise PredictionServiceError(
            "Cannot reach the prediction service. Check that the API is running and the URL is correct."
        ) from exc
    if response.status_code == 503:
        raise PredictionServiceError("The API is running but has no loaded model.")
    if response.status_code != 200:
        raise PredictionServiceError(f"The prediction service returned HTTP {response.status_code} for its model information.")
    try:
        info = response.json()
        metrics = info["evaluation"]["test_metrics"]
        if not all(isinstance(metrics[name], (int, float)) for name in ("mae", "rmse", "r2")):
            raise ValueError("Expected numeric evaluation metrics")
    except (ValueError, KeyError, TypeError) as exc:
        raise PredictionServiceError("The prediction service returned invalid model information.") from exc
    return info


def request_prediction(base_url: str, observation: dict[str, Any]) -> float:
    """Submit one specimen to FastAPI and validate its numeric response."""
    validate_base_url(base_url)
    try:
        response = httpx.post(f"{base_url.rstrip('/')}/predict", json=observation, timeout=15.0)
    except httpx.RequestError as exc:
        raise PredictionServiceError(
            "Cannot reach the prediction service. Check that the API is running and the URL is correct."
        ) from exc
    if response.status_code == 503:
        raise PredictionServiceError(
            "The prediction service has no loaded model. Ask the service operator to train and load one."
        )
    if response.status_code == 422:
        raise PredictionServiceError("The API rejected this specimen. Check the category and measurement values.")
    if response.status_code != 201:
        raise PredictionServiceError(f"The prediction service returned HTTP {response.status_code}. Please try again later.")
    try:
        value = response.json()["prediction"]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError("Expected a finite numeric prediction")
    except (ValueError, KeyError, TypeError) as exc:
        raise PredictionServiceError("The prediction service returned an invalid result.") from exc
    return float(value)
