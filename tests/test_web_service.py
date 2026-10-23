"""Tests for the FastAPI prediction service."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from modelling.main import build_metadata
from modelling.preprocessing import load_dataset
from modelling.training import train_and_evaluate
from src.web_service import main as web_service

VALID_OBSERVATION = {
    "Sex": "M",
    "Length": 0.455,
    "Diameter": 0.365,
    "Height": 0.095,
    "Whole weight": 0.514,
    "Shucked weight": 0.2245,
    "Viscera weight": 0.101,
    "Shell weight": 0.15,
}


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> TestClient:
    """Serve a pipeline trained on the project dataset, without relying on a saved artifact."""
    monkeypatch.setattr(web_service, "MODEL_PATH", tmp_path / "missing.pkl")
    result = train_and_evaluate(load_dataset("data/abalone.csv"))
    metadata = build_metadata(Path("data/abalone.csv"), result.production_training_rows, result)
    monkeypatch.setattr(web_service, "artifact", {"pipeline": result.pipeline, "metadata": metadata})
    with TestClient(web_service.app) as test_client:
        yield test_client


def test_health_check(client: TestClient) -> None:
    """Return the service health response."""
    assert client.get("/").json() == {"health_check": "App up and running!"}


def test_predict_returns_a_number_of_rings(client: TestClient) -> None:
    """Return a numeric ring estimate for a valid specimen."""
    response = client.post("/predict", json=VALID_OBSERVATION)

    assert response.status_code == 201
    assert 1 < response.json()["prediction"] < 30


@pytest.mark.parametrize("override", [{"Sex": "X"}, {"Length": -1}, {"Shell weight": "heavy"}])
def test_predict_rejects_invalid_input(client: TestClient, override: dict) -> None:
    """Reject invalid categories, negative values, and nonnumeric measurements."""
    assert client.post("/predict", json={**VALID_OBSERVATION, **override}).status_code == 422


def test_model_info_describes_the_loaded_model(client: TestClient) -> None:
    """Expose the training metadata stored alongside the pipeline."""
    response = client.get("/model-info")

    assert response.status_code == 200
    info = response.json()
    assert info["model"] == "linear_regression"
    assert info["target"] == "Rings"
    assert len(info["features"]) == 8
    assert len(info["dataset_sha256"]) == 64
    assert set(info["evaluation"]["test_metrics"]) == {"mae", "rmse", "r2"}


def test_model_info_without_model_returns_503(monkeypatch: pytest.MonkeyPatch) -> None:
    """Report an unavailable model instead of empty metadata."""
    monkeypatch.setattr(web_service, "artifact", {})
    monkeypatch.setattr(web_service, "MODEL_PATH", web_service.Path("does-not-exist.pkl"))
    with TestClient(web_service.app) as test_client:
        assert test_client.get("/model-info").status_code == 503


def test_predict_without_model_returns_503(monkeypatch: pytest.MonkeyPatch) -> None:
    """Report unavailable predictions when no artifact has been loaded."""
    monkeypatch.setattr(web_service, "artifact", {})
    monkeypatch.setattr(web_service, "MODEL_PATH", web_service.Path("does-not-exist.pkl"))
    with TestClient(web_service.app) as test_client:
        assert test_client.post("/predict", json=VALID_OBSERVATION).status_code == 503
