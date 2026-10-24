"""Exercise the optional UI's real API boundary, demo inputs, and error states."""

from pathlib import Path
from typing import Any

import httpx
import pytest

from research_app.client import PredictionServiceError, request_model_info, request_prediction


def test_prediction_client_posts_exact_specimen(monkeypatch: pytest.MonkeyPatch) -> None:
    """Use the prediction endpoint rather than a local model or simulated value."""
    calls = []

    def post(url: str, **kwargs: Any) -> httpx.Response:
        calls.append((url, kwargs["json"]))
        return httpx.Response(201, json={"prediction": 9.25})

    monkeypatch.setattr("research_app.client.httpx.post", post)
    specimen = {"Sex": "M", "Length": 0.455}
    assert request_prediction("http://api:8001/", specimen) == 9.25
    assert calls == [("http://api:8001/predict", specimen)]


@pytest.mark.parametrize("code", [422, 503, 500])
def test_prediction_client_reports_service_errors(monkeypatch: pytest.MonkeyPatch, code: int) -> None:
    """Surface validation, unavailable-model, and server failures."""
    monkeypatch.setattr("research_app.client.httpx.post", lambda *_args, **_kwargs: httpx.Response(code))
    with pytest.raises(PredictionServiceError):
        request_prediction("http://api:8001", {})


@pytest.mark.parametrize("payload", [{}, {"prediction": "9"}, {"prediction": True}, {"prediction": None}])
def test_prediction_client_rejects_invalid_results(monkeypatch: pytest.MonkeyPatch, payload: dict) -> None:
    """Do not turn malformed API output into an apparently valid age estimate."""
    monkeypatch.setattr("research_app.client.httpx.post", lambda *_args, **_kwargs: httpx.Response(201, json=payload))
    with pytest.raises(PredictionServiceError, match="invalid result"):
        request_prediction("http://api:8001", {})


def test_prediction_client_reports_offline_api(monkeypatch: pytest.MonkeyPatch) -> None:
    """Give a useful message when the Docker API is not running."""

    def offline(*_args: Any, **_kwargs: Any) -> None:
        """Simulate a refused API connection."""
        raise httpx.ConnectError("Connection refused")

    monkeypatch.setattr("research_app.client.httpx.post", offline)
    with pytest.raises(PredictionServiceError, match="Cannot reach"):
        request_prediction("http://api:8001", {})


def test_demo_form_predicts_through_api_and_resets(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fill real example inputs, call FastAPI, display results, and clear stale output."""
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    calls = []

    def post(url: str, **kwargs: Any) -> httpx.Response:
        calls.append((url, kwargs["json"]))
        return httpx.Response(201, json={"prediction": 9.25})

    monkeypatch.setattr("research_app.client.httpx.post", post)
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "streamlit_app.py"), default_timeout=30).run()
    assert not app.exception
    assert app.number_input[0].value is None
    next(button for button in app.button if button.label == "Predict shell rings").click().run()
    assert app.error and not calls
    next(button for button in app.button if button.label == "Fill sample values").click().run()
    assert app.number_input[0].value == 0.455
    next(button for button in app.button if button.label == "Predict shell rings").click().run()
    assert not app.exception
    assert calls and calls[0][0].endswith("/predict")
    assert set(calls[0][1]) == {
        "Sex",
        "Length",
        "Diameter",
        "Height",
        "Whole weight",
        "Shucked weight",
        "Viscera weight",
        "Shell weight",
    }
    assert any(metric.label == "Predicted shell rings" and metric.value == "9.25" for metric in app.metric)
    next(button for button in app.button if button.label == "Clear inputs").click().run()
    assert app.number_input[0].value is None
    assert not any(metric.label == "Predicted shell rings" for metric in app.metric)
    for sample_name, expected_category in [("Sample 2", "F"), ("Sample 3", "I")]:
        app.selectbox(key="sample_name").select(sample_name).run()
        next(button for button in app.button if button.label == "Fill sample values").click().run()
        assert app.selectbox(key="sex").value == expected_category
    assert len(calls) == 1  # Filling samples must not silently request predictions.


@pytest.mark.parametrize("url", ["", "not-a-url", "ftp://example.org", "http://example.org:bad"])
def test_prediction_client_rejects_invalid_api_urls(url: str) -> None:
    """Reject invalid endpoint configuration before attempting a request."""
    with pytest.raises(PredictionServiceError, match="valid HTTP"):
        request_prediction(url, {})


MODEL_INFO = {
    "model_path": "/app/src/web_service/local_objects/abalone_pipeline.pkl",
    "model": "linear_regression",
    "features": ["Sex", "Length", "Diameter", "Height", "Whole weight", "Shucked weight", "Viscera weight", "Shell weight"],
    "target": "Rings",
    "dataset_sha256": "6fda3c4b0c0c" + "0" * 52,
    "production_training_rows": 4177,
    "evaluation": {"test_rows": 836, "test_metrics": {"mae": 1.593, "rmse": 2.212, "r2": 0.548}},
}


def test_model_info_client_reads_the_model_info_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    """Request GET /model-info and return the API's metadata unchanged."""
    calls = []

    def get(url: str, **_kwargs: Any) -> httpx.Response:
        calls.append(url)
        return httpx.Response(200, json=MODEL_INFO)

    monkeypatch.setattr("research_app.client.httpx.get", get)
    assert request_model_info("http://api:8001/") == MODEL_INFO
    assert calls == ["http://api:8001/model-info"]


@pytest.mark.parametrize(
    ("response", "message"),
    [
        (httpx.Response(503), "no loaded model"),
        (httpx.Response(404), "HTTP 404"),
        (httpx.Response(200, json={"model": "linear_regression"}), "invalid model information"),
        (httpx.Response(200, json={**MODEL_INFO, "evaluation": {"test_metrics": {"mae": "low"}}}), "invalid model information"),
    ],
)
def test_model_info_client_reports_unusable_responses(
    monkeypatch: pytest.MonkeyPatch, response: httpx.Response, message: str
) -> None:
    """Explain a missing model, an older API without the endpoint, or malformed metadata."""
    monkeypatch.setattr("research_app.client.httpx.get", lambda *_args, **_kwargs: response)
    with pytest.raises(PredictionServiceError, match=message):
        request_model_info("http://api:8001")


def test_model_info_client_reports_offline_api(monkeypatch: pytest.MonkeyPatch) -> None:
    """Give the same actionable message as predictions when the API is down."""

    def offline(*_args: Any, **_kwargs: Any) -> None:
        """Simulate a refused API connection."""
        raise httpx.ConnectError("Connection refused")

    monkeypatch.setattr("research_app.client.httpx.get", offline)
    with pytest.raises(PredictionServiceError, match="Cannot reach"):
        request_model_info("http://api:8001")


def test_sidebar_shows_loaded_model_details(monkeypatch: pytest.MonkeyPatch) -> None:
    """Display the served model's holdout metrics only after the user asks for them."""
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    calls = []

    def get(url: str, **_kwargs: Any) -> httpx.Response:
        calls.append(url)
        return httpx.Response(200, json=MODEL_INFO)

    monkeypatch.setattr("research_app.client.httpx.get", get)
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "streamlit_app.py"), default_timeout=30).run()
    assert not calls
    next(button for button in app.sidebar.button if button.label == "Show model details").click().run()
    assert not app.exception
    assert calls == ["http://127.0.0.1:8002/model-info"]
    sidebar_text = " ".join(block.value for block in app.sidebar.markdown)
    assert "Linear regression" in sidebar_text
    assert "| 1.59 | 2.21 | 0.55 |" in sidebar_text


def test_sidebar_reports_unavailable_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Show the API error instead of stale or invented model details."""
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    monkeypatch.setattr("research_app.client.httpx.get", lambda *_args, **_kwargs: httpx.Response(503))
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "streamlit_app.py"), default_timeout=30).run()
    next(button for button in app.sidebar.button if button.label == "Show model details").click().run()
    assert any("no loaded model" in error.value for error in app.sidebar.error)
    assert not any("RMSE" in block.value for block in app.sidebar.markdown)
