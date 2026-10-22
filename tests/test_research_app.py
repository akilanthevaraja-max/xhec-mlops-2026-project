"""Exercise the optional UI's real API boundary, demo inputs, and error states."""

from pathlib import Path
from typing import Any

import httpx
import pytest

from research_app.client import PredictionServiceError, request_prediction


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
