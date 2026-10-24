"""Check that the Compose stack wires the UI to the API container."""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_compose_connects_ui_to_healthy_api() -> None:
    """The UI must call the API by service name, on the port FastAPI listens on, once it is healthy."""
    services = yaml.safe_load((ROOT / "docker-compose.yml").read_text())["services"]

    assert services["api"]["build"]["dockerfile"] == "Dockerfile.app"
    assert services["ui"]["build"]["dockerfile"] == "Dockerfile.ui"
    assert services["ui"]["environment"]["API_BASE_URL"] == "http://api:8001"
    assert services["ui"]["depends_on"]["api"]["condition"] == "service_healthy"
    assert "8501:8501" in services["ui"]["ports"]
    assert "EXPOSE 8001" in (ROOT / "Dockerfile.app").read_text()


def test_ui_image_contains_no_model_or_api_code() -> None:
    """The UI image copies only its own package: predictions must go through FastAPI."""
    dockerfile = (ROOT / "Dockerfile.ui").read_text()

    assert "--extra app" in dockerfile
    assert "COPY src/research_app" in dockerfile
    assert "web_service" not in dockerfile
    assert "modelling" not in dockerfile
