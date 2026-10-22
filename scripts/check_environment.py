"""Verify environment compatibility without starting later project deliverables."""

import os
import sys
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("PREFECT_HOME", str(ROOT / "work/prefect"))
os.environ.setdefault("MPLBACKEND", "Agg")


def main() -> None:
    """Check the course stack, a tiny sklearn workflow, and FastAPI validation."""
    import numpy as np
    import pandas as pd
    import prefect
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from pydantic import BaseModel, Field
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import root_mean_squared_error

    if sys.version_info[:2] != (3, 11):
        raise RuntimeError("Use Python 3.11, as in the course activities.")
    records = pd.DataFrame({"category": ["A", "B", "A", "B"]}).to_dict(orient="records")
    vectorizer = DictVectorizer()
    features = vectorizer.fit_transform(records)
    target = np.array([1.0, 2.0, 1.0, 2.0])
    prediction = LinearRegression().fit(features, target).predict(features)
    np.testing.assert_allclose(prediction, target)
    assert root_mean_squared_error(target, prediction) < 1e-10
    assert callable(prefect.flow) and callable(prefect.task)

    class Payload(BaseModel):
        value: int = Field(..., ge=1)

    # Keep this local smoke check independent of the later prediction API.
    app = FastAPI()

    @app.post("/check")
    def check(payload: Payload) -> dict[str, int]:
        """Return a validated integer for the environment check."""
        return {"value": payload.value}

    with TestClient(app) as client:
        assert client.post("/check", json={"value": 1}).json() == {"value": 1}
        assert client.post("/check", json={"value": 0}).status_code == 422
    for package in ["pandas", "scikit-learn", "pydantic", "fastapi", "prefect", "ruff", "pre-commit"]:
        print(f"{package}=={version(package)}")
    print("Course environment compatibility checks passed.")


if __name__ == "__main__":
    main()
