"""Separate Prefect inference workflow for a trusted saved pipeline."""

import argparse
import json
from pathlib import Path
from typing import Any

from prefect import flow, get_run_logger, task

from modelling.predicting import predict_from_artifact


@task(name="Load pipeline and predict")
def predict_observations(artifact_path: str, observations: list[dict[str, Any]]) -> list[float]:
    """Reuse the stage-2 validation and record-based prediction interface."""
    return predict_from_artifact(artifact_path, observations).tolist()


@flow(name="Abalone batch inference", retries=1, retry_delay_seconds=5, log_prints=True)
def batch_predict_workflow(artifact_path: str, observations: list[dict[str, Any]]) -> list[float]:
    """Predict new observations with the saved encoder and regression model."""
    predictions = predict_observations(artifact_path, observations)
    get_run_logger().info("Predicted rings for %s observations", len(predictions))
    return predictions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict records from a JSON list using a Prefect flow.")
    parser.add_argument("artifact_path", type=Path)
    parser.add_argument("observations_path", type=Path, help="JSON list with the eight original predictors")
    arguments = parser.parse_args()
    records = json.loads(arguments.observations_path.read_text(encoding="utf-8"))
    print(batch_predict_workflow(str(arguments.artifact_path.resolve()), records))
