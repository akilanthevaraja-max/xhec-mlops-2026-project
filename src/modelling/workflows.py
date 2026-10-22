"""Prefect training workflow reusing the stage-2 training recipe."""

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
from prefect import flow, get_run_logger, task

from modelling.main import DEFAULT_MODEL_PATH, build_metadata
from modelling.preprocessing import load_dataset
from modelling.training import TrainingResult, train_and_evaluate
from modelling.utils import pickle_object


@task(name="Load and validate training data")
def load_training_data(trainset_path: str) -> pd.DataFrame:
    """Read the CSV with the same validation as the plain training CLI."""
    return load_dataset(trainset_path)


@task(name="Evaluate recipe and refit production model")
def fit_model(dataframe: pd.DataFrame) -> TrainingResult:
    """Run the unchanged holdout evaluation and full-data refit."""
    return train_and_evaluate(dataframe)


@task(name="Save pipeline and provenance")
def save_training_artifact(trainset_path: str, output_path: str, dataset_rows: int, result: TrainingResult) -> dict[str, Any]:
    """Serialize the fitted encoder/model and correctly scoped metadata."""
    metadata = build_metadata(Path(trainset_path), dataset_rows, result)
    destination = pickle_object({"pipeline": result.pipeline, "metadata": metadata}, output_path)
    return {"artifact_path": str(destination.resolve()), "metadata": metadata}


@flow(name="Abalone model training", log_prints=True)
def train_model_workflow(trainset_path: str, output_path: str = str(DEFAULT_MODEL_PATH)) -> dict[str, Any]:
    """Load, validate, train, evaluate, and save an Abalone regression pipeline."""
    logger = get_run_logger()
    dataframe = load_training_data(trainset_path)
    result = fit_model(dataframe)
    artifact = save_training_artifact(trainset_path, output_path, len(dataframe), result)
    logger.info("Saved model: %s", artifact["artifact_path"])
    logger.info("Holdout evaluation metrics: %s", result.metrics)
    return artifact


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Abalone training as a Prefect flow.")
    parser.add_argument("trainset_path", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_MODEL_PATH)
    arguments = parser.parse_args()
    train_model_workflow(str(arguments.trainset_path.resolve()), str(arguments.output.resolve()))
