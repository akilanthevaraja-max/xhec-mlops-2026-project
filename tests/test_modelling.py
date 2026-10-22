"""Tests for the reusable modelling pipeline."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from modelling.main import main
from modelling.predicting import predict, predict_from_artifact
from modelling.preprocessing import FEATURE_COLUMNS, load_dataset, split_features_target
from modelling.training import train_and_evaluate
from modelling.utils import load_pickle


def make_dataset(row_count: int = 60) -> pd.DataFrame:
    """Create deterministic, schema-compatible synthetic observations."""
    index = np.arange(row_count)
    length = 0.25 + 0.005 * index
    dataframe = pd.DataFrame(
        {
            "Sex": np.array(["F", "I", "M"])[index % 3],
            "Length": length,
            "Diameter": length * 0.80,
            "Height": 0.05 + length * 0.18,
            "Whole weight": 0.08 + length * 1.60,
            "Shucked weight": 0.04 + length * 0.70,
            "Viscera weight": 0.02 + length * 0.35,
            "Shell weight": 0.03 + length * 0.45,
        }
    )
    dataframe["Rings"] = 2.0 + 14.0 * dataframe["Shell weight"] + (index % 4) * 0.1
    return dataframe


def test_load_dataset_rejects_an_invalid_schema(tmp_path: Path) -> None:
    """Reject a training file that omits a required predictor."""
    dataset_path = tmp_path / "invalid.csv"
    make_dataset().drop(columns="Height").to_csv(dataset_path, index=False)

    with pytest.raises(ValueError, match="Expected columns"):
        load_dataset(dataset_path)


def test_training_pipeline_predicts_valid_rows() -> None:
    """Train the pipeline and return finite predictions with evaluation metrics."""
    dataframe = make_dataset()
    result = train_and_evaluate(dataframe)
    features, _ = split_features_target(dataframe)
    predictions = predict(result.pipeline, features.iloc[:3])

    assert predictions.shape == (3,)
    assert np.isfinite(predictions).all()
    assert set(result.metrics) == {"mae", "rmse", "r2"}


def test_main_serializes_reusable_pipeline(tmp_path: Path) -> None:
    """Persist one pipeline containing both categorical encoding and regression."""
    dataframe = make_dataset()
    dataset_path = tmp_path / "abalone.csv"
    artifact_path = tmp_path / "abalone_pipeline.pkl"
    dataframe.to_csv(dataset_path, index=False)

    metadata = main(dataset_path, artifact_path)
    artifact = load_pickle(artifact_path)
    observation = dataframe.loc[0, list(FEATURE_COLUMNS)].to_dict()
    direct_prediction = predict(artifact["pipeline"], observation)
    loaded_prediction = predict_from_artifact(artifact_path, observation)

    assert metadata["dataset_rows"] == len(dataframe)
    assert metadata["features"] == list(FEATURE_COLUMNS)
    np.testing.assert_allclose(loaded_prediction, direct_prediction)
