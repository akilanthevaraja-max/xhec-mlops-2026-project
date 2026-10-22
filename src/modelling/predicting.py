"""Reusable inference interface for the trained Abalone pipeline."""

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from modelling.preprocessing import records_from_features
from modelling.utils import load_pickle


def predict(
    pipeline: Pipeline,
    observations: pd.DataFrame | Mapping[str, Any] | Sequence[Mapping[str, Any]],
) -> np.ndarray:
    """Predict continuous ring counts for one or more validated observations."""
    _, records = records_from_features(observations)
    return np.asarray(pipeline.predict(records), dtype=float)


def predict_from_artifact(
    artifact_path: str | Path,
    observations: pd.DataFrame | Mapping[str, Any] | Sequence[Mapping[str, Any]],
) -> np.ndarray:
    """Load a trusted training artifact and predict ring counts."""
    artifact = load_pickle(artifact_path)
    if not isinstance(artifact, dict) or "pipeline" not in artifact:
        raise ValueError("Invalid model artifact: missing pipeline")
    pipeline = artifact["pipeline"]
    if not isinstance(pipeline, Pipeline):
        raise ValueError("Invalid model artifact: pipeline has an unexpected type")
    return predict(pipeline, observations)
