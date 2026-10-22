"""Model construction, training, and evaluation."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from modelling.preprocessing import records_from_features, split_features_target

RANDOM_STATE = 42
TEST_SIZE = 0.20


@dataclass(frozen=True)
class TrainingResult:
    """Fitted production pipeline and held-out evaluation metadata."""

    pipeline: Pipeline
    metrics: dict[str, float]
    evaluation_training_rows: int
    evaluation_test_rows: int
    production_training_rows: int


def build_pipeline() -> Pipeline:
    """Build the preprocessing and linear regression pipeline selected in PR 1."""
    return Pipeline(
        [
            ("encoder", DictVectorizer(sparse=False)),
            ("regressor", LinearRegression()),
        ]
    )


def regression_metrics(y_true: pd.Series, predictions: np.ndarray) -> dict[str, float]:
    """Calculate MAE, RMSE, and R-squared for ring predictions."""
    return {
        "mae": float(mean_absolute_error(y_true, predictions)),
        "rmse": float(root_mean_squared_error(y_true, predictions)),
        "r2": float(r2_score(y_true, predictions)),
    }


def train_and_evaluate(
    dataframe: pd.DataFrame,
    *,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> TrainingResult:
    """Evaluate on a fixed holdout and refit the production pipeline on all rows.

    Metrics describe a separate model fitted on the evaluation training split. The split
    reproduces the notebook evaluation; it is not a new independent test. A new pipeline
    is then fitted on the full dataset and returned for downstream inference.
    """
    features, target = split_features_target(dataframe)
    train_features, test_features, train_target, test_target = train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
    )

    _, train_records = records_from_features(train_features)
    _, test_records = records_from_features(test_features)
    evaluation_pipeline = build_pipeline()
    evaluation_pipeline.fit(train_records, train_target)
    metrics = regression_metrics(test_target, evaluation_pipeline.predict(test_records))

    _, all_records = records_from_features(features)
    production_pipeline = build_pipeline()
    production_pipeline.fit(all_records, target)
    return TrainingResult(
        pipeline=production_pipeline,
        metrics=metrics,
        evaluation_training_rows=len(train_features),
        evaluation_test_rows=len(test_features),
        production_training_rows=len(features),
    )
