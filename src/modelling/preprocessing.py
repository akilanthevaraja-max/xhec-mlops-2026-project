"""Dataset loading, validation, and feature conversion."""

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

FEATURE_COLUMNS = (
    "Sex",
    "Length",
    "Diameter",
    "Height",
    "Whole weight",
    "Shucked weight",
    "Viscera weight",
    "Shell weight",
)
NUMERIC_FEATURE_COLUMNS = FEATURE_COLUMNS[1:]
TARGET_COLUMN = "Rings"
EXPECTED_SEX_VALUES = frozenset({"F", "I", "M"})


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load and validate an Abalone training dataset.

    Args:
        path: CSV file containing the eight predictors and the Rings target.

    Returns:
        A validated dataframe in the expected column order.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
        ValueError: If its schema or values are invalid.
    """
    dataset_path = Path(path)
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Training dataset not found: {dataset_path}")

    dataframe = pd.read_csv(dataset_path)
    validate_training_data(dataframe)
    return dataframe


def validate_training_data(dataframe: pd.DataFrame) -> None:
    """Validate the schema and required values of a training dataframe."""
    expected_columns = [*FEATURE_COLUMNS, TARGET_COLUMN]
    if dataframe.columns.tolist() != expected_columns:
        raise ValueError(f"Expected columns {expected_columns}, got {dataframe.columns.tolist()}")
    _validate_features(dataframe.loc[:, list(FEATURE_COLUMNS)])
    if dataframe[TARGET_COLUMN].isna().any():
        raise ValueError(f"Column {TARGET_COLUMN} contains missing values")
    if not pd.api.types.is_numeric_dtype(dataframe[TARGET_COLUMN]):
        raise ValueError(f"Column {TARGET_COLUMN} must be numeric")
    if (dataframe[TARGET_COLUMN] <= 0).any():
        raise ValueError(f"Column {TARGET_COLUMN} must contain positive values")


def split_features_target(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return an explicit feature frame and target series."""
    validate_training_data(dataframe)
    return dataframe.loc[:, list(FEATURE_COLUMNS)].copy(), dataframe[TARGET_COLUMN].copy()


def records_from_features(
    features: pd.DataFrame | Mapping[str, Any] | Sequence[Mapping[str, Any]],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Validate prediction features and convert them to DictVectorizer records."""
    if isinstance(features, pd.DataFrame):
        feature_frame = features.copy()
    elif isinstance(features, Mapping):
        feature_frame = pd.DataFrame([features])
    else:
        feature_frame = pd.DataFrame(list(features))

    if feature_frame.empty:
        raise ValueError("At least one observation is required")
    if set(feature_frame.columns) != set(FEATURE_COLUMNS):
        raise ValueError(f"Prediction input must contain exactly these columns: {list(FEATURE_COLUMNS)}")

    feature_frame = feature_frame.loc[:, list(FEATURE_COLUMNS)]
    _validate_features(feature_frame)
    return feature_frame, feature_frame.to_dict(orient="records")


def _validate_features(features: pd.DataFrame) -> None:
    """Validate feature values shared by training and inference."""
    if features.isna().any().any():
        missing_columns = features.columns[features.isna().any()].tolist()
        raise ValueError(f"Missing feature values in columns: {missing_columns}")

    unexpected_sex_values = set(features["Sex"].unique()) - EXPECTED_SEX_VALUES
    if unexpected_sex_values:
        raise ValueError(f"Unexpected Sex values: {sorted(unexpected_sex_values)}")

    non_numeric_columns = [column for column in NUMERIC_FEATURE_COLUMNS if not pd.api.types.is_numeric_dtype(features[column])]
    if non_numeric_columns:
        raise ValueError(f"Numeric features have invalid types: {non_numeric_columns}")
