"""Command-line training flow for the Abalone regression pipeline."""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from modelling.preprocessing import FEATURE_COLUMNS, TARGET_COLUMN, load_dataset
from modelling.training import RANDOM_STATE, TEST_SIZE, train_and_evaluate
from modelling.utils import pickle_object

DEFAULT_MODEL_PATH = Path("src/web_service/local_objects/abalone_pipeline.pkl")


def main(trainset_path: str | Path, output_path: str | Path = DEFAULT_MODEL_PATH) -> dict[str, Any]:
    """Train, evaluate, and serialize the complete model pipeline.

    Args:
        trainset_path: Path to the Abalone CSV file.
        output_path: Destination for the trusted local pickle artifact.

    Returns:
        Metadata stored alongside the fitted pipeline.
    """
    dataset_path = Path(trainset_path)
    dataframe = load_dataset(dataset_path)
    result = train_and_evaluate(dataframe)
    metadata: dict[str, Any] = {
        "model": "linear_regression",
        "features": list(FEATURE_COLUMNS),
        "target": TARGET_COLUMN,
        "dataset_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
        "dataset_rows": len(dataframe),
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "production_training_rows": result.production_training_rows,
        "evaluation": {
            "model_scope": "separate_holdout_model_before_full_data_refit",
            "training_rows": result.evaluation_training_rows,
            "test_rows": result.evaluation_test_rows,
            "test_metrics": result.metrics,
        },
    }
    pickle_object({"pipeline": result.pipeline, "metadata": metadata}, output_path)
    print(json.dumps({"artifact": str(Path(output_path)), **metadata}, indent=2))
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model using the data at the given path.")
    parser.add_argument("trainset_path", type=Path, help="Path to the Abalone CSV training set")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help=f"Pickle destination (default: {DEFAULT_MODEL_PATH})",
    )
    args = parser.parse_args()
    main(args.trainset_path, args.output)
