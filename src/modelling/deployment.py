"""Serve the weekly training deployment using the course Prefect runner pattern."""

import argparse
from pathlib import Path

from prefect import serve
from prefect.client.schemas.schedules import CronSchedule
from prefect.deployments.runner import RunnerDeployment

from modelling.workflows import train_model_workflow

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CRON = "0 0 * * 0"
DEFAULT_TIMEZONE = "Europe/Paris"


def build_training_deployment(
    trainset_path: str | Path = PROJECT_ROOT / "data/abalone.csv",
    output_path: str | Path = PROJECT_ROOT / "src/web_service/local_objects/abalone_pipeline.pkl",
    cron: str = DEFAULT_CRON,
    timezone: str = DEFAULT_TIMEZONE,
) -> RunnerDeployment:
    """Build a weekly local deployment without registering or starting it."""
    return train_model_workflow.to_deployment(
        name="weekly-retraining",
        version="0.1.0",
        tags=["abalone", "training"],
        schedule=CronSchedule(cron=cron, timezone=timezone),
        parameters={"trainset_path": str(Path(trainset_path).resolve()), "output_path": str(Path(output_path).resolve())},
        description="Retrain the fixed linear recipe and save the full-data pipeline; holdout metrics are separately scoped.",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Register and serve the scheduled Abalone training deployment.")
    parser.add_argument("--cron", default=DEFAULT_CRON)
    parser.add_argument("--timezone", default=DEFAULT_TIMEZONE)
    arguments = parser.parse_args()
    deployment = build_training_deployment(cron=arguments.cron, timezone=arguments.timezone)
    serve(deployment)
