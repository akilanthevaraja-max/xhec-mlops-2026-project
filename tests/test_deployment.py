"""Validate the scheduled deployment without contacting a server."""

from pathlib import Path

from modelling.deployment import build_training_deployment


def test_deployment_uses_weekly_local_schedule_and_absolute_paths() -> None:
    """Keep the course schedule explicit and paths independent of runner activation."""
    deployment = build_training_deployment()
    assert deployment.name == "weekly-retraining"
    assert deployment.schedule.cron == "0 0 * * 0"
    assert deployment.schedule.timezone == "Europe/Paris"
    assert Path(deployment.parameters["trainset_path"]).is_absolute()
    assert Path(deployment.parameters["output_path"]).is_absolute()
