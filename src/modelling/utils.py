"""Persistence helpers for trained Python objects."""

import pickle
from pathlib import Path
from typing import Any


def pickle_object(obj: Any, output_path: str | Path) -> Path:
    """Serialize an object atomically with the highest pickle protocol."""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination.with_suffix(f"{destination.suffix}.tmp")
    with temporary_path.open("wb") as file_handle:
        pickle.dump(obj, file_handle, protocol=pickle.HIGHEST_PROTOCOL)
    temporary_path.replace(destination)
    return destination


def load_pickle(path: str | Path) -> Any:
    """Load a trusted local pickle artifact.

    Pickle can execute arbitrary code during loading. Only artifacts produced by this
    project and obtained from a trusted location should be loaded.
    """
    artifact_path = Path(path)
    if not artifact_path.is_file():
        raise FileNotFoundError(f"Model artifact not found: {artifact_path}")
    with artifact_path.open("rb") as file_handle:
        return pickle.load(file_handle)  # noqa: S301
