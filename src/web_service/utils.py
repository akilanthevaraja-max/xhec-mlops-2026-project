"""Helpers for the web service."""

from pathlib import Path
from typing import Any

from modelling.utils import load_pickle


def load_object(path: str | Path) -> Any:
    """Load a trusted pickle object, such as the training artifact written by the training flow."""
    return load_pickle(path)
