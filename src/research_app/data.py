"""Reference dataset and input definitions for the research interface."""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
FEATURES = ["Length", "Diameter", "Height", "Whole weight", "Shucked weight", "Viscera weight", "Shell weight"]
SEX_LABELS = {"F": "Female (F)", "M": "Male (M)", "I": "Infant / immature (I)"}
DESCRIPTIONS = {
    "Length": "Longest shell measurement",
    "Diameter": "Measurement perpendicular to length",
    "Height": "Height with meat in the shell",
    "Whole weight": "Weight of the whole specimen",
    "Shucked weight": "Weight of the meat",
    "Viscera weight": "Gut weight after bleeding",
    "Shell weight": "Weight of the dried shell",
}


def load_reference_data() -> pd.DataFrame:
    """Read the bundled CSV for exploration; never fit or load a prediction model."""
    dataframe = pd.read_csv(ROOT / "data/abalone.csv")
    if dataframe.columns.tolist() != ["Sex", *FEATURES, "Rings"]:
        raise ValueError("The reference CSV does not have the expected schema.")
    return dataframe


def range_warnings(observation: dict[str, str | float], reference: pd.DataFrame) -> list[str]:
    """Flag measurements outside observed ranges without treating them as hard limits."""
    messages = []
    for feature in FEATURES:
        value = float(observation[feature])
        lower, upper = float(reference[feature].min()), float(reference[feature].max())
        if not lower <= value <= upper:
            messages.append(f"{feature} ({value:g}) is outside the reference range {lower:g} to {upper:g}.")
    if float(observation["Height"]) == 0:
        messages.append("A zero height is present in the dataset but is physically implausible. Recheck this measurement.")
    return messages
