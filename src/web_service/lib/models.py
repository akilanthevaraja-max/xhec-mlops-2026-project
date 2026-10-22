"""Pydantic models for the web service."""

from typing import Literal

from pydantic import BaseModel, Field


class AbaloneInput(BaseModel):
    """One abalone observation, using the dataset's original column names as aliases."""

    sex: Literal["F", "I", "M"] = Field(alias="Sex")
    length: float = Field(alias="Length", ge=0)
    diameter: float = Field(alias="Diameter", ge=0)
    height: float = Field(alias="Height", ge=0)
    whole_weight: float = Field(alias="Whole weight", ge=0)
    shucked_weight: float = Field(alias="Shucked weight", ge=0)
    viscera_weight: float = Field(alias="Viscera weight", ge=0)
    shell_weight: float = Field(alias="Shell weight", ge=0)


class AbaloneOutput(BaseModel):
    """Predicted number of rings (age is Rings + 1.5)."""

    prediction: float
