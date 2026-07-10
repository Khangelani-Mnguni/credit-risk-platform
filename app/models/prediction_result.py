"""
Prediction result.
"""

from dataclasses import dataclass


@dataclass(slots=True)
class PredictionResult:

    probability: float

    score: int

    odds: float

    risk_band: str

    decision: str

    rating: str

    intercept_points: float | None = None

    feature_points: dict | None = None