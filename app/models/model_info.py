"""
Model metadata.
"""

from dataclasses import dataclass


@dataclass(slots=True)
class ModelInfo:

    model_name: str

    version: str

    auc: float

    ks: float

    gini: float

    train_date: str

    developer: str