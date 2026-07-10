"""
artifact_loader.py

Loads and caches all trained model artifacts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import streamlit as st

from config import (
    PIPELINE_PATH,
    MODEL_PATH,
    SCALER_PATH,
    METADATA_PATH,
)


import sys
from pathlib import Path

# 1. Dynamically find the absolute path to the project root (credit-risk-platform/)
current_file_path = Path(__file__).resolve()
project_root = current_file_path.parent.parent.parent  # goes up: services -> app -> credit-risk-platform

# 2. Append the root path so Python can find preprocessing, modeling, scorecard, etc.
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from preprocessing.preprocessing_pipeline import PreprocessingPipeline
from modeling.logistic_model import LogisticRegressionModel
from scorecard.scorecard_scaler import ScoreScaler


@dataclass(frozen=True)
class Artifacts:
    pipeline: PreprocessingPipeline
    model: LogisticRegressionModel
    scaler: ScoreScaler
    metadata: dict


@st.cache_resource(show_spinner="Loading model artifacts...")
def load_artifacts() -> Artifacts:
    """
    Load all persisted artifacts.

    Returns
    -------
    Artifacts
        Container with model, pipeline, scaler and metadata.
    """

    pipeline = PreprocessingPipeline.load(PIPELINE_PATH)

    model = LogisticRegressionModel.load(MODEL_PATH)

    scaler = ScoreScaler.load(SCALER_PATH)

    with open(METADATA_PATH, "r") as f:
        metadata = json.load(f)

    return Artifacts(
        pipeline=pipeline,
        model=model,
        scaler=scaler,
        metadata=metadata,
    )