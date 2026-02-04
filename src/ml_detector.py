"""
ml_detector.py

Loads the trained ML model used for grammatical error detection.
"""

import joblib
import os

MODEL_PATH = "models/grammar_detector.joblib"


def load_detector():
    """
    Load the trained grammar error detection model.
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. "
            "Train the model in Phase 4 and save it first."
        )

    return joblib.load(MODEL_PATH)
