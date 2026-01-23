"""
ml_detector.py

Sentence-level grammatical error detection using ML.
Acts as a gatekeeper before correction is applied.
"""

import joblib
import os


MODEL_PATH = "models/grammar_detector.joblib"


def load_detector():
    """
    Load trained grammar error detection model.
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "ML detector model not found. "
            "Train the model and save it to models/grammar_detector.joblib"
        )

    return joblib.load(MODEL_PATH)
