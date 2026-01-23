"""
corrector.py

Main orchestration module for the Grammar Auto-Corrector.

Pipeline:
1. ML-based error detection (Phase 4)
2. Spelling correction (Phase 3)
3. Rule-based grammar correction (Phase 2)

Design principle:
- Precision over recall
- Correct only when confident
"""

import spacy
import pandas as pd

# Phase 3
from src.spelling import correct_spelling_sentence

# Phase 2
from src.rules import (
    subject_verb_agreement_rule,
    article_rule,
    tense_rule
)

# Phase 4 (ML detector)
from src.ml_detector import load_detector


# -------------------- Setup --------------------

nlp = spacy.load("en_core_web_sm")

RULES = [
    subject_verb_agreement_rule,
    article_rule,
    tense_rule
]

# Load trained ML model (sentence-level classifier)
ml_detector = load_detector()


# -------------------- Core Function --------------------

def correct_sentence(sentence: str) -> str:
    """
    Correct a sentence using ML + spelling + grammar rules.

    Parameters
    ----------
    sentence : str

    Returns
    -------
    str
        Corrected sentence
    """

    # ---------- Phase 4: ML-based error detection ----------
    df = pd.DataFrame([{"sentence": sentence}])
    is_correct = ml_detector.predict(df)[0]

    # If ML predicts sentence is correct, return as-is
    if is_correct == 1:
        return sentence

    # ---------- Phase 3: Spelling correction ----------
    sentence = correct_spelling_sentence(sentence)

    # ---------- Phase 2: Grammar rules ----------
    doc = nlp(sentence)

    for rule in RULES:
        corrected = rule(doc)
        if corrected:
            return corrected

    # If no rule fires, return spell-corrected sentence
    return sentence


# -------------------- Batch Utility --------------------

def correct_sentences(sentences):
    """
    Apply correction to a list of sentences.
    """
    return [correct_sentence(s) for s in sentences]
