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
from src.ml_detector import load_detector

ml_detector = load_detector()


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

def extract_pos_sequence(sentence: str) -> str:
    doc = nlp(sentence)
    return " ".join(tok.pos_ for tok in doc)

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
    print("Starting correction pipeline...")
    # ---------- Phase 4: ML-based error detection ----------
    pos_seq = extract_pos_sequence(sentence)
    sent_len = len(sentence)

    df = pd.DataFrame([{
        "sentence": sentence,
        "pos_seq": pos_seq,
        "sent_len": sent_len
    }])

    is_correct = ml_detector.predict(df)[0]

    print("detect here")

    # If ML predicts sentence is correct, return as-is

    if is_correct == 1:
        return sentence

    # ---------- Phase 3: Spelling correction ----------
    sentence = correct_spelling_sentence(sentence)
    print("spelling here")

    # ---------- Phase 2: Grammar rules ----------
    doc = nlp(sentence)
    print("grammar here")

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
