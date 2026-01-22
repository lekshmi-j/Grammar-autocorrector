"""
spelling.py

Statistical spelling correction module using:
- Edit distance–based candidate generation
- Frequency-based candidate ranking (Noisy Channel Model)

This module is intentionally conservative to avoid over-correction.
"""

import nltk
from nltk.tokenize import word_tokenize
from spellchecker import SpellChecker

# Ensure tokenizer is available
nltk.download("punkt", quiet=True)

spell = SpellChecker()

def generate_candidates(word):
    return spell.candidates(word)

def rank_candidates(candidates):
    """
    Selects the best spelling candidate based on word frequency.

    Parameters:
    candidates (list): A list of possible corrected spellings

    Returns:
    str: The candidate with the highest word frequency
    """

    best_word = None
    highest_frequency = -1

    # Check each candidate word
    for word in candidates:
        # Get how often this word appears in language data
        frequency = spell.word_frequency[word]

        # Update best word if this one is more frequent
        if frequency > highest_frequency:
            highest_frequency = frequency
            best_word = word

    return best_word

def correct_spelling_word(word):
    if word in spell:
        return word

    candidates = generate_candidates(word)
    if not candidates:
        return word

    return rank_candidates(candidates)

def correct_spelling_sentence(sentence):
    tokens = word_tokenize(sentence)
    corrected_tokens = []

    for token in tokens:
        if token.isalpha():
            corrected = correct_spelling_word(token.lower())
            corrected_tokens.append(corrected)
        else:
            corrected_tokens.append(token)

    return " ".join(corrected_tokens)
