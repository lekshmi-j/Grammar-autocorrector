"""
transformer_corrector.py

Transformer-based grammar correction using a pretrained T5 model.
Used as a fallback when rule-based and ML-based correction is insufficient.
"""

import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

# -------------------- Model Setup (GLOBAL, LOAD ONCE) --------------------

MODEL_NAME = "t5-small"

tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)

model.eval()  # inference mode


# -------------------- Prompt Builder --------------------

def build_prompt(sentence: str) -> str:
    """
    Build instruction-style prompt for T5.
    """
    return f"correct grammar: {sentence}"


# -------------------- Core Correction Function --------------------

def transformer_correct(sentence: str, max_length: int = 64) -> str:
    """
    Correct grammar using a pretrained transformer.

    Parameters
    ----------
    sentence : str
        Input sentence
    max_length : int
        Maximum output length

    Returns
    -------
    str
        Grammar-corrected sentence
    """

    input_text = build_prompt(sentence)

    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        max_length=64
    )

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            num_beams=4,
            early_stopping=True
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)
