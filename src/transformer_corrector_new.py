"""
transformer_corrector.py

Safe transformer-based grammar correction for Streamlit.
Uses lazy loading + caching to avoid segmentation faults.
"""

import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer


def load_transformer(model_name="t5-small"):
    """
    Lazily load transformer model and tokenizer.
    """
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    model.eval()
    model.to("cpu")  # FORCE CPU

    return tokenizer, model


def transformer_correct(sentence, tokenizer, model, max_length=64):
    """
    Grammar correction using pretrained transformer.
    """
    prompt = f"correct grammar: {sentence}"

    inputs = tokenizer(
        prompt,
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
