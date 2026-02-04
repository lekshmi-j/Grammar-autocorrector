import streamlit as st
import pandas as pd
import difflib

from src.corrector import correct_sentence
import torch
from src.transformer_corrector_new import load_transformer, transformer_correct

from src.ml_detector import load_detector


# -------------------- Page Config --------------------

st.set_page_config(
    page_title="Grammar Auto-Corrector",
    page_icon="✍️",
    layout="centered"
)
@st.cache_resource(show_spinner="Loading transformer model...")
def get_transformer():
    return load_transformer()

import spacy

@st.cache_resource
def get_nlp():
    return spacy.load("en_core_web_sm")

def extract_pos_sequence(sentence: str) -> str:
    nlp = get_nlp()
    doc = nlp(sentence)
    return " ".join(tok.pos_ for tok in doc)



# -------------------- Load ML Detector (ONCE) --------------------

ml_detector = load_detector()


# -------------------- Helper: Highlight Changes --------------------

def highlight_changes(original, corrected):
    original_words = original.split()
    corrected_words = corrected.split()

    diff = difflib.ndiff(original_words, corrected_words)

    highlighted = []
    for word in diff:
        if word.startswith("+ "):
            highlighted.append(f"🟢 **{word[2:]}**")
        elif word.startswith("- "):
            highlighted.append(f"🔴 ~~{word[2:]}~~")
        elif word.startswith("  "):
            highlighted.append(word[2:])

    return " ".join(highlighted)


# -------------------- UI --------------------

st.title("✍️ Grammar Auto-Corrector")
st.write(
    "A hybrid grammar correction system using rules, ML, and transformers."
)

user_input = st.text_area(
    "Enter a sentence:",
    placeholder="She go to school yesterday",
    height=120
)

use_transformer = st.checkbox(
    "Use transformer fallback (more fluent, less explainable)",
    value=False
)

# -------------------- Button Logic --------------------

if st.button("Correct Grammar"):
    if not user_input.strip():
        st.warning("Please enter a sentence.")
    else:
        # ✅ FIX: Always pass a DataFrame to sklearn
        pos_seq = extract_pos_sequence(user_input)

        input_df = pd.DataFrame({
            "sentence": [user_input],
            "pos_seq": [pos_seq],
            "sent_len": [len(user_input)]
        })


        # ML gate
        is_correct = ml_detector.predict(input_df)[0]

        if is_correct == 1:
            corrected = user_input
            st.info("Sentence appears grammatically correct.")
        else:
            corrected = correct_sentence(user_input)

            if use_transformer:
                tokenizer, model = get_transformer()
                corrected = transformer_correct(corrected, tokenizer, model)


        st.subheader("Corrected Sentence")
        st.success(corrected)

        st.subheader("Highlighted Changes")
        st.markdown(
            highlight_changes(user_input, corrected),
            unsafe_allow_html=True
        )


# -------------------- Footer --------------------

st.markdown("---")
st.caption(
    "Built with Rule-based NLP, ML-based error detection, and Transformer models."
)
