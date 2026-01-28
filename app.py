import streamlit as st

from src.corrector import correct_sentence
from src.transformer_corrector import transformer_correct
from src.ml_detector import load_detector

import difflib


# -------------------- Page Config --------------------

st.set_page_config(
    page_title="Grammar Auto-Corrector",
    page_icon="✍️",
    layout="centered"
)


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


# -------------------- Load ML Detector --------------------

ml_detector = load_detector()


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

if st.button("Correct Grammar"):
    if not user_input.strip():
        st.warning("Please enter a sentence.")
    else:
        # ML gate
        is_correct = ml_detector.predict(
            [{"sentence": user_input}]
        )[0]

        if is_correct == 1:
            corrected = user_input
            st.info("Sentence appears grammatically correct.")
        else:
            corrected = correct_sentence(user_input)

            if use_transformer:
                corrected = transformer_correct(corrected)

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
