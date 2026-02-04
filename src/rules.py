"""
rules.py

Rule-based grammar correction functions.
Currently implemented:
- Subject–Verb Agreement (SVA)
"""

import spacy

# Load spaCy model once
nlp = spacy.load("en_core_web_sm")


def subject_verb_agreement_rule(doc):
    """
    Fixes subject–verb agreement errors for singular third-person subjects.

    Example:
    Input : He go to market
    Output: He goes to market

    Returns:
    - corrected sentence (str)
    - None if no correction needed
    """
    
    for token in doc:

        # Step 1: Identify nominal subject
        if token.dep_ != "nsubj":
            continue

        subject = token
        verb = token.head

        # Step 2: Ensure head is a verb
        if verb.pos_ != "VERB":
            continue

        # Step 3: Check if subject is singular third-person
        singular_pronouns = {"he", "she", "it"}

        is_singular_subject = (
            subject.text.lower() in singular_pronouns
            or subject.morph.get("Person") == ["3"]
            and subject.morph.get("Number") != ["Plur"]
        )

        # Step 4: Check if verb form is wrong
        # VBP / VB are wrong for 3rd person singular
        is_wrong_verb_form = verb.tag_ in {"VB", "VBP"}

        if is_singular_subject and is_wrong_verb_form:

            # Step 5: Simple correction (regular verbs)
            corrected_verb = verb.text + "s"

            # Replace only first occurrence
            corrected_sentence = doc.text.replace(
                verb.text,
                corrected_verb,
                1
            )

            return corrected_sentence

    return None

def article_rule(doc):
    for token in doc:
        if token.pos_ == "NOUN":
            has_det = any(child.dep_ == "det" for child in token.children)

            if not has_det:
                # simple heuristic: use "a"
                corrected_sentence = doc.text.replace(
                    token.text, "a " + token.text, 1
                )
                return corrected_sentence

    return None



def tense_rule(doc):
    PAST_TIME_WORDS = {"yesterday", "ago", "last"}
    tokens_lower = {t.text.lower() for t in doc}

    if tokens_lower & PAST_TIME_WORDS:
        for token in doc:
            #print(token," ",token.dep_," ",token.tag_)
            if token.dep_ == "ROOT" and token.tag_ == "VBP":
                corrected_verb = token.text + "ed"
                corrected_sentence = doc.text.replace(
                    token.text, corrected_verb, 1
                )
                return corrected_sentence

    return None

