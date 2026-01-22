from src.rules import (
    subject_verb_agreement_rule,
    article_rule,
    tense_rule
)
import spacy

nlp = spacy.load("en_core_web_sm")

RULES = [
    subject_verb_agreement_rule,
    article_rule,
    tense_rule
]

def correct_sentence(sentence):
    doc = nlp(sentence)

    for rule in RULES:
        result = rule(doc)
        if result:
            return result

    return sentence
