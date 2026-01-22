# Grammar-autocorrector
Modular grammar auto-correction system using rule-based NLP and transformers.

**Problem Statement**

Many English learners and casual writers make basic grammatical errors such as spelling mistakes, incorrect articles, and subject–verb agreement errors.
This project builds a modular grammar auto-correction system that detects and corrects simple grammatical errors using a combination of rule-based NLP, statistical methods, and transformer models.


## Scope
### Supported
- Spelling errors
- Subject–verb agreement
- Article errors
- Simple tense errors

### Not Supported (yet)
- Style rewriting
- Paraphrasing
- Complex discourse-level grammar

## Project Structure
(brief tree)

## Approach
- Rule-based grammar correction
- Statistical spell checking
- Machine learning for error detection
- Transformer-based correction (later phase)

## Learning Goals
- Error taxonomy in NLP
- Precision vs Recall trade-offs
- Hybrid NLP systems

## Status
🚧 Phase 0: Project setup & scoping

In grammar correction, incorrect corrections are worse than missed corrections.

## Consolidated Error Patterns

| Error Type | Linguistic Signal | Detection Level |
|-----------|------------------|----------------|
| SPELL | Non-dictionary token | Token |
| SVA | Singular subject + base verb | Dependency |
| ARTICLE | Noun without determiner | Dependency |
| TENSE | Past time word + non-past verb | POS + Dependency |
| VERB FORM | Auxiliary misuse / wrong verb POS | POS |

IF token not found in dictionary
THEN candidate spelling error

IF subject (nsubj) is singular
AND verb is base form (VB)
THEN subject–verb agreement error

IF noun (NN) has no determiner child (det)
AND noun is countable
THEN missing article error

IF sentence contains past-time modifier (yesterday, last, ago)
AND main verb is not past tense (VBD)
THEN tense error

IF auxiliary verb present
AND main verb POS does not match auxiliary
THEN verb form error

## Rule-based vs ML-based Decisions

| Error Type | Approach | Reason |
|-----------|--------|--------|
| SPELL | Rule-based | Dictionary & edit distance works well |
| SVA | Rule-based | Clear dependency patterns |
| ARTICLE | Hybrid | Simple cases rule-based, others contextual |
| TENSE | Rule-based (initial) | Time-word patterns detectable |
| VERB FORM | ML-based (later) | Context-sensitive, ambiguous |

## Phase 2 Strategy (Locked)

Phase 2 will implement:
- Rule-based SPELL correction
- Rule-based SVA correction
- Rule-based ARTICLE correction (simple cases only)
- Rule-based TENSE correction using time expressions

ML-based correction is deferred to Phase 4.

## Design Principle: Precision over Recall

Incorrect corrections degrade user trust more than missed corrections.
Therefore, rules will only fire when confidence is high.


## Phase 1 Summary

- Identified core grammar error types
- Analyzed errors at token, POS, and dependency levels
- Defined formal detection patterns
- Chose rule-based vs ML-based strategies
- Locked Phase 2 implementation plan


## Rule-Based System Limitations

- Over-simplified verb inflections
- No handling of irregular verbs
- Article rules are heuristic-based
- Rules fire conservatively to avoid over-correction

## Spelling Correction Limitations

- Context-free spelling correction
- Proper nouns may be incorrectly corrected
- Grammar-aware spelling is not handled yet
- Conservative correction preferred to avoid noise
