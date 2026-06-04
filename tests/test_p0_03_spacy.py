"""Smoke test for spaCy installation."""

import spacy


def test_spacy_loads() -> None:
    """Verify en_core_web_lg loads without error."""
    nlp = spacy.load("en_core_web_lg")
    assert nlp is not None


def test_spacy_lemmatization() -> None:
    """Verify lemmatization (key feature for vocabulary tracking)."""
    nlp = spacy.load("en_core_web_lg")
    doc = nlp("The studies showed surprising results.")
    lemmas = [token.lemma_ for token in doc if token.is_alpha]
    assert "study" in lemmas
    assert "show" in lemmas
    assert "result" in lemmas


def test_spacy_pos_tagging() -> None:
    """Verify POS tagging works."""
    nlp = spacy.load("en_core_web_lg")
    doc = nlp("Cats are running fast.")
    pos_tags = {token.text.lower(): token.pos_ for token in doc if token.is_alpha}
    assert pos_tags.get("cats") == "NOUN"
    assert pos_tags.get("running") == "VERB"
    assert pos_tags.get("fast") == "ADV"
