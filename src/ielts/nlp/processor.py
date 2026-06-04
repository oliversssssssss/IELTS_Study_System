"""Unified spaCy NLP wrapper for IELTS text processing."""

from functools import lru_cache

import spacy
from spacy.language import Language
from spacy.tokens import Token

SPACY_MODEL = "en_core_web_lg"
EXCLUDED_PIPELINES = ("ner", "parser")
SENTENCIZER_NAME = "sentencizer"


@lru_cache(maxsize=1)
def _get_nlp() -> Language:
    """Lazily load and cache the spaCy English model.

    The model is large, so callers must go through this cached getter instead
    of loading spaCy at import time or inside every public function call.

    Returns:
        Cached spaCy language pipeline configured for tokenization, POS tagging,
        lemmatization, and sentence splitting.
    """
    nlp = spacy.load(SPACY_MODEL, exclude=list(EXCLUDED_PIPELINES))
    if SENTENCIZER_NAME not in nlp.pipe_names:
        nlp.add_pipe(SENTENCIZER_NAME)
    return nlp


def _normalized_lemma(token: Token) -> str:
    """Return a lower-case lemma, falling back to lower-case token text."""
    lemma = token.lemma_.strip().lower()
    if lemma:
        return lemma
    return token.text.lower()


def _sentence_text(token: Token) -> str:
    """Return the sentence text for a token without surrounding whitespace."""
    return token.sent.text.strip()


def tokenize_with_lemma(text: str) -> list[dict]:
    """Tokenize English text and attach lemma, POS, and sentence metadata.

    Args:
        text: English text to process.

    Returns:
        A list of serializable token dictionaries. Each dictionary contains at
        least text, lemma, pos, tag, is_stop, is_alpha, sentence, and idx.
    """
    doc = _get_nlp()(text)
    return [
        {
            "text": token.text,
            "lemma": _normalized_lemma(token),
            "pos": token.pos_,
            "tag": token.tag_,
            "is_stop": token.is_stop,
            "is_alpha": token.is_alpha,
            "sentence": _sentence_text(token),
            "idx": token.idx,
        }
        for token in doc
    ]


def split_sentences(text: str) -> list[str]:
    """Split English text into sentence strings.

    Args:
        text: English text to process.

    Returns:
        Sentence texts, stripped of surrounding whitespace.
    """
    doc = _get_nlp()(text)
    return [sentence.text.strip() for sentence in doc.sents if sentence.text.strip()]


def extract_unique_lemmas(
    text: str,
    *,
    include_stop: bool = False,
    include_punct: bool = False,
) -> dict[str, list[dict]]:
    """Extract unique lemmas and preserve every original occurrence.

    Args:
        text: English text to process.
        include_stop: Whether to include spaCy stop words.
        include_punct: Whether to include non-alphabetic tokens such as
            punctuation and numbers.

    Returns:
        Mapping of lemma to occurrence dictionaries containing original,
        sentence, and pos.
    """
    lemma_occurrences: dict[str, list[dict]] = {}
    for token in tokenize_with_lemma(text):
        if not include_stop and token["is_stop"]:
            continue
        if not include_punct and not token["is_alpha"]:
            continue

        lemma = token["lemma"]
        lemma_occurrences.setdefault(lemma, []).append(
            {
                "original": token["text"],
                "sentence": token["sentence"],
                "pos": token["pos"],
            }
        )
    return lemma_occurrences
