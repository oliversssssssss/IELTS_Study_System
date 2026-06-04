"""Unit tests for the spaCy NLP processor wrapper.

All public API tests patch `_get_nlp` so the real 750MB spaCy model is not
loaded during unit tests.
"""

from collections.abc import Iterator
from dataclasses import dataclass

from src.ielts.nlp import processor
from src.ielts.nlp.processor import (
    extract_unique_lemmas,
    split_sentences,
    tokenize_with_lemma,
)


@dataclass(frozen=True)
class FakeSpan:
    """Minimal sentence span test double."""

    text: str


@dataclass(frozen=True)
class FakeToken:
    """Minimal spaCy Token test double."""

    text: str
    lemma_: str
    pos_: str
    tag_: str
    is_stop: bool
    is_alpha: bool
    idx: int
    sent: FakeSpan


class FakeDoc:
    """Minimal iterable spaCy Doc test double."""

    def __init__(self, tokens: list[FakeToken], sentences: list[FakeSpan]) -> None:
        self._tokens = tokens
        self.sents = sentences

    def __iter__(self) -> Iterator[FakeToken]:
        return iter(self._tokens)


class FakeNlp:
    """Callable spaCy pipeline test double."""

    def __init__(self, doc: FakeDoc) -> None:
        self.doc = doc
        self.calls: list[str] = []

    def __call__(self, text: str) -> FakeDoc:
        self.calls.append(text)
        return self.doc


class FakeLoadedNlp:
    """spaCy pipeline double used for singleton loading tests."""

    def __init__(self) -> None:
        self.pipe_names: list[str] = []
        self.added_pipes: list[str] = []

    def add_pipe(self, name: str) -> None:
        self.pipe_names.append(name)
        self.added_pipes.append(name)


def _make_doc() -> FakeDoc:
    """Build a reusable document with stop words, alpha words, and punctuation."""
    sentence = FakeSpan("The studies showed surprising results.")
    tokens = [
        FakeToken("The", "the", "DET", "DT", True, True, 0, sentence),
        FakeToken("studies", "study", "NOUN", "NNS", False, True, 4, sentence),
        FakeToken("showed", "show", "VERB", "VBD", False, True, 12, sentence),
        FakeToken("surprising", "surprising", "ADJ", "JJ", False, True, 19, sentence),
        FakeToken("results", "result", "NOUN", "NNS", False, True, 30, sentence),
        FakeToken(".", "", "PUNCT", ".", False, False, 37, sentence),
    ]
    return FakeDoc(tokens, [sentence])


def test_tokenize_with_lemma_returns_serializable_token_dicts(mocker) -> None:
    doc = _make_doc()
    nlp = FakeNlp(doc)
    mocker.patch("src.ielts.nlp.processor._get_nlp", return_value=nlp)

    result = tokenize_with_lemma("The studies showed surprising results.")

    assert nlp.calls == ["The studies showed surprising results."]
    assert result == [
        {
            "text": "The",
            "lemma": "the",
            "pos": "DET",
            "tag": "DT",
            "is_stop": True,
            "is_alpha": True,
            "sentence": "The studies showed surprising results.",
            "idx": 0,
        },
        {
            "text": "studies",
            "lemma": "study",
            "pos": "NOUN",
            "tag": "NNS",
            "is_stop": False,
            "is_alpha": True,
            "sentence": "The studies showed surprising results.",
            "idx": 4,
        },
        {
            "text": "showed",
            "lemma": "show",
            "pos": "VERB",
            "tag": "VBD",
            "is_stop": False,
            "is_alpha": True,
            "sentence": "The studies showed surprising results.",
            "idx": 12,
        },
        {
            "text": "surprising",
            "lemma": "surprising",
            "pos": "ADJ",
            "tag": "JJ",
            "is_stop": False,
            "is_alpha": True,
            "sentence": "The studies showed surprising results.",
            "idx": 19,
        },
        {
            "text": "results",
            "lemma": "result",
            "pos": "NOUN",
            "tag": "NNS",
            "is_stop": False,
            "is_alpha": True,
            "sentence": "The studies showed surprising results.",
            "idx": 30,
        },
        {
            "text": ".",
            "lemma": ".",
            "pos": "PUNCT",
            "tag": ".",
            "is_stop": False,
            "is_alpha": False,
            "sentence": "The studies showed surprising results.",
            "idx": 37,
        },
    ]


def test_split_sentences_strips_and_skips_empty_sentences(mocker) -> None:
    sentences = [FakeSpan(" First sentence. "), FakeSpan("Second sentence."), FakeSpan("  ")]
    doc = FakeDoc([], sentences)
    nlp = FakeNlp(doc)
    mocker.patch("src.ielts.nlp.processor._get_nlp", return_value=nlp)

    result = split_sentences("First sentence. Second sentence.")

    assert result == ["First sentence.", "Second sentence."]
    assert nlp.calls == ["First sentence. Second sentence."]


def test_extract_unique_lemmas_excludes_stop_words_and_punctuation_by_default(mocker) -> None:
    nlp = FakeNlp(_make_doc())
    mocker.patch("src.ielts.nlp.processor._get_nlp", return_value=nlp)

    result = extract_unique_lemmas("The studies showed surprising results.")

    assert result == {
        "study": [
            {
                "original": "studies",
                "sentence": "The studies showed surprising results.",
                "pos": "NOUN",
            }
        ],
        "show": [
            {
                "original": "showed",
                "sentence": "The studies showed surprising results.",
                "pos": "VERB",
            }
        ],
        "surprising": [
            {
                "original": "surprising",
                "sentence": "The studies showed surprising results.",
                "pos": "ADJ",
            }
        ],
        "result": [
            {
                "original": "results",
                "sentence": "The studies showed surprising results.",
                "pos": "NOUN",
            }
        ],
    }


def test_extract_unique_lemmas_can_include_stop_words_and_punctuation(mocker) -> None:
    nlp = FakeNlp(_make_doc())
    mocker.patch("src.ielts.nlp.processor._get_nlp", return_value=nlp)

    result = extract_unique_lemmas(
        "The studies showed surprising results.",
        include_stop=True,
        include_punct=True,
    )

    assert "the" in result
    assert "." in result
    assert result["."] == [
        {
            "original": ".",
            "sentence": "The studies showed surprising results.",
            "pos": "PUNCT",
        }
    ]


def test_extract_unique_lemmas_groups_repeated_lemmas(mocker) -> None:
    sentence = FakeSpan("Results show more results.")
    tokens = [
        FakeToken("Results", "result", "NOUN", "NNS", False, True, 0, sentence),
        FakeToken("show", "show", "VERB", "VBP", False, True, 8, sentence),
        FakeToken("more", "more", "ADJ", "JJR", True, True, 13, sentence),
        FakeToken("results", "result", "NOUN", "NNS", False, True, 18, sentence),
    ]
    nlp = FakeNlp(FakeDoc(tokens, [sentence]))
    mocker.patch("src.ielts.nlp.processor._get_nlp", return_value=nlp)

    result = extract_unique_lemmas("Results show more results.")

    assert [item["original"] for item in result["result"]] == ["Results", "results"]
    assert "more" not in result


def test_get_nlp_loads_spacy_once_and_adds_sentencizer(mocker) -> None:
    loaded_nlp = FakeLoadedNlp()
    spacy_load = mocker.patch("src.ielts.nlp.processor.spacy.load", return_value=loaded_nlp)
    processor._get_nlp.cache_clear()

    try:
        first = processor._get_nlp()
        second = processor._get_nlp()
    finally:
        processor._get_nlp.cache_clear()

    assert first is loaded_nlp
    assert second is loaded_nlp
    spacy_load.assert_called_once_with(
        processor.SPACY_MODEL,
        exclude=list(processor.EXCLUDED_PIPELINES),
    )
    assert loaded_nlp.added_pipes == [processor.SENTENCIZER_NAME]


def setup_function() -> None:
    processor._get_nlp.cache_clear()


def teardown_function() -> None:
    processor._get_nlp.cache_clear()
