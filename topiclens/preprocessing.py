"""Turning raw abstracts into token strings, and token strings into matrices.

The pipeline is deliberately plain: clean, tokenize, lemmatize, filter. Two
choices deserve a note.

First, tokenization is a regular expression rather than ``nltk.word_tokenize``.
A bag-of-words model does not care about clitics or sentence boundaries, and on
twenty thousand abstracts the difference in runtime is minutes.

Second, the preprocessor emits a space-joined string of lemmas, and the
vectorizers split on whitespace. Passing a ``tokenizer=`` callable to
scikit-learn would do the same work but make the fitted vectorizer depend on
this module at unpickling time, which is a poor trade for a deployed app.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Any, Literal

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from topiclens.config import PreprocessingConfig
from topiclens.utils.logging_config import get_logger
from topiclens.utils.nltk_data import ensure_nltk_resources

logger = get_logger(__name__)

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
#: ``$...$`` and ``$$...$$`` spans; abstracts carry plenty of inline math.
_MATH_RE = re.compile(r"\$\$.+?\$\$|\$[^$]*\$", re.DOTALL)
#: A LaTeX command with optional star, bracket argument and brace argument.
_LATEX_COMMAND_RE = re.compile(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?(?:\{[^{}]*\})?")
_TOKEN_LETTERS_RE = re.compile(r"[A-Za-z]+(?:[-'][A-Za-z]+)*")
_TOKEN_ALNUM_RE = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*")

#: Vectorizer input is pre-tokenized, so "a token" is simply "a run of
#: non-space characters" — this also keeps hyphenated terms intact.
TOKEN_PATTERN = r"(?u)\S+"

#: Endings that look plural but are not: analysis, bias, corpus, chaos, class.
_NOT_PLURAL_ENDINGS = ("ss", "us", "is", "as", "os")


@lru_cache(maxsize=1)
def _lemmatizer() -> Any:
    ensure_nltk_resources()
    from nltk.stem import WordNetLemmatizer

    return WordNetLemmatizer()


def _fold_technical_plural(word: str) -> str:
    """Singularize a plural WordNet has never heard of.

    WordNet is a general-English lexicon, so it lemmatizes 'transformers' but
    leaves 'embeddings', 'datasets' and 'autoencoders' untouched — precisely the
    vocabulary that carries the topic signal here. Words WordNet does know are
    left alone, which is what keeps 'series' from becoming 'sery'.
    """
    if len(word) < 5 or not word.endswith("s") or word.endswith(_NOT_PLURAL_ENDINGS):
        return word

    from nltk.corpus import wordnet

    if wordnet.synsets(word):
        return word
    if word.endswith("ies"):
        return word[:-3] + "y"
    if word.endswith(("ches", "shes", "sses", "xes", "zes")):
        return word[:-2]
    return word[:-1]


@lru_cache(maxsize=200_000)
def lemmatize(word: str) -> str:
    """Reduce a word to its lemma, trying the noun reading before the verb one.

    Skipping the POS tagger costs some accuracy on ambiguous words but handles
    the two cases that matter here — plural nouns and inflected verbs. Domain
    plurals outside WordNet get a conservative fallback.
    """
    lemmatizer = _lemmatizer()
    noun = lemmatizer.lemmatize(word, pos="n")
    if noun != word:
        return noun
    verb = lemmatizer.lemmatize(word, pos="v")
    if verb != word:
        return verb
    return _fold_technical_plural(word)


def build_stopwords(config: PreprocessingConfig) -> frozenset[str]:
    """Combine NLTK's English list with the domain words from the config.

    The result is lemmatized as well: the config names ``results`` and
    ``methods``, but by the time filtering happens the text holds ``result``
    and ``method``, and an unlemmatized list would silently miss them.
    """
    ensure_nltk_resources()
    from nltk.corpus import stopwords

    words = {word.lower() for word in stopwords.words("english")}
    words.update(word.lower() for word in config.extra_stopwords)
    if config.lemmatize:
        words.update(lemmatize(word) for word in list(words))
    return frozenset(words)


class Preprocessor:
    """Configurable text cleaner and tokenizer."""

    def __init__(
        self, config: PreprocessingConfig, *, stopwords: frozenset[str] | None = None
    ) -> None:
        self.config = config
        self.stopwords = stopwords if stopwords is not None else build_stopwords(config)
        self._token_re = _TOKEN_LETTERS_RE if config.remove_numbers else _TOKEN_ALNUM_RE

    def clean(self, text: str) -> str:
        """Strip URLs and, if configured, LaTeX math and commands."""
        cleaned = _URL_RE.sub(" ", text)
        if self.config.remove_latex:
            cleaned = _MATH_RE.sub(" ", cleaned)
            cleaned = _LATEX_COMMAND_RE.sub(" ", cleaned)
        return " ".join(cleaned.split())

    def tokens(self, text: str) -> list[str]:
        """Clean, tokenize, lemmatize and filter one document."""
        cleaned = self.clean(text)
        raw_tokens = self._token_re.findall(cleaned)

        result: list[str] = []
        for token in raw_tokens:
            word = token.lower() if self.config.lowercase else token
            if len(word) < self.config.min_token_length:
                continue
            comparable = word.lower()
            if comparable in self.stopwords:
                continue
            if self.config.lemmatize:
                word = lemmatize(comparable)
                if word in self.stopwords or len(word) < self.config.min_token_length:
                    continue
            result.append(word)
        return result

    def transform(self, text: str) -> str:
        """Return the document as a space-joined string of tokens."""
        return " ".join(self.tokens(text))

    def transform_many(self, texts: list[str], *, show_progress: bool = False) -> list[str]:
        """Transform a corpus, optionally with a progress bar."""
        iterator: Any = texts
        if show_progress:
            from tqdm.auto import tqdm

            iterator = tqdm(texts, desc="preprocessing", unit="doc")

        processed = [self.transform(text) for text in iterator]
        empty = sum(1 for document in processed if not document)
        if empty:
            logger.warning("%d document(s) became empty after preprocessing", empty)
        logger.info(
            "Preprocessed %d documents, %.1f tokens on average",
            len(processed),
            sum(document.count(" ") + 1 for document in processed if document)
            / max(len(processed), 1),
        )
        return processed


def build_vectorizer(
    kind: Literal["count", "tfidf"], config: PreprocessingConfig
) -> CountVectorizer | TfidfVectorizer:
    """Create the vectorizer for a model family.

    LDA is a generative model of counts, so it gets raw frequencies; NMF and LSA
    work on TF-IDF weights, where rare informative terms are not drowned out.
    """
    if kind == "count":
        settings = config.vectorizer.count
        return CountVectorizer(
            max_features=settings.max_features,
            min_df=settings.min_df,
            max_df=settings.max_df,
            ngram_range=settings.ngram_range,
            token_pattern=TOKEN_PATTERN,
            lowercase=False,
        )

    settings = config.vectorizer.tfidf
    return TfidfVectorizer(
        max_features=settings.max_features,
        min_df=settings.min_df,
        max_df=settings.max_df,
        ngram_range=settings.ngram_range,
        sublinear_tf=settings.sublinear_tf,
        token_pattern=TOKEN_PATTERN,
        lowercase=False,
    )
