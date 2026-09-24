"""One-time setup of the NLTK resources the pipeline relies on.

Only two are needed: the English stop-word list and WordNet for lemmatization.
Sentence tokenizers and taggers are deliberately avoided — a bag-of-words
pipeline does not need them, and every extra resource is one more thing to
download on a fresh machine or a deployment slot.
"""

from __future__ import annotations

from topiclens.utils.logging_config import get_logger

logger = get_logger(__name__)

#: Package name -> the lookup paths that indicate it is already installed.
NLTK_RESOURCES = {
    "stopwords": ("corpora/stopwords", "corpora/stopwords.zip"),
    "wordnet": ("corpora/wordnet", "corpora/wordnet.zip"),
}


def _is_installed(paths: tuple[str, ...]) -> bool:
    import nltk

    for path in paths:
        try:
            nltk.data.find(path)
        except LookupError:
            continue
        return True
    return False


def ensure_nltk_resources(*, quiet: bool = True) -> list[str]:
    """Download any missing resource; return the names that were fetched."""
    import nltk

    downloaded: list[str] = []
    for package, paths in NLTK_RESOURCES.items():
        if _is_installed(paths):
            continue
        logger.info("Downloading NLTK resource %r", package)
        nltk.download(package, quiet=quiet)
        downloaded.append(package)
    return downloaded
