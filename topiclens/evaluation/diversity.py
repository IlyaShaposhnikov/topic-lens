"""How distinct the topics are from one another.

Coherence alone is easy to game: a model that collapses every topic onto the
same handful of words scores well and says nothing. Diversity is the check that
keeps it honest, and the two are meant to be read together.
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import combinations

import numpy as np


def topic_diversity(topics: Sequence[Sequence[str]], top_n: int | None = None) -> float:
    """Share of distinct words among all topic top words.

    1.0 means no topic repeats another's vocabulary; values near 1/len(topics)
    mean the topics have collapsed onto each other.
    """
    if not topics:
        return 0.0
    words = [word for topic in topics for word in list(topic)[: top_n or len(topic)]]
    return len(set(words)) / len(words) if words else 0.0


def pairwise_overlap(topics: Sequence[Sequence[str]], top_n: int | None = None) -> float:
    """Mean Jaccard overlap between pairs of topics — the flip side of diversity.

    Diversity is a single corpus-level number; this one localizes the problem,
    since two near-identical topics among ten barely move the diversity score.
    """
    if len(topics) < 2:
        return 0.0

    vocabularies = [set(list(topic)[: top_n or len(topic)]) for topic in topics]
    scores = [
        len(first & second) / len(first | second)
        for first, second in combinations(vocabularies, 2)
        if first or second
    ]
    return float(np.mean(scores)) if scores else 0.0


def most_similar_pair(
    topics: Sequence[Sequence[str]], top_n: int | None = None
) -> tuple[int, int, float]:
    """The two topics that overlap most, with their Jaccard score.

    Useful in reports: it names the redundancy instead of only measuring it.
    """
    if len(topics) < 2:
        return (0, 0, 0.0)

    vocabularies = [set(list(topic)[: top_n or len(topic)]) for topic in topics]
    worst = (0, 1, -1.0)
    for first, second in combinations(range(len(vocabularies)), 2):
        union = vocabularies[first] | vocabularies[second]
        if not union:
            continue
        score = len(vocabularies[first] & vocabularies[second]) / len(union)
        if score > worst[2]:
            worst = (first, second, float(score))
    return worst
