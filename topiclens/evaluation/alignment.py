"""Agreement between discovered topics and the corpus's own labels.

Most topic-modeling projects stop at intrinsic metrics because no ground truth
exists. Here it does: every arXiv paper carries a primary category. Treating the
dominant topic of each document as a cluster assignment turns the comparison
into a standard clustering-evaluation problem.

The absolute numbers should not be over-read — ten topics cannot map onto five
categories one-to-one, and a good topic model is not supposed to reproduce a
taxonomy. What the numbers are good for is comparing models against each other
under identical conditions.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def cluster_purity(assignments: Sequence[int], labels: Sequence[str]) -> float:
    """Share of documents whose label matches the majority label of their topic."""
    assignments = np.asarray(assignments)
    labels = np.asarray(labels)
    if assignments.size == 0:
        return 0.0

    correct = 0
    for topic in np.unique(assignments):
        members = labels[assignments == topic]
        if members.size:
            _, counts = np.unique(members, return_counts=True)
            correct += int(counts.max())
    return correct / assignments.size


def alignment_scores(assignments: Sequence[int], labels: Sequence[str]) -> dict[str, float]:
    """NMI, ARI and purity for one model's dominant-topic assignment.

    NMI answers how much knowing the topic tells you about the category; ARI
    corrects for chance agreement; purity is the readable one for a report.

    Raises:
        ValueError: if the two sequences have different lengths.
    """
    if len(assignments) != len(labels):
        raise ValueError(
            f"got {len(assignments)} assignments for {len(labels)} labels; they must match"
        )
    if not len(assignments):
        return {"nmi": 0.0, "ari": 0.0, "purity": 0.0}

    return {
        "nmi": float(normalized_mutual_info_score(labels, assignments)),
        "ari": float(adjusted_rand_score(labels, assignments)),
        "purity": float(cluster_purity(assignments, labels)),
    }


def topic_label_distribution(
    assignments: Sequence[int], labels: Sequence[str], n_topics: int
) -> dict[int, dict[str, int]]:
    """Per-topic breakdown of the true labels, for tables and stacked bars."""
    assignments = np.asarray(assignments)
    labels = np.asarray(labels)

    distribution: dict[int, dict[str, int]] = {}
    for topic in range(n_topics):
        members = labels[assignments == topic]
        values, counts = np.unique(members, return_counts=True)
        distribution[topic] = {
            str(value): int(count) for value, count in zip(values, counts, strict=True)
        }
    return distribution
