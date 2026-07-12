"""Metrics for evaluating agent runs.

Each metric takes a per-question result record and returns a numeric score.
Metrics are separate small functions so they can be reused and tested.
"""
from __future__ import annotations

from typing import Iterable


def recall_at_k(retrieved_pmids: Iterable[str], expected_pmids: Iterable[str], k: int) -> float:
    """Fraction of expected PMIDs that appear in the top-k retrieved.

    If no PMIDs are expected, returns 1.0 (nothing to miss, nothing to hit).
    If k is 0 or retrieved is empty, returns 0.0 (unless expected is empty).
    """
    expected_set = set(expected_pmids)
    if not expected_set:
        return 1.0  # vacuously true

    retrieved_list = list(retrieved_pmids)[:k]
    if not retrieved_list:
        return 0.0

    hits = sum(1 for pmid in retrieved_list if pmid in expected_set)
    return hits / len(expected_set)


def citation_faithfulness(
    cited_pmids: Iterable[str],
    retrieved_pmids: Iterable[str],
) -> float:
    """Fraction of cited PMIDs that actually appear in the retrieved set.

    A perfect score (1.0) means every citation in the answer traces back to
    a paper the agent actually retrieved. A score < 1.0 indicates the agent
    is hallucinating citations — citing PMIDs it never actually pulled.

    If no citations were made, returns 1.0 (nothing to be unfaithful about).
    """
    cited_list = list(cited_pmids)
    if not cited_list:
        return 1.0

    retrieved_set = set(retrieved_pmids)
    grounded = sum(1 for pmid in cited_list if pmid in retrieved_set)
    return grounded / len(cited_list)


def citation_coverage(
    cited_pmids: Iterable[str],
    expected_pmids: Iterable[str],
) -> float:
    """Fraction of expected PMIDs that were actually cited in the answer.

    Different from recall_at_k: recall measures whether the agent *retrieved*
    the right papers; coverage measures whether it *used* them in the answer.
    An agent can retrieve well but cite badly (or vice versa).
    """
    expected_set = set(expected_pmids)
    if not expected_set:
        return 1.0

    cited_list = list(cited_pmids)
    if not cited_list:
        return 0.0

    hits = sum(1 for pmid in cited_list if pmid in expected_set)
    return hits / len(expected_set)