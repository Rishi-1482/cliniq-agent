"""Tests for eval metrics."""
from evals.harness.metrics import citation_coverage, citation_faithfulness, recall_at_k


def test_recall_at_k_perfect():
    assert recall_at_k(["1", "2", "3"], ["1", "2", "3"], k=3) == 1.0


def test_recall_at_k_partial():
    # 2 of 3 expected found in top-3
    assert recall_at_k(["1", "2", "9"], ["1", "2", "3"], k=3) == 2 / 3


def test_recall_at_k_respects_k():
    # Correct PMID present but beyond k
    assert recall_at_k(["9", "8", "7", "1"], ["1"], k=3) == 0.0
    assert recall_at_k(["9", "8", "7", "1"], ["1"], k=4) == 1.0


def test_recall_at_k_empty_expected_is_vacuously_true():
    assert recall_at_k(["1", "2"], [], k=5) == 1.0


def test_citation_faithfulness_hallucination_detected():
    # Cited "99" but never retrieved it — unfaithful citation
    assert citation_faithfulness(["1", "2", "99"], ["1", "2", "3"]) == 2 / 3


def test_citation_faithfulness_no_citations_is_vacuous():
    assert citation_faithfulness([], ["1", "2"]) == 1.0


def test_citation_coverage_distinct_from_recall():
    # Agent retrieved everything but only cited 1 of the 3 expected papers
    retrieved = ["1", "2", "3", "10", "20"]
    cited = ["1"]
    expected = ["1", "2", "3"]
    # Retrieval was perfect; coverage is not
    assert recall_at_k(retrieved, expected, k=5) == 1.0
    assert citation_coverage(cited, expected) == 1 / 3