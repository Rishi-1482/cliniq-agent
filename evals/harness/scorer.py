"""Score a completed eval run using metrics.py. Runs offline on saved results."""
from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Any

from evals.harness.metrics import citation_coverage, citation_faithfulness, recall_at_k


def score_result(result: dict[str, Any], k: int = 10) -> dict[str, float]:
    """Compute per-question metrics from a result record."""
    if "error" in result:
        return {"error": True}

    expected = result.get("expected_relevant_pmids", [])
    retrieved = result.get("retrieved_pmids", [])
    cited = result.get("cited_pmids", [])

    return {
        "recall_at_k": recall_at_k(retrieved, expected, k=k),
        "citation_faithfulness": citation_faithfulness(cited, retrieved),
        "citation_coverage": citation_coverage(cited, expected),
        "iterations": result.get("iterations", 0),
        "elapsed_seconds": result.get("elapsed_seconds", 0),
    }


def score_run(results_path: str | Path, k: int = 10) -> dict[str, Any]:
    """Score all results in a run file and produce aggregates."""
    results_path = Path(results_path)
    with open(results_path) as f:
        results = [json.loads(line) for line in f]

    scored = []
    for r in results:
        per_q = {"question_id": r.get("question_id"), "category": r.get("category")}
        per_q.update(score_result(r, k=k))
        scored.append(per_q)

    # Aggregate (skip errored questions)
    valid = [s for s in scored if not s.get("error")]
    if not valid:
        return {"per_question": scored, "aggregate": {"error": "no valid results"}}

    aggregate = {
        "n_valid": len(valid),
        "n_errored": len(scored) - len(valid),
        "mean_recall_at_k": round(mean(s["recall_at_k"] for s in valid), 3),
        "mean_citation_faithfulness": round(mean(s["citation_faithfulness"] for s in valid), 3),
        "mean_citation_coverage": round(mean(s["citation_coverage"] for s in valid), 3),
        "mean_iterations": round(mean(s["iterations"] for s in valid), 2),
        "mean_elapsed_seconds": round(mean(s["elapsed_seconds"] for s in valid), 1),
        "k": k,
    }

    return {"per_question": scored, "aggregate": aggregate}