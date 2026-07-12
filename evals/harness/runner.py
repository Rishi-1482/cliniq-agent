"""Eval harness: run the agent on a question and record a result record."""
from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent.graph import compiled_graph


async def run_one_question(question_record: dict[str, Any]) -> dict[str, Any]:
    """Run the agent on a single eval question and return a result record.

    Returns a dict with the input, the agent's outputs, and timing/cost
    fingerprints. Does NOT compute metrics — that's a separate step so we
    can re-score without re-running the agent.
    """
    thread_id = f"eval-{question_record['id']}-{uuid.uuid4().hex[:8]}"
    start = time.perf_counter()

    async with compiled_graph() as graph:
        state = await graph.ainvoke(
            {"question": question_record["question"]},
            config={"configurable": {"thread_id": thread_id}},
        )

    elapsed = time.perf_counter() - start

    findings = state.get("findings", [])
    citations = state.get("citations", [])

    return {
        "question_id": question_record["id"],
        "question": question_record["question"],
        "category": question_record.get("category", "uncategorized"),
        "expected_relevant_pmids": question_record.get("expected_relevant_pmids", []),
        # Retrieval artifacts
        "retrieved_pmids": state.get("retrieved_pmids", []),
        "indexed_pmids": state.get("indexed_pmids", []),
        "top_findings_pmids": [f["pmid"] for f in findings],
        # Answer artifacts
        "final_answer": state.get("final_answer", ""),
        "cited_pmids": [c["pmid"] for c in citations],
        # Process
        "iterations": state.get("iteration", 0),
        "enough_evidence": state.get("enough_evidence", False),
        "elapsed_seconds": round(elapsed, 2),
        # Bookkeeping
        "thread_id": thread_id,
        "ran_at": datetime.now(timezone.utc).isoformat(),
    }


async def run_eval_set(
    questions_path: str | Path,
    results_dir: str | Path = "evals/results",
) -> Path:
    """Run the agent on every question in a JSON file and write results to disk."""
    questions_path = Path(questions_path)
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    with open(questions_path) as f:
        questions = json.load(f)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    output_path = results_dir / f"run-{run_id}.jsonl"

    with open(output_path, "w") as f:
        for i, q in enumerate(questions, 1):
            print(f"[{i}/{len(questions)}] Running {q['id']}: {q['question'][:60]}...")
            try:
                result = await run_one_question(q)
            except Exception as e:
                print(f"  ERROR on {q['id']}: {type(e).__name__}: {e}")
                result = {
                    "question_id": q["id"],
                    "question": q["question"],
                    "error": f"{type(e).__name__}: {e}",
                }
            f.write(json.dumps(result) + "\n")
            f.flush()  # so partial runs are recoverable

    print(f"\nDone. Results: {output_path}")
    return output_path