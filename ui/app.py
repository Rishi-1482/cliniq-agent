"""ClinIQ Agent — FastAPI web UI."""
from __future__ import annotations

import sys
from pathlib import Path

# Bootstrap: ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

import asyncio
import json
from typing import Any

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

from agent.graph import compiled_graph

app = FastAPI(title="ClinIQ Agent")


class QuestionRequest(BaseModel):
    question: str


@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = Path(__file__).parent / "static" / "index.html"
    return HTMLResponse(content=html_path.read_text())


@app.post("/ask")
async def ask(req: QuestionRequest):
    """Run the agent and stream back JSON results as SSE."""

    async def generate():
        yield f"data: {json.dumps({'status': 'running', 'message': 'Searching PubMed...'})}\n\n"
        await asyncio.sleep(0)

        try:
            import uuid
            thread_id = f"ui-{uuid.uuid4().hex[:8]}"
            async with compiled_graph() as graph:
                state: dict[str, Any] = await graph.ainvoke(
                    {"question": req.question},
                    config={"configurable": {"thread_id": thread_id}},
                )

            result = {
                "status": "done",
                "final_answer": state.get("final_answer", ""),
                "citations": state.get("citations", []),
                "retrieved_pmids": state.get("retrieved_pmids", []),
                "top_findings_pmids": [f["pmid"] for f in state.get("findings", [])],
                "iterations": state.get("iteration", 0),
                "enough_evidence": state.get("enough_evidence", False),
            }
            yield f"data: {json.dumps(result)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'status': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/score")
async def score_latest():
    """Re-score the latest eval run and return aggregates."""
    from evals.harness.scorer import score_run

    results_dir = Path(__file__).parent.parent / "evals" / "results"
    runs = sorted(results_dir.glob("run-*.jsonl"), key=lambda p: p.stat().st_mtime)
    if not runs:
        return {"error": "No eval runs found"}
    latest = runs[-1]
    scores = score_run(latest)
    return {"run": latest.name, **scores}
