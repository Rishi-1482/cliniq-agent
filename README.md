# ClinIQ Agent

A biomedical research agent that answers clinical questions by searching PubMed, indexing relevant papers into a vector store, and synthesizing a cited answer — all in a single pipeline.

![Architecture](https://img.shields.io/badge/stack-LangGraph%20%7C%20ChromaDB%20%7C%20FastAPI-blue)
![Python](https://img.shields.io/badge/python-3.12-blue)

---

## What it does

1. **Planner** — decomposes your question into 2–4 focused PubMed sub-queries
2. **Retriever** — searches PubMed for each sub-query, collects PMIDs
3. **Indexer** — fetches abstracts and chunks them into a ChromaDB vector store
4. **Reader** — semantic search over the corpus to pull the most relevant chunks
5. **Reflector** — decides if there's enough evidence; if not, loops back to re-plan with different queries (max 3 iterations)
6. **Synthesizer** — writes a 2–4 paragraph answer grounded in the findings, with inline `[PMID: ...]` citations

---

## Quickstart

### Prerequisites
- Python 3.12+
- [`uv`](https://docs.astral.sh/uv/) for package management
- An OpenAI API key

### Setup

```bash
git clone https://github.com/Rishi-1482/cliniq-agent.git
cd cliniq-agent

# Install dependencies
uv sync

# Add your OpenAI key
echo "OPENAI_API_KEY=sk-..." > .env
```

### Run the web UI

```bash
uv run uvicorn ui.app:app --reload --port 8000
```

Open **http://localhost:8000** in your browser.

---

## Project structure

```
cliniq-agent/
├── agent/
│   ├── graph.py          # LangGraph state graph — wires all nodes together
│   ├── state.py          # AgentState TypedDict
│   ├── mcp_client.py     # MCP client factory + result unpacker
│   └── nodes/
│       ├── planner.py
│       ├── retriever.py
│       ├── indexer.py
│       ├── reader.py
│       ├── reflector.py
│       └── synthesizer.py
├── mcp_server/
│   └── server.py         # FastMCP server: search_pubmed + index_papers + query_corpus
├── evals/
│   ├── harness/
│   │   ├── runner.py     # runs agent on a question set, writes JSONL
│   │   ├── scorer.py     # offline scoring from saved results
│   │   └── metrics.py    # recall@k, citation_faithfulness, citation_coverage
│   ├── questions/
│   │   └── seed.json     # 5 seed eval questions
│   └── results/          # JSONL run outputs
├── ui/
│   ├── app.py            # FastAPI server
│   └── static/
│       └── index.html    # Web frontend
├── scripts/              # One-off exploration scripts
└── tests/                # Unit tests (parser, metrics)
```

---

## Running evals

```bash
# Run the agent on all seed questions and save results
uv run python -c "
import asyncio
from evals.harness.runner import run_eval_set
asyncio.run(run_eval_set('evals/questions/seed.json'))
"

# Score the latest run
uv run python -c "
import json
from evals.harness.scorer import score_run
result = score_run(sorted(__import__('pathlib').Path('evals/results').glob('run-*.jsonl'))[-1])
print(json.dumps(result['aggregate'], indent=2))
"
```

Current results on 5 seed questions:

| Metric | Score |
|---|---|
| mean_recall@10 | 1.0 |
| mean_citation_faithfulness | 1.0 |
| mean_citation_coverage | 1.0 |
| mean_iterations | 1.4 |

---

## Tech stack

| Component | Library |
|---|---|
| Agent orchestration | [LangGraph](https://github.com/langchain-ai/langgraph) |
| LLM calls | `langchain-openai` (gpt-4o-mini) |
| Tool serving | [FastMCP](https://github.com/jlowin/fastmcp) |
| Tool consumption | `langchain-mcp-adapters` |
| Vector store | [ChromaDB](https://www.trychroma.com/) |
| Embeddings | OpenAI `text-embedding-3-small` |
| Literature search | PubMed E-utilities API |
| Web UI | [FastAPI](https://fastapi.tiangolo.com/) + vanilla JS |
| Package management | [uv](https://docs.astral.sh/uv/) |
