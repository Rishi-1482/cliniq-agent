# Build Journal

## 2026-06-27 — Project setup

- Set up uv-managed project, Python 3.11
- Installed: langgraph, langchain-openai, langchain-mcp-adapters, mcp, httpx, python-dotenv
- Dev deps: pytest, pytest-asyncio, ruff
- Created public GitHub repo

**Decisions made today:**
- uv over poetry/pip: speed, automatic lockfile, single source of truth in pyproject.toml
- (more as they come up)

**Open questions:**
- (note things you don't yet know how to decide)

2026-06-29 — Hello-world working
Got the MCP server + LangGraph agent talking end-to-end. Was confused why the agent said "hello" instead of "echo: hello" — turns out the tool returned the right thing, but the LLM summarizes tool output into a natural response by default. The raw tool output appears as a ToolMessage in the message history, and the LLM produces a separate AIMessage as its final answer. To get verbatim output, the prompt has to explicitly ask for it.
Decision: keep the noise around ListToolsRequest for now — it's stderr output from FastMCP and useful for debugging. Will silence in production.

**Debug: ModuleNotFoundError from anaconda-uv conflict**

Symptom: `uv run python -c "import mcp_server"` succeeded from the project root, but `uv run python scripts/try_pubmed.py` failed with ModuleNotFoundError. Same venv, same Python, different result.

Root cause: anaconda's `base` environment was auto-activating in every shell. `uv run` was launching the venv but conda's env vars were leaking into the subprocess, causing Python to load anaconda's stdlib (`/opt/anaconda3/lib/python3.12/...`) while looking in the venv's site-packages — a broken hybrid environment. The `.pth` files were also getting concatenated without newlines, which mangled `sys.path`.

Fix: `conda deactivate`, disabled auto-activation with `conda config --set auto_activate_base false`, deleted `.venv`, ran `uv sync` clean.

**Lesson:** uv and conda do not play well together. On any machine with conda installed, the first thing to verify in a uv project is `which python` does not point into `anaconda3/`. The "it works from -c but not from script" pattern was the key diagnostic — same env should give same result.

**Understanding tests (2026-07-14)**

Wrote three unit tests for `_parse_summaries`. The exercise of intentionally breaking the parser and watching the test catch it made the value obvious — 2 seconds of test run vs. 30 minutes of "why is my agent returning weird data" debugging.

**What I test:** pure logic functions where input → output has rules. Parsers, transformers, business logic.

**What I don't test (yet):** network calls, the MCP server, the LLM. Different tools for those (mocks, integration tests, evals).

**Design principle:** the parser is separated from the HTTP call in `search_pubmed` specifically so it's testable in isolation. If they were fused, testing would require mocking `httpx`, and every test would be slower and more fragile.

**Made scripts self-bootstrapping**

After the editable install regressed for the fourth time, I stopped trying to
make it stable and instead made every entry-point script add the project root
to sys.path explicitly. `conftest.py` handles pytest, `scripts/_bootstrap.py`
handles the manual scripts, and inline sys.path.insert handles agent/ and
mcp_server/.

Tradeoff: purists dislike sys.path manipulation. In practice, for a
research/agent project with entry-point scripts, it removes a whole category
of environment bugs. If this ever becomes an installable library, I'll rework
the entry points to use console_scripts — but for now, "clone and run" beats
"clone, sync, hope."

**Observation: answer drifts toward corpus composition (2026-07-07)**

First full end-to-end run produced a good cited answer, but the answer drifted
toward breast density assessment specifically — one narrow slice of the
broader question about "evaluating DL vs radiologist performance in mammography
screening." Root cause: the corpus was small (~7 papers) and skewed by which
papers PubMed's relevance ranking surfaced for the sub-questions. Semantic
retrieval then preferentially pulled from those density-focused papers.

Not a code bug — a behavioral characteristic of RAG systems when the corpus
is small/skewed. Two potential mitigations for later:
  1. Retrieve more papers per sub-question (retmax=10+) at the cost of
     latency and noise.
  2. Add a coverage check to Reader — if all top chunks come from ≤2 papers,
     re-query with different phrasing or broader top-k.

Filing this for Week 3's eval to quantify.

**Eval harness surfaced real bugs (2026-07-11)**

First eval run on 5 seed questions showed mean_citation_faithfulness = 0.483,
which looked alarming. Investigation revealed two separate issues:

1. **State-field data loss.** `retrieved_pmids` and `indexed_pmids` were empty
   in the final state despite the Retriever and Indexer nodes clearly running.
   Root cause: on iteration 2 the retriever returned new results via
   `{"retrieved_pmids": new_list}` which *replaced* iteration 1's results
   (LangGraph replaces fields without a reducer). When the second search
   returned 0 hits for a sub-question, the list became [].
   Fix: `retriever_node` now seeds `all_pmids` from `state.get("retrieved_pmids", [])`
   so results accumulate across iterations. Same fix applied to `indexer_node`
   for `indexed_pmids`.

2. **Wrong denominator for faithfulness.** Even with the state issue, faithfulness
   was comparing cited_pmids against retrieved_pmids. But the Synthesizer only
   sees findings (Reader output), not the retrieval set. Citations are faithful
   if they trace back to what the Synthesizer was shown, not what was searched
   for. Fixed by comparing against top_findings_pmids.

**Meta-lesson: metrics require the same rigor as production code.** My first
faithfulness metric looked reasonable in isolation but measured the wrong
thing. Without eval, I would have shipped an agent I thought had 100%
citation faithfulness. With eval, I discovered a metric bug AND a state
bug in one 5-question run.

Interview material: "The value of evaluation is not the numbers — it's the
questions you're forced to ask when the numbers surprise you."