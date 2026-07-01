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