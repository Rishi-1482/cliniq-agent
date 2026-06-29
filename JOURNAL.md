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

