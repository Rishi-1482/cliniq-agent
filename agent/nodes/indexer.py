"""Indexer node — fetches abstracts and adds them to the vector corpus."""
from __future__ import annotations

from langchain_core.messages import HumanMessage
from langchain_mcp_adapters.client import MultiServerMCPClient

from agent.mcp_client import unpack_mcp_result
from agent.state import AgentState


async def indexer_node(state: AgentState, mcp_client: MultiServerMCPClient) -> dict:
    """Index the retrieved papers into the corpus so we can query them."""
    pmids = state.get("retrieved_pmids", [])
    if not pmids:
        return {"indexed_pmids": [], "messages": [
            HumanMessage(content="[Indexer] No PMIDs to index; skipping."),
        ]}

    tools_by_name = {t.name: t for t in await mcp_client.get_tools()}
    index_tool = tools_by_name["index_papers"]

    raw = await index_tool.ainvoke({"pmids": pmids})
    result = unpack_mcp_result(raw)

    # index_papers returns a single dict summary, but MCP wraps it in a list
    if isinstance(result, list) and result:
        result = result[0]

    return {
        "indexed_pmids": pmids,  # for our purposes, all attempted PMIDs
        "messages": [
            HumanMessage(content=(
                f"[Indexer] papers_indexed={result.get('papers_indexed', 0)} "
                f"chunks_added={result.get('chunks_added', 0)} "
                f"failures={len(result.get('failures', []))} "
                f"corpus_total={result.get('corpus_total_chunks', '?')}"
            )),
        ],
    }