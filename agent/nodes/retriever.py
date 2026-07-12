"""Retriever node — searches PubMed for each sub-question."""
from __future__ import annotations

from langchain_core.messages import HumanMessage
from langchain_mcp_adapters.client import MultiServerMCPClient

from agent.state import AgentState

import json
from typing import Any


def _unpack_mcp_result(result: Any) -> Any:
    """Unpack a langchain-mcp-adapters tool result into a plain Python value.

    The adapter (v0.3.0) returns tool results as a list of TextContent-style
    dicts: [{"type": "text", "text": "<json>", ...}, ...]. Each `text` field
    is a JSON string of one item from the tool's return value. We parse each
    one and return the flat list (or single value if the tool returned one).
    """
    # Some tools return a single value; some return a list. Normalize.
    if isinstance(result, list):
        parsed = []
        for item in result:
            if isinstance(item, dict) and item.get("type") == "text":
                parsed.append(json.loads(item["text"]))
            else:
                parsed.append(item)
        return parsed
    if isinstance(result, dict) and result.get("type") == "text":
        return json.loads(result["text"])
    return result


async def retriever_node(state: AgentState, mcp_client: MultiServerMCPClient) -> dict:
    """For each sub-question, call search_pubmed and collect PMIDs."""
    sub_questions = state.get("sub_questions", [])
    if not sub_questions:
        return {"retrieved_pmids": []}

    # Get MCP tools once
    tools_by_name = {t.name: t for t in await mcp_client.get_tools()}
    search_tool = tools_by_name["search_pubmed"]

    # Start from PMIDs already collected in previous iterations
    all_pmids: list[str] = list(state.get("retrieved_pmids", []))
    per_query_counts: list[str] = []

    for sq in sub_questions:
        raw = await search_tool.ainvoke({"query": sq, "max_results": 5})
        results = _unpack_mcp_result(raw)
        pmids = [r["pmid"] for r in results]
        all_pmids.extend(pmids)
        per_query_counts.append(f"{sq!r}: {len(pmids)} results")

    # De-dupe while preserving order (a PMID may be relevant to multiple sub-questions)
    seen = set()
    unique_pmids = []
    for pmid in all_pmids:
        if pmid not in seen:
            seen.add(pmid)
            unique_pmids.append(pmid)

    return {
        "retrieved_pmids": unique_pmids,
        "messages": [
            HumanMessage(content=(
                f"[Retriever] Searched {len(sub_questions)} queries, "
                f"got {len(unique_pmids)} unique PMIDs. "
                + " | ".join(per_query_counts)
            )),
        ],
    }