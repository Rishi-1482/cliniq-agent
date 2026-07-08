"""Reader node — queries the indexed corpus for each sub-question."""
from __future__ import annotations

from langchain_core.messages import HumanMessage
from langchain_mcp_adapters.client import MultiServerMCPClient

from agent.mcp_client import unpack_mcp_result
from agent.state import AgentState


async def reader_node(state: AgentState, mcp_client: MultiServerMCPClient) -> dict:
    """For each sub-question, retrieve the most relevant chunks from the corpus."""
    sub_questions = state.get("sub_questions", [])
    if not sub_questions:
        return {"findings": []}

    tools_by_name = {t.name: t for t in await mcp_client.get_tools()}
    query_tool = tools_by_name["query_corpus"]

    all_findings: list[dict] = []
    per_query_counts: list[str] = []

    for sq in sub_questions:
        raw = await query_tool.ainvoke({"question": sq, "k": 4})
        chunks = unpack_mcp_result(raw)
        for c in chunks:
            all_findings.append({
                "pmid": c["pmid"],
                "section": c["section"],
                "text": c["text"],
                "title": c["title"],
                "similarity_score": c["similarity_score"],
            })
        per_query_counts.append(f"{sq[:40]!r}: {len(chunks)} chunks")

    # De-duplicate findings by chunk identity (pmid + section)
    seen = set()
    unique_findings = []
    for f in all_findings:
        key = (f["pmid"], f["section"])
        if key not in seen:
            seen.add(key)
            unique_findings.append(f)

    return {
        "findings": unique_findings,
        "messages": [
            HumanMessage(content=(
                f"[Reader] Retrieved {len(unique_findings)} unique chunks. "
                + " | ".join(per_query_counts)
            )),
        ],
    }