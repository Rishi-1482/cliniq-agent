"""ClinIQ agent — LangGraph state graph.

Currently a work in progress. Nodes wired so far: planner, retriever.
"""
from __future__ import annotations

import functools
from contextlib import asynccontextmanager

from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.memory import MemorySaver

from agent.mcp_client import build_mcp_client
from agent.nodes.planner import planner_node
from agent.nodes.retriever import retriever_node
from agent.state import AgentState


def build_graph(mcp_client):
    """Construct the LangGraph. Requires an MCP client for tool-calling nodes."""
    builder = StateGraph(AgentState)

    builder.add_node("planner", planner_node)
    builder.add_node(
        "retriever",
        functools.partial(retriever_node, mcp_client=mcp_client),
    )

    builder.add_edge(START, "planner")
    builder.add_edge("planner", "retriever")
    builder.add_edge("retriever", END)  # temporary — will be replaced

    return builder.compile(checkpointer=MemorySaver())


@asynccontextmanager
async def compiled_graph():
    """Context manager: builds MCP client + graph, cleans up on exit."""
    mcp_client = build_mcp_client()
    graph = build_graph(mcp_client)
    try:
        yield graph
    finally:
        # MultiServerMCPClient manages its own subprocess lifecycle via
        # per-call sessions; no explicit teardown needed here.
        pass