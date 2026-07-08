"""ClinIQ agent — LangGraph state graph."""
from __future__ import annotations

import functools
from contextlib import asynccontextmanager

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from agent.mcp_client import build_mcp_client
from agent.nodes.indexer import indexer_node
from agent.nodes.planner import planner_node
from agent.nodes.reader import reader_node
from agent.nodes.reflector import reflector_node
from agent.nodes.retriever import retriever_node
from agent.state import AgentState
from agent.nodes.synthesizer import synthesizer_node




MAX_ITERATIONS = 3


def _route_from_reflector(state: AgentState) -> str:
    """Conditional edge: loop back to planner, or move on to synthesis."""
    if state.get("enough_evidence"):
        return "done"
    if state.get("iteration", 0) >= MAX_ITERATIONS:
        return "done"
    return "replan"


def build_graph(mcp_client):
    builder = StateGraph(AgentState)
    builder.add_node("synthesizer", synthesizer_node)
    builder.add_node("planner", planner_node)
    builder.add_node(
        "retriever",
        functools.partial(retriever_node, mcp_client=mcp_client),
    )
    builder.add_node(
        "indexer",
        functools.partial(indexer_node, mcp_client=mcp_client),
    )
    builder.add_node(
        "reader",
        functools.partial(reader_node, mcp_client=mcp_client),
    )
    builder.add_node("reflector", reflector_node)

    builder.add_edge(START, "planner")
    builder.add_edge("planner", "retriever")
    builder.add_edge("retriever", "indexer")
    builder.add_edge("indexer", "reader")
    builder.add_edge("reader", "reflector")

    # Conditional edge — the cycle
    builder.add_conditional_edges(
        "reflector",
        _route_from_reflector,
        {
            "replan": "planner",  # loop back
            "done": "synthesizer",           # will become "synthesizer" in the next chunk
        },
    )
    builder.add_edge("synthesizer", END)

    return builder.compile(checkpointer=MemorySaver())


@asynccontextmanager
async def compiled_graph():
    mcp_client = build_mcp_client()
    graph = build_graph(mcp_client)
    try:
        yield graph
    finally:
        pass