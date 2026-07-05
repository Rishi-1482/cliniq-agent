"""State schema for the ClinIQ agent graph.

Every node reads from this state and returns updates that get merged in.
Fields fall into three groups:
  1. Input   — what the user asked
  2. Working — intermediate results the graph builds up
  3. Output  — what the graph produces
"""
from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class Finding(TypedDict):
    """A single piece of evidence retrieved from the corpus."""
    pmid: str
    section: str
    text: str
    title: str
    similarity_score: float


class Citation(TypedDict):
    """A source cited in the final answer."""
    pmid: str
    title: str
    journal: str
    pub_date: str


class AgentState(TypedDict, total=False):
    # ---- Input ----
    question: str  # the original user question

    # ---- Working memory ----
    sub_questions: list[str]     # planner's decomposition of the question
    retrieved_pmids: list[str]   # PMIDs from search across all sub-questions
    indexed_pmids: list[str]     # subset actually added to the corpus
    findings: list[Finding]      # top chunks retrieved from corpus
    iteration: int               # loop counter, capped at 3
    enough_evidence: bool        # reflector's decision
    missing_topics: list[str]    # what the reflector says is still needed

    # ---- Output ----
    final_answer: str
    citations: list[Citation]

    # ---- Observability ----
    messages: Annotated[list[BaseMessage], add_messages]