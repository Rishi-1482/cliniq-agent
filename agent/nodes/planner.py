"""Planner node — decomposes the user's question into targeted sub-questions."""
from __future__ import annotations

import json
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from agent.state import AgentState


_PLANNER_SYSTEM_PROMPT = """You are a research assistant planning a literature search.

Given a biomedical research question, decompose it into 2-4 focused
sub-questions that together will fully answer the original. Each sub-question
should be searchable in PubMed — concrete, specific, and framed around a
single concept.

Return ONLY valid JSON (no code fences, no commentary):
{"sub_questions": ["...", "...", "..."]}

Guidelines:
- Prefer specific over broad.
- Each sub-question should be independent — searchable on its own.
- If the question is already narrow, 2 sub-questions is fine. Don't pad.
- Include methodology sub-questions when relevant.

**When re-planning to fill gaps** (you'll see a "prior sub-questions" list):
- DO NOT repeat prior sub-questions verbatim.
- Try a different search strategy: broader terms, different vocabulary,
  related MeSH concepts, or attack the gap from a different angle.
- If a gap was searched twice already and nothing was found, it may not
  exist in the accessible literature — try adjacent topics instead.
"""


async def planner_node(state: AgentState) -> dict:
    """Decompose the user's question into searchable sub-questions."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    context = ""
    prior = state.get("sub_questions", [])
    if state.get("iteration", 0) > 0 and state.get("missing_topics"):
        prior_list = "\n".join(f"  - {sq}" for sq in prior)
        missing_list = "\n".join(f"  - {t}" for t in state["missing_topics"])
        context = (
            f"\n\nPRIOR SUB-QUESTIONS (already tried — do not repeat these):\n{prior_list}"
            f"\n\nREMAINING GAPS (target these with DIFFERENT phrasing/strategy):\n{missing_list}"
        )

    user_prompt = f"Question: {state['question']}{context}"

    response = await llm.ainvoke([
        SystemMessage(content=_PLANNER_SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ])
    content = response.content.strip()
    if content.startswith("```"):
        content = content.strip("`").lstrip("json").strip()

    parsed = json.loads(content)
    sub_questions = parsed["sub_questions"]

    return {
        "sub_questions": sub_questions,
        "iteration": state.get("iteration", 0) + 1,
        "messages": [
            SystemMessage(content=f"[Planner] iteration {state.get('iteration', 0) + 1}"),
            HumanMessage(content=f"Planned {len(sub_questions)} sub-questions: {sub_questions}"),
        ],
    }