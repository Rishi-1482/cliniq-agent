"""Planner node — decomposes the user's question into targeted sub-questions."""
from __future__ import annotations

import json
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from agent.state import AgentState


_PLANNER_SYSTEM_PROMPT = """You are a research assistant planning a literature search.

Given a biomedical research question, decompose it into 2-4 focused sub-questions
that together will fully answer the original. Each sub-question should be
searchable in PubMed — concrete, specific, and framed around a single concept.

Return ONLY valid JSON in this exact shape (no code fences, no commentary):
{"sub_questions": ["...", "...", "..."]}

Guidelines:
- Prefer specific over broad. "What CNN architectures are used for mammography?"
  beats "Tell me about AI in radiology."
- Each sub-question should be independent — searchable on its own.
- If the question is already narrow, 2 sub-questions is fine. Don't pad.
- Include methodology sub-questions when relevant (e.g. "How is X evaluated?")
"""


async def planner_node(state: AgentState) -> dict:
    """Decompose the user's question into searchable sub-questions."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # If we're on a re-plan iteration, the reflector told us what's missing.
    # Feed that back in so the new plan targets gaps, not the whole question again.
    context = ""
    if state.get("iteration", 0) > 0 and state.get("missing_topics"):
        context = (
            f"\n\nPrior findings covered the question partially. "
            f"Focus this new plan on gaps that remain:\n"
            + "\n".join(f"- {t}" for t in state["missing_topics"])
        )

    user_prompt = f"Question: {state['question']}{context}"

    response = await llm.ainvoke([
        SystemMessage(content=_PLANNER_SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ])
    content = response.content.strip()

    # Strip code fences if the model added them despite our instructions
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