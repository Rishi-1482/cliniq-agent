"""Reflector node — decides whether we have enough evidence to answer."""
from __future__ import annotations

import json

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from agent.state import AgentState


_REFLECTOR_SYSTEM_PROMPT = """You are evaluating whether we have enough evidence
to write a comprehensive, cited answer to a biomedical research question.

You will see:
  - The original question
  - Sub-questions that were searched
  - Findings retrieved (chunks of paper abstracts, with PMID and section)
  - Current iteration number

Your job: return JSON `{"enough_evidence": true|false, "missing": [...]}`.

**Be pragmatic about "enough".** A comprehensive answer does not require
covering every possible angle. You have "enough" when:
  - The findings substantively address the majority of what the question asks
  - You can point to specific chunks that support each key claim
  - Any remaining gaps are minor, not critical to the core question

Say "not enough" ONLY when there is a genuine, critical gap — e.g. no
methodology chunks at all when the question is about how something is done,
or no results chunks when the question asks what was found.

**Do not demand perfection.** If findings cover ~70% of what would ideally
be there and the missing 30% is peripheral, that is enough. Answers can
acknowledge limitations.

**Consider iteration count.** By iteration 2, if you've asked for the same
thing twice and the corpus still doesn't have it, it's not going to appear.
Mark enough_evidence=true and let the synthesizer note the limitation in
the final answer.

Return ONLY valid JSON (no code fences, no commentary):
{"enough_evidence": true|false, "missing": ["gap 1"]}

If enough_evidence is true, `missing` should be [].
If false, list AT MOST 2 specific gaps.
"""


async def reflector_node(state: AgentState) -> dict:
    """Decide: do we have enough evidence, or loop back for more?"""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    findings = state.get("findings", [])
    # Show the reflector a compact summary — full text of every chunk would
    # blow up the context window if the corpus is large.
    findings_summary = "\n".join(
        f"- [{f['pmid']} / {f['section']}] {f['text'][:200]}..."
        for f in findings[:15]  # cap what we show
    )

    user_prompt = (
        f"Original question: {state['question']}\n\n"
        f"Sub-questions:\n"
        + "\n".join(f"  - {sq}" for sq in state.get("sub_questions", []))
        + f"\n\nFindings ({len(findings)} chunks total, showing up to 15):\n"
        + findings_summary
        + f"\n\nCurrent iteration: {state.get('iteration', 1)}"
    )

    response = await llm.ainvoke([
        SystemMessage(content=_REFLECTOR_SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ])
    content = response.content.strip()
    if content.startswith("```"):
        content = content.strip("`").lstrip("json").strip()

    parsed = json.loads(content)
    enough = bool(parsed.get("enough_evidence", False))
    missing = parsed.get("missing", []) if not enough else []

    return {
        "enough_evidence": enough,
        "missing_topics": missing,
        "messages": [
            HumanMessage(content=(
                f"[Reflector] enough_evidence={enough} "
                f"missing={missing if missing else 'nothing'}"
            )),
        ],
    }