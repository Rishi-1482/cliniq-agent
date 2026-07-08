"""Synthesizer node — writes the final cited answer from collected findings."""
from __future__ import annotations

import json

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from agent.state import AgentState


_SYNTHESIZER_SYSTEM_PROMPT = """You are a clinical research assistant writing a
final answer to a biomedical research question, grounded in the paper findings
you've been shown.

You will receive:
  - The original question
  - A list of findings (chunks of paper abstracts, each tagged with a PMID)

Your job:
  1. Write a clear, well-structured answer (2-4 paragraphs) that directly
     addresses the question.
  2. Ground every substantive claim in the findings. Cite by PMID in square
     brackets, like [PMID: 12345678]. Multiple sources per claim are welcome:
     [PMID: 12345678, PMID: 87654321].
  3. If the findings are limited or leave clear gaps, acknowledge those
     limitations at the end — don't fabricate to fill the gap.

Rules:
  - Only cite PMIDs that appear in the findings. Do NOT invent citations.
  - Do not restate the findings verbatim. Synthesize across them.
  - Prefer specifics over generalities: numbers, methods, and specific
     claims are more useful than vague summaries.

After the prose answer, output a JSON block with the structured citation list.
Format your response EXACTLY as:

<answer>
[Your prose answer here, with [PMID: ...] citations inline.]
</answer>

<citations>
{"citations": [{"pmid": "...", "title": "..."}, ...]}
</citations>

Include only PMIDs you actually cited in the answer.
"""


async def synthesizer_node(state: AgentState) -> dict:
    """Generate the final cited answer from findings."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

    findings = state.get("findings", [])
    if not findings:
        return {
            "final_answer": (
                "I wasn't able to retrieve enough evidence from the literature "
                "to answer this question. This may indicate the topic is not "
                "well-covered by PubMed's indexed abstracts, or the query "
                "needs refinement."
            ),
            "citations": [],
        }

    # Compact but complete: PMID + title + full chunk text
    findings_block = "\n\n".join(
        f"[PMID: {f['pmid']}] ({f['section']}) — {f['title']}\n{f['text']}"
        for f in findings
    )

    user_prompt = (
        f"Question: {state['question']}\n\n"
        f"Findings ({len(findings)} chunks):\n\n{findings_block}"
    )

    response = await llm.ainvoke([
        SystemMessage(content=_SYNTHESIZER_SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ])

    content = response.content

    # Extract the answer and citation blocks
    answer = _extract_between(content, "<answer>", "</answer>").strip()
    citations_raw = _extract_between(content, "<citations>", "</citations>").strip()

    citations = []
    if citations_raw:
        try:
            citations = json.loads(citations_raw).get("citations", [])
        except json.JSONDecodeError:
            # Fall back: extract PMIDs from the answer text itself
            import re
            cited_pmids = set(re.findall(r"PMID:\s*(\d+)", answer))
            findings_by_pmid = {f["pmid"]: f for f in findings}
            citations = [
                {"pmid": pmid, "title": findings_by_pmid[pmid]["title"]}
                for pmid in cited_pmids
                if pmid in findings_by_pmid
            ]

    return {
        "final_answer": answer or content,  # fall back to full content if parsing failed
        "citations": citations,
        "messages": [
            HumanMessage(content=(
                f"[Synthesizer] Produced answer with {len(citations)} citations."
            )),
        ],
    }


def _extract_between(text: str, start: str, end: str) -> str:
    """Extract the substring between the first `start` and next `end`. Empty if not found."""
    start_idx = text.find(start)
    if start_idx == -1:
        return ""
    start_idx += len(start)
    end_idx = text.find(end, start_idx)
    if end_idx == -1:
        return text[start_idx:]
    return text[start_idx:end_idx]