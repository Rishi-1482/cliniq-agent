"""Run the LangGraph agent through planner → retriever → indexer → reader → reflector."""
import _bootstrap  # noqa: F401
import asyncio
import uuid
from dotenv import load_dotenv

from agent.graph import compiled_graph

load_dotenv()


async def main():
    question = (
        "How are researchers evaluating deep learning models against radiologist "
        "performance in mammography screening?"
    )
    thread_id = f"trygraph-{uuid.uuid4().hex[:8]}"

    async with compiled_graph() as graph:
        result = await graph.ainvoke(
            {"question": question},
            config={"configurable": {"thread_id": thread_id}},
        )

    print("=" * 60)
    print("Question:", result["question"])
    print(f"\nIterations: {result.get('iteration')}")
    print(f"Enough evidence: {result.get('enough_evidence')}")
    print(f"Missing topics: {result.get('missing_topics', [])}")
    print(f"\nSub-questions ({len(result.get('sub_questions', []))}):")
    for sq in result.get("sub_questions", []):
        print(f"  - {sq}")

    findings = result.get("findings", [])
    print(f"\nFindings ({len(findings)} unique chunks):")
    for f in findings[:5]:
        print(f"  [{f['pmid']} / {f['section']}] score={f['similarity_score']:.3f}")
        print(f"    {f['text'][:150]}...")
    if len(findings) > 5:
        print(f"  ... and {len(findings) - 5} more")

    print(f"\nThread ID: {thread_id}")


if __name__ == "__main__":
    asyncio.run(main())