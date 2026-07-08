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
    print()
    print("=" * 60)
    print("ANSWER")
    print("=" * 60)
    print(result.get("final_answer", "(no answer generated)"))
    print()
    print("=" * 60)
    print("CITATIONS")
    print("=" * 60)
    for c in result.get("citations", []):
        print(f"  [{c['pmid']}] {c['title']}")
    print(f"\nThread ID: {thread_id}")


if __name__ == "__main__":
    asyncio.run(main())