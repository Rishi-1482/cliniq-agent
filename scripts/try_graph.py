"""Run the partial graph (planner + retriever only) end-to-end."""
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
    print()
    print("Sub-questions:")
    for sq in result.get("sub_questions", []):
        print(f"  - {sq}")
    print()
    print(f"Retrieved {len(result.get('retrieved_pmids', []))} unique PMIDs:")
    for pmid in result.get("retrieved_pmids", []):
        print(f"  - {pmid}")
    print()
    print(f"Iteration: {result.get('iteration')}")
    print(f"Thread ID (for follow-ups): {thread_id}")


if __name__ == "__main__":
    asyncio.run(main())