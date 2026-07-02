"""Smoke test: index a few papers and query the corpus."""
import _bootstrap  # noqa: F401
import asyncio
from dotenv import load_dotenv

from mcp_server.tools.corpus import Corpus, OpenAIEmbedding, chunk_abstract
from mcp_server.tools.pubmed import fetch_abstract, search_pubmed

load_dotenv()


async def main():
    # 1. Find some papers
    print("Searching PubMed for deep learning + mammography papers...")
    papers = await search_pubmed("deep learning mammography screening", max_results=5)
    print(f"Found {len(papers)} papers.\n")

    # 2. Build a corpus
    corpus = Corpus(embedding=OpenAIEmbedding())

    # 3. Fetch abstracts and index them
    print("Fetching abstracts and indexing...")
    all_chunks = []
    for paper in papers:
        full = await fetch_abstract(paper.pmid)
        if full is None:
            print(f"  Skipped {paper.pmid} (no abstract)")
            continue
        chunks = chunk_abstract(
            pmid=full.pmid,
            title=full.title,
            abstract=full.abstract,
            journal=full.journal,
            pub_date=full.pub_date,
        )
        all_chunks.extend(chunks)
        print(f"  Indexed {full.pmid}: {len(chunks)} chunks — {full.title[:60]}...")

    added = await corpus.add(all_chunks)
    print(f"\nAdded {added} new chunks. Corpus now has {corpus.count()} total.\n")

    # 4. Query it
    question = "What methodology is used to evaluate deep learning models on mammograms?"
    print(f"Query: {question}\n")
    results = await corpus.query(question, k=3)

    for i, r in enumerate(results, 1):
        print(f"[{i}] PMID {r.chunk.pmid} — {r.chunk.section} "
              f"(score={r.score:.3f})")
        print(f"    {r.chunk.title[:80]}")
        print(f"    {r.chunk.text[:200]}...\n")


if __name__ == "__main__":
    asyncio.run(main())