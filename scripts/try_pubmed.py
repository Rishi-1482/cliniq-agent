"""Quick manual smoke test for the PubMed client. Not part of the MCP server."""
import asyncio
from dotenv import load_dotenv

from mcp_server.tools.pubmed import search_pubmed

load_dotenv()


async def main():
    papers = await search_pubmed(
        "deep learning mammography screening",
        max_results=5,
    )
    print(f"Found {len(papers)} papers:\n")
    for p in papers:
        first_author = p.authors[0] if p.authors else "?"
        et_al = " et al." if len(p.authors) > 1 else ""
        print(f"  [{p.pmid}] {p.title}")
        print(f"    {first_author}{et_al} — {p.journal} ({p.pub_date})")
        if p.doi:
            print(f"    DOI: {p.doi}")
        print()


if __name__ == "__main__":
    asyncio.run(main())