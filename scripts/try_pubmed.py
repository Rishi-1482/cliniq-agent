"""Quick manual smoke test for the PubMed client."""
import asyncio
import sys
from pathlib import Path
from dotenv import load_dotenv

import _bootstrap

# Guard: this script must run from the project root
_PROJECT_ROOT = Path(__file__).parent.parent
if Path.cwd() != _PROJECT_ROOT:
    print(f"⚠️  Run this from the project root: {_PROJECT_ROOT}", file=sys.stderr)
    print(f"   Currently in: {Path.cwd()}", file=sys.stderr)
    sys.exit(1)

from mcp_server.tools.pubmed import search_pubmed

load_dotenv()


async def main():
    papers = await search_pubmed(
        "Stephen Adamo",
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