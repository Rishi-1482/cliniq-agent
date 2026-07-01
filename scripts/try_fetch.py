"""Quick smoke test for fetch_abstract."""
import asyncio
import sys
from pathlib import Path
from dotenv import load_dotenv

import _bootstrap

_PROJECT_ROOT = Path(__file__).parent.parent
if Path.cwd() != _PROJECT_ROOT:
    print(f"⚠️  Run this from the project root: {_PROJECT_ROOT}", file=sys.stderr)
    sys.exit(1)

from mcp_server.tools.pubmed import fetch_abstract

load_dotenv()


async def main():
    # Pick a real PMID — a well-known deep learning + mammography paper.
    # This is McKinney et al. 2020 in Nature, on AI for breast cancer screening.
    pmid = "31894144"
    paper = await fetch_abstract(pmid)
    if paper is None:
        print(f"No abstract for PMID {pmid}")
        return

    print(f"[{paper.pmid}] {paper.title}\n")
    print(f"  {', '.join(paper.authors[:3])}"
          f"{' et al.' if len(paper.authors) > 3 else ''}"
          f" — {paper.journal} ({paper.pub_date})\n")
    print("Abstract:")
    print(paper.abstract)


if __name__ == "__main__":
    asyncio.run(main())