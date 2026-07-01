"""FastMCP server for ClinIQ. Hello-world version with one trivial tool."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mcp.server.fastmcp import FastMCP
from mcp_server.tools.pubmed import search_pubmed

from mcp_server.tools.pubmed import (
    search_pubmed as _search_pubmed,
    fetch_abstract as _fetch_abstract,
)

mcp = FastMCP("cliniq")


@mcp.tool()
def echo(text: str) -> str:
    """Echo back the provided text. Used for testing the server plumbing."""
    return f"echo: {text}"


@mcp.tool()
async def search_pubmed(query: str, max_results: int = 10) -> list[dict]:
    """Search PubMed for biomedical research papers.

    Returns lightweight metadata (title, authors, journal) — no abstract.
    Use fetch_abstract afterward to get the full text for a specific paper.

    Args:
        query: A PubMed search query. Supports standard PubMed syntax —
            field tags like [Title], boolean operators (AND, OR, NOT),
            and MeSH terms. Examples:
              - "deep learning mammography"
              - "BRCA1[Gene] AND breast neoplasms[MeSH]"
              - "convolutional neural network AND radiology[Journal]"
        max_results: Number of papers to return (default 10, max 100).

    Returns:
        A list of paper records, each with pmid, title, authors,
        journal, pub_date, and doi (when available).
    """
    papers = await _search_pubmed(query, max_results=max_results)
    return [
        {
            "pmid": p.pmid,
            "title": p.title,
            "authors": p.authors,
            "journal": p.journal,
            "pub_date": p.pub_date,
            "doi": p.doi,
        }
        for p in papers
    ]


@mcp.tool()
async def fetch_abstract(pmid: str) -> dict | None:
    """Fetch the full abstract for a specific PubMed paper.

    Use this after search_pubmed to get the actual abstract text you can
    read and reason about. Structured abstracts (with sections like
    BACKGROUND, METHODS, RESULTS) are preserved with their section labels.

    Args:
        pmid: The PubMed ID of the paper, as a numeric string
            (e.g. "31894144").

    Returns:
        A dict with pmid, title, abstract, journal, pub_date, and authors,
        or None if the paper has no abstract available (some editorials,
        errata, and case reports don't).
    """
    paper = await _fetch_abstract(pmid)
    if paper is None:
        return None
    return {
        "pmid": paper.pmid,
        "title": paper.title,
        "abstract": paper.abstract,
        "journal": paper.journal,
        "pub_date": paper.pub_date,
        "authors": paper.authors,
    }


if __name__ == "__main__":
    # Run as a stdio MCP server
    mcp.run()