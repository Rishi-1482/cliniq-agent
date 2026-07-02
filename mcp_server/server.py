"""FastMCP server for ClinIQ. Hello-world version with one trivial tool."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mcp.server.fastmcp import FastMCP
from mcp_server.tools.pubmed import search_pubmed

from dotenv import load_dotenv

load_dotenv()

from mcp_server.tools.pubmed import (
    search_pubmed as _search_pubmed,
    fetch_abstract as _fetch_abstract,
)

from mcp_server.tools.corpus import Corpus, OpenAIEmbedding, chunk_abstract
from mcp_server.tools.pubmed import fetch_abstract as _fetch_abstract_internal

# Initialize the corpus once at server startup.
# The embedding backend is a module-level default so tests can override it.
_embedding = OpenAIEmbedding()
_corpus = Corpus(embedding=_embedding)

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


@mcp.tool()
async def index_papers(pmids: list[str]) -> dict:
    """Fetch abstracts for the given PMIDs and add them to the searchable corpus.

    After indexing, use query_corpus to ask questions grounded in these papers.
    Papers already in the index are skipped, so calling this repeatedly with
    overlapping PMID lists is safe and cheap.

    Args:
        pmids: List of PubMed IDs (as strings, e.g. ["31894144", "32234571"]).

    Returns:
        A summary dict with pmids_requested, papers_indexed, chunks_added,
        and any failures.
    """
    if not pmids:
        return {"pmids_requested": 0, "papers_indexed": 0, "chunks_added": 0, "failures": []}

    indexed_count = 0
    chunk_count = 0
    failures: list[dict] = []
    all_chunks = []

    for pmid in pmids:
        try:
            paper = await _fetch_abstract_internal(pmid)
            if paper is None:
                failures.append({"pmid": pmid, "reason": "no abstract available"})
                continue
            chunks = chunk_abstract(
                pmid=paper.pmid,
                title=paper.title,
                abstract=paper.abstract,
                journal=paper.journal,
                pub_date=paper.pub_date,
            )
            all_chunks.extend(chunks)
            indexed_count += 1
        except Exception as e:
            failures.append({"pmid": pmid, "reason": f"fetch error: {e}"})

    if all_chunks:
        chunk_count = await _corpus.add(all_chunks)

    return {
        "pmids_requested": len(pmids),
        "papers_indexed": indexed_count,
        "chunks_added": chunk_count,
        "corpus_total_chunks": _corpus.count(),
        "failures": failures,
    }


@mcp.tool()
async def query_corpus(question: str, k: int = 5) -> list[dict]:
    """Semantic search over papers you've already indexed.

    Returns the top-k most relevant chunks from the corpus, each with source
    metadata. Use this after index_papers to answer questions grounded in the
    actual paper content. Cite results by their PMID.

    Args:
        question: A natural-language question. Retrieval is semantic, not
            keyword-based, so phrasing matters less than for search_pubmed.
        k: Number of chunks to return (default 5, typical range 3-10).

    Returns:
        A list of chunk records, each with pmid, section, text, title,
        journal, pub_date, and similarity_score. Results are sorted by
        relevance (most relevant first).
    """
    results = await _corpus.query(question, k=k)
    return [
        {
            "pmid": r.chunk.pmid,
            "section": r.chunk.section,
            "text": r.chunk.text,
            "title": r.chunk.title,
            "journal": r.chunk.journal,
            "pub_date": r.chunk.pub_date,
            "similarity_score": round(r.score, 4),
        }
        for r in results
    ]

if __name__ == "__main__":
    # Run as a stdio MCP server
    mcp.run()