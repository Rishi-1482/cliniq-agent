"""PubMed E-utilities client.

Talks to NCBI's eutils API to search PubMed and retrieve paper metadata.
Designed to be framework-agnostic — callable from anywhere, not just MCP.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import httpx

import xml.etree.ElementTree as ET

EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
DEFAULT_TIMEOUT = 30.0
TOOL_NAME = "cliniq-agent"


@dataclass
class PaperSummary:
    """Lightweight metadata for a PubMed paper."""
    pmid: str
    title: str
    authors: list[str]
    journal: str
    pub_date: str
    doi: str | None = None

@dataclass
class PaperAbstract:
    """Full abstract content for a single paper."""
    pmid: str
    title: str
    abstract: str
    journal: str
    pub_date: str
    authors: list[str]


def _common_params() -> dict[str, str]:
    """Params required on every eutils request for identification/auth."""
    params = {
        "tool": TOOL_NAME,
        "email": os.environ.get("NCBI_EMAIL", ""),
    }
    api_key = os.environ.get("NCBI_API_KEY")
    if api_key:
        params["api_key"] = api_key
    return params


async def search_pubmed(
    query: str,
    max_results: int = 10,
    *,
    client: httpx.AsyncClient | None = None,
) -> list[PaperSummary]:
    """Search PubMed and return summary metadata for matching papers.
    
    Args:
        query: PubMed search query (supports standard PubMed syntax,
            e.g. "deep learning AND mammography").
        max_results: Maximum number of papers to return (1-100).
        client: Optional httpx client (for testing/reuse). If None,
            a new client is created and closed within this call.

    Returns:
        List of PaperSummary objects, possibly empty if no matches.
    """
    if not 1 <= max_results <= 100:
        raise ValueError("max_results must be between 1 and 100")

    own_client = client is None
    if own_client:
        client = httpx.AsyncClient(timeout=DEFAULT_TIMEOUT)

    try:
        # Step 1: ESearch -> get PMIDs
        search_params = {
            **_common_params(),
            "db": "pubmed",
            "term": query,
            "retmax": str(max_results),
            "retmode": "json",
            "sort": "relevance",
        }
        resp = await client.get(f"{EUTILS_BASE}/esearch.fcgi", params=search_params)
        resp.raise_for_status()
        search_data = resp.json()
        pmids = search_data.get("esearchresult", {}).get("idlist", [])

        if not pmids:
            return []

        # Step 2: ESummary -> get metadata
        summary_params = {
            **_common_params(),
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "json",
        }
        resp = await client.get(f"{EUTILS_BASE}/esummary.fcgi", params=summary_params)
        resp.raise_for_status()
        summary_data = resp.json()

        return _parse_summaries(pmids, summary_data)
    finally:
        if own_client:
            await client.aclose()


def _parse_summaries(pmids: list[str], summary_data: dict[str, Any]) -> list[PaperSummary]:
    """Extract PaperSummary objects from an esummary JSON response."""
    result = summary_data.get("result", {})
    papers: list[PaperSummary] = []

    for pmid in pmids:
        record = result.get(pmid)
        if not record or "error" in record:
            continue

        authors = [
            a.get("name", "") for a in record.get("authors", [])
            if a.get("authtype") == "Author"
        ]

        # Find DOI in articleids list
        doi = None
        for aid in record.get("articleids", []):
            if aid.get("idtype") == "doi":
                doi = aid.get("value")
                break

        papers.append(PaperSummary(
            pmid=pmid,
            title=record.get("title", "").rstrip("."),
            authors=authors,
            journal=record.get("fulljournalname") or record.get("source", ""),
            pub_date=record.get("pubdate", ""),
            doi=doi,
        ))

    return papers
    

async def fetch_abstract(
    pmid: str,
    *,
    client: httpx.AsyncClient | None = None,
) -> PaperAbstract | None:
    """Fetch the full abstract and metadata for a single PubMed paper.

    Args:
        pmid: PubMed ID as a string.
        client: Optional httpx client for reuse/testing.

    Returns:
        A PaperAbstract object, or None if the paper has no abstract
        available (some papers, like errata, don't have one).
    """
    if not pmid.isdigit():
        raise ValueError(f"pmid must be numeric, got: {pmid!r}")

    own_client = client is None
    if own_client:
        client = httpx.AsyncClient(timeout=DEFAULT_TIMEOUT)

    try:
        params = {
            **_common_params(),
            "db": "pubmed",
            "id": pmid,
            "rettype": "abstract",
            "retmode": "xml",
        }
        resp = await client.get(f"{EUTILS_BASE}/efetch.fcgi", params=params)
        resp.raise_for_status()
        return _parse_abstract_xml(pmid, resp.text)
    finally:
        if own_client:
            await client.aclose()


def _parse_abstract_xml(pmid: str, xml_text: str) -> PaperAbstract | None:
    """Extract abstract fields from an EFetch XML response.

    Returns None if the requested article isn't found or has no abstract.
    """
    root = ET.fromstring(xml_text)
    article = root.find(".//PubmedArticle")
    if article is None:
        return None

    # Title
    title_el = article.find(".//ArticleTitle")
    title = _text(title_el).rstrip(".")

    # Abstract — may have multiple <AbstractText> elements with labels
    # (Background, Methods, Results, Conclusion). We concatenate them.
    abstract_parts: list[str] = []
    for abs_text_el in article.findall(".//Abstract/AbstractText"):
        label = abs_text_el.get("Label")
        text = _text(abs_text_el)
        if not text:
            continue
        if label:
            abstract_parts.append(f"{label}: {text}")
        else:
            abstract_parts.append(text)

    if not abstract_parts:
        return None  # No abstract available

    abstract = "\n\n".join(abstract_parts)

    # Journal
    journal_el = article.find(".//Journal/Title")
    journal = _text(journal_el)

    # Publication date — pick year at minimum
    pub_year_el = article.find(".//Journal/JournalIssue/PubDate/Year")
    pub_month_el = article.find(".//Journal/JournalIssue/PubDate/Month")
    year = _text(pub_year_el)
    month = _text(pub_month_el)
    pub_date = f"{year} {month}".strip() if year else ""

    # Authors
    authors: list[str] = []
    for author_el in article.findall(".//AuthorList/Author"):
        last = _text(author_el.find("LastName"))
        initials = _text(author_el.find("Initials"))
        if last:
            name = f"{last} {initials}".strip() if initials else last
            authors.append(name)

    return PaperAbstract(
        pmid=pmid,
        title=title,
        abstract=abstract,
        journal=journal,
        pub_date=pub_date,
        authors=authors,
    )


def _text(el: ET.Element | None) -> str:
    """Safely extract text from an ElementTree element, or empty string if None."""
    if el is None:
        return ""
    # itertext() concatenates text from nested elements too — some abstracts
    # have <i>, <b>, etc. inside them
    return "".join(el.itertext()).strip()