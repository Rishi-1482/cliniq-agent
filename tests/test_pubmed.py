"""Tests for the PubMed client.

Uses recorded response fixtures to avoid hitting the live API.
The point of these tests isn't to verify NCBI works — it's to verify
our parsing is correct and stays correct as we refactor.
"""
import pytest

from mcp_server.tools.pubmed import _parse_summaries, PaperSummary
from mcp_server.tools.pubmed import _parse_abstract_xml


# A trimmed-down but realistic esummary response shape
SAMPLE_ESUMMARY = {
    "result": {
        "uids": ["12345678"],
        "12345678": {
            "uid": "12345678",
            "title": "Deep learning for mammography screening: a review.",
            "authors": [
                {"name": "Smith J", "authtype": "Author"},
                {"name": "Doe A", "authtype": "Author"},
                {"name": "SomeGroup", "authtype": "CollectiveName"},
            ],
            "fulljournalname": "Journal of Medical Imaging",
            "source": "J Med Imaging",
            "pubdate": "2024 Mar",
            "articleids": [
                {"idtype": "pubmed", "value": "12345678"},
                {"idtype": "doi", "value": "10.1000/example"},
            ],
        },
    }
}


def test_parse_summaries_extracts_expected_fields():
    papers = _parse_summaries(["12345678"], SAMPLE_ESUMMARY)
    assert len(papers) == 1

    p = papers[0]
    assert p.pmid == "12345678"
    assert p.title == "Deep learning for mammography screening: a review"  # period stripped
    assert p.authors == ["Smith J", "Doe A"]  # collective name filtered out
    assert p.journal == "Journal of Medical Imaging"
    assert p.pub_date == "2024 Mar"
    assert p.doi == "10.1000/example"


def test_parse_summaries_skips_missing_or_errored_records():
    response = {
        "result": {
            "12345678": {"error": "cannot get document summary"},
        }
    }
    papers = _parse_summaries(["12345678", "99999999"], response)
    assert papers == []


def test_parse_summaries_handles_missing_doi():
    response = {
        "result": {
            "12345678": {
                "uid": "12345678",
                "title": "A paper",
                "authors": [],
                "fulljournalname": "J",
                "pubdate": "2024",
                "articleids": [{"idtype": "pubmed", "value": "12345678"}],
            }
        }
    }
    papers = _parse_summaries(["12345678"], response)
    assert len(papers) == 1
    assert papers[0].doi is None

SAMPLE_ABSTRACT_XML = """<?xml version="1.0"?>
<PubmedArticleSet>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>12345678</PMID>
      <Article>
        <Journal>
          <Title>Journal of Testing</Title>
          <JournalIssue>
            <PubDate>
              <Year>2024</Year>
              <Month>Mar</Month>
            </PubDate>
          </JournalIssue>
        </Journal>
        <ArticleTitle>A study of test parsing.</ArticleTitle>
        <Abstract>
          <AbstractText Label="BACKGROUND">The problem is X.</AbstractText>
          <AbstractText Label="METHODS">We did Y.</AbstractText>
          <AbstractText Label="RESULTS">Z happened.</AbstractText>
        </Abstract>
        <AuthorList>
          <Author>
            <LastName>Smith</LastName>
            <Initials>JA</Initials>
          </Author>
          <Author>
            <LastName>Doe</LastName>
            <Initials>B</Initials>
          </Author>
        </AuthorList>
      </Article>
    </MedlineCitation>
  </PubmedArticle>
</PubmedArticleSet>
"""


def test_parse_abstract_xml_extracts_structured_abstract():
    paper = _parse_abstract_xml("12345678", SAMPLE_ABSTRACT_XML)
    assert paper is not None
    assert paper.pmid == "12345678"
    assert paper.title == "A study of test parsing"  # period stripped
    assert paper.journal == "Journal of Testing"
    assert paper.pub_date == "2024 Mar"
    assert paper.authors == ["Smith JA", "Doe B"]
    assert "BACKGROUND: The problem is X." in paper.abstract
    assert "METHODS: We did Y." in paper.abstract
    assert "RESULTS: Z happened." in paper.abstract


def test_parse_abstract_xml_returns_none_when_no_abstract():
    xml_no_abstract = """<?xml version="1.0"?>
    <PubmedArticleSet>
      <PubmedArticle>
        <MedlineCitation>
          <Article>
            <ArticleTitle>A paper with no abstract.</ArticleTitle>
          </Article>
        </MedlineCitation>
      </PubmedArticle>
    </PubmedArticleSet>
    """
    paper = _parse_abstract_xml("99999999", xml_no_abstract)
    assert paper is None


def test_parse_abstract_xml_returns_none_when_article_missing():
    xml_empty = """<?xml version="1.0"?><PubmedArticleSet></PubmedArticleSet>"""
    paper = _parse_abstract_xml("99999999", xml_empty)
    assert paper is None