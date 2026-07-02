from mcp_server.tools.corpus import chunk_abstract, _has_labeled_sections, _split_labeled_sections


def test_chunk_abstract_splits_by_section_when_labeled():
    abstract = (
        "BACKGROUND: The disease is X.\n\n"
        "METHODS: We ran a trial.\n\n"
        "RESULTS: It worked.\n\n"
        "CONCLUSIONS: Ship it."
    )
    chunks = chunk_abstract(
        pmid="123", title="A paper", abstract=abstract,
        journal="J", pub_date="2024",
    )
    assert len(chunks) == 4
    sections = [c.section for c in chunks]
    assert sections == ["BACKGROUND", "METHODS", "RESULTS", "CONCLUSIONS"]
    assert chunks[1].text.startswith("METHODS: We ran a trial")
    # Chunk IDs are deterministic
    assert chunks[0].chunk_id == "123::BACKGROUND"


def test_chunk_abstract_returns_single_chunk_when_unstructured():
    abstract = "This is an unstructured abstract with no section labels."
    chunks = chunk_abstract(
        pmid="456", title="A paper", abstract=abstract,
        journal="J", pub_date="2024",
    )
    assert len(chunks) == 1
    assert chunks[0].section == "FULL"
    assert chunks[0].chunk_id == "456::FULL"


def test_has_labeled_sections_detects_structure():
    assert _has_labeled_sections(
        "BACKGROUND: X\n\nMETHODS: Y\n\nRESULTS: Z"
    ) is True
    assert _has_labeled_sections("Just a paragraph of text.") is False
    # One label alone isn't enough — could be an incidental colon
    assert _has_labeled_sections("BACKGROUND: only one section") is False