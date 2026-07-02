"""Vector index and semantic search over indexed papers.

Uses ChromaDB as the vector store. Embedding model is injectable so we can
compare OpenAI embeddings against domain-specific alternatives (PubMedBERT)
later without rewriting the storage layer.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import chromadb
from chromadb.config import Settings
from openai import AsyncOpenAI


# ------------------------------------------------------------------
# Embedding backends — injectable, so we can swap models later
# ------------------------------------------------------------------

class EmbeddingBackend(Protocol):
    """Anything that can embed a batch of strings into vectors."""
    name: str
    async def embed(self, texts: list[str]) -> list[list[float]]: ...


class OpenAIEmbedding:
    """OpenAI text-embedding-3-small. 1536-dim, cheap, fast."""
    name = "openai-text-embedding-3-small"

    def __init__(self, model: str = "text-embedding-3-small"):
        self._client = AsyncOpenAI()
        self._model = model

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        resp = await self._client.embeddings.create(
            model=self._model,
            input=texts,
        )
        return [d.embedding for d in resp.data]


# ------------------------------------------------------------------
# Chunking
# ------------------------------------------------------------------

@dataclass
class Chunk:
    """A single indexable piece of a paper."""
    chunk_id: str        # unique across all chunks
    pmid: str
    section: str         # "BACKGROUND", "METHODS", ..., or "FULL" for unstructured
    text: str
    title: str           # paper title, denormalized for retrieval convenience
    journal: str
    pub_date: str


def chunk_abstract(
    pmid: str,
    title: str,
    abstract: str,
    journal: str,
    pub_date: str,
) -> list[Chunk]:
    """Split an abstract into indexable chunks.

    Strategy: if the abstract has labeled sections (BACKGROUND, METHODS, etc.),
    each section becomes its own chunk — better precision for questions like
    "what methodology did they use". Otherwise, the whole abstract is one chunk.
    """
    chunks: list[Chunk] = []

    if _has_labeled_sections(abstract):
        for i, (label, body) in enumerate(_split_labeled_sections(abstract)):
            chunks.append(Chunk(
                chunk_id=f"{pmid}::{label}",
                pmid=pmid,
                section=label,
                text=f"{label}: {body}",  # keep the label in the embedding
                title=title,
                journal=journal,
                pub_date=pub_date,
            ))
    else:
        chunks.append(Chunk(
            chunk_id=f"{pmid}::FULL",
            pmid=pmid,
            section="FULL",
            text=abstract,
            title=title,
            journal=journal,
            pub_date=pub_date,
        ))

    return chunks


def _has_labeled_sections(abstract: str) -> bool:
    """Heuristic: our fetch code emits 'LABEL: text\\n\\nLABEL: text'."""
    lines = [line.strip() for line in abstract.split("\n") if line.strip()]
    labeled_lines = sum(1 for line in lines if _looks_like_section_label(line))
    return labeled_lines >= 2


def _looks_like_section_label(line: str) -> bool:
    """A line 'BACKGROUND: something' has an all-caps label before the colon."""
    if ":" not in line:
        return False
    label = line.split(":", 1)[0].strip()
    return bool(label) and label.isupper() and len(label) <= 30


def _split_labeled_sections(abstract: str) -> list[tuple[str, str]]:
    """Return [(LABEL, body), ...] from an abstract that has labeled sections."""
    sections: list[tuple[str, str]] = []
    current_label: str | None = None
    current_body: list[str] = []

    for line in abstract.split("\n"):
        stripped = line.strip()
        if _looks_like_section_label(stripped):
            if current_label is not None:
                sections.append((current_label, "\n".join(current_body).strip()))
            label, body = stripped.split(":", 1)
            current_label = label.strip()
            current_body = [body.strip()] if body.strip() else []
        elif current_label is not None:
            current_body.append(line)

    if current_label is not None:
        sections.append((current_label, "\n".join(current_body).strip()))

    return sections


# ------------------------------------------------------------------
# The corpus itself
# ------------------------------------------------------------------

@dataclass
class RetrievedChunk:
    """A chunk returned by a query, with its similarity score."""
    chunk: Chunk
    score: float  # cosine distance from query; lower = more similar


class Corpus:
    """A persistent vector index of paper chunks."""

    def __init__(
        self,
        embedding: EmbeddingBackend,
        persist_dir: str | Path = "./chroma_db",
    ):
        self._embedding = embedding
        self._client = chromadb.PersistentClient(
            path=str(persist_dir),
            settings=Settings(anonymized_telemetry=False),
        )
        # One collection per embedding backend — vectors of different models
        # aren't comparable, so we isolate them.
        self._collection = self._client.get_or_create_collection(
            name=f"papers-{embedding.name}",
            metadata={"hnsw:space": "cosine"},
        )

    async def add(self, chunks: list[Chunk]) -> int:
        """Add chunks to the index. Returns number of new chunks added
        (already-indexed chunk_ids are skipped)."""
        if not chunks:
            return 0

        # Skip anything already in the store
        existing = set(self._collection.get(ids=[c.chunk_id for c in chunks])["ids"])
        new_chunks = [c for c in chunks if c.chunk_id not in existing]
        if not new_chunks:
            return 0

        texts = [c.text for c in new_chunks]
        vectors = await self._embedding.embed(texts)

        self._collection.add(
            ids=[c.chunk_id for c in new_chunks],
            embeddings=vectors,
            documents=texts,
            metadatas=[
                {
                    "pmid": c.pmid,
                    "section": c.section,
                    "title": c.title,
                    "journal": c.journal,
                    "pub_date": c.pub_date,
                }
                for c in new_chunks
            ],
        )
        return len(new_chunks)

    async def query(self, question: str, k: int = 5) -> list[RetrievedChunk]:
        """Return the top-k most semantically similar chunks."""
        [query_vec] = await self._embedding.embed([question])
        results = self._collection.query(
            query_embeddings=[query_vec],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )

        retrieved: list[RetrievedChunk] = []
        for chunk_id, doc, meta, dist in zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            retrieved.append(RetrievedChunk(
                chunk=Chunk(
                    chunk_id=chunk_id,
                    pmid=meta["pmid"],
                    section=meta["section"],
                    text=doc,
                    title=meta["title"],
                    journal=meta["journal"],
                    pub_date=meta["pub_date"],
                ),
                score=float(dist),
            ))
        return retrieved

    def count(self) -> int:
        return self._collection.count()