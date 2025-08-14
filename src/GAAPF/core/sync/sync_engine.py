"""Content Synchronization Engine for GAAPF

This engine detects new documentation or blog updates, processes the content,
embeds it with Vertex AI, and upserts vectors into ChromaDB so that
Retrieval-Augmented Generation (RAG) agents can leverage the latest material.

Phase-3 MVP implements a synchronous `run_once()` workflow;
scheduling / incremental sync will come later.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

import chromadb
from langchain.docstore.document import Document
from langchain_community.document_loaders import GitHubRepositoryLoader, SitemapLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_vertexai import VertexAIEmbeddings

from ..tools.vector_store import VectorStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ContentSyncEngine:
    """Engine to synchronise external content into the vector store."""

    DEFAULT_GITHUB_REPOS: List[str] = [
        # Example repo list – replace / extend via constructor
        "langchain-ai/langchain",
        "google/gemini-api-python",
    ]

    DEFAULT_SITEMAPS: List[str] = [
        "https://python.langchain.com/sitemap.xml",
    ]

    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        github_repos: Optional[List[str]] = None,
        sitemaps: Optional[List[str]] = None,
        chunk_size: int = 1024,
        chunk_overlap: int = 128,
    ):
        self.vector_store = vector_store or VectorStore()
        self.github_repos = github_repos or self.DEFAULT_GITHUB_REPOS
        self.sitemaps = sitemaps or self.DEFAULT_SITEMAPS
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run_once(self) -> Dict[str, Any]:
        """End-to-end sync workflow (detect → load → process → embed → upsert)."""
        logger.info("Starting content sync run…")
        new_docs: List[Document] = []

        # 1. Detect + Load changes
        new_docs.extend(self._load_github())
        new_docs.extend(self._load_sitemaps())
        logger.info(f"Loaded {len(new_docs)} raw documents.")

        # 2. Process & License filter
        processed_docs = self._process_content(new_docs)
        logger.info(f"After processing, {len(processed_docs)} chunks ready for embedding.")

        # 3. Generate embeddings & Upsert
        ids = self._upsert_to_vectorstore(processed_docs)

        # 4. Log
        self._write_sync_log(len(ids))
        logger.info("Content sync completed.")
        return {"chunks_upserted": len(ids)}

    # ------------------------------------------------------------------
    # Step-wise helpers (can be overridden / unit-tested)
    # ------------------------------------------------------------------
    def _load_github(self) -> List[Document]:
        docs: List[Document] = []
        for repo in self.github_repos:
            try:
                owner, name = repo.split("/")
                loader = GitHubRepositoryLoader(
                    repository=repo,
                    branch="main",
                    file_filter=lambda p: p.endswith(".md") or p.endswith(".rst"),
                )
                docs.extend(loader.load())
                logger.info(f"Loaded {len(docs)} docs from {repo}")
            except Exception as e:
                logger.warning(f"Failed to load repo {repo}: {e}")
        return docs

    def _load_sitemaps(self) -> List[Document]:
        docs: List[Document] = []
        for sitemap_url in self.sitemaps:
            try:
                loader = SitemapLoader(web_path=sitemap_url)
                docs.extend(loader.load())
                logger.info(f"Loaded {len(docs)} docs from sitemap {sitemap_url}")
            except Exception as e:
                logger.warning(f"Failed to load sitemap {sitemap_url}: {e}")
        return docs

    def _process_content(self, docs: List[Document]) -> List[Dict[str, Any]]:
        processed: List[Dict[str, Any]] = []
        for doc in docs:
            # License filter – simplistic (TODO: SPDX parsing)
            if not self._license_filter(doc):
                continue
            # Chunking
            chunks = self.text_splitter.split_text(doc.page_content)
            processed.extend([
                {"text": chunk, "metadata": doc.metadata} for chunk in chunks
            ])
        return processed

    def _license_filter(self, doc: Document) -> bool:
        """Return True if content license is acceptable."""
        # Placeholder – accept all for MVP
        return True

    def _upsert_to_vectorstore(self, docs: List[Dict[str, Any]]) -> List[str]:
        ids = self.vector_store.add_documents(docs)
        return ids

    def _write_sync_log(self, num_new_docs: int):
        log_dir = Path("monitoring_data")
        log_dir.mkdir(exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        log_path = log_dir / f"sync_{timestamp}.json"
        payload = {
            "timestamp": timestamp,
            "num_new_docs": num_new_docs,
            "collection": self.vector_store.collection_name,
        }
        try:
            with open(log_path, "w", encoding="utf-8") as f:
                import json
                json.dump(payload, f, indent=2)
            logger.info(f"Sync log written to {log_path}")
        except Exception as e:
            logger.error(f"Failed to write sync log: {e}")