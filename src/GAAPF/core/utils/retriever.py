"""Retriever Utility for GAAPF

Provides a simple interface to perform semantic search over the framework
vector store and return the most relevant documents.
"""

from typing import List, Dict, Any, Optional
import logging

from ..tools.vector_store import VectorStore

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Retriever:
    """Semantic search utility built on top of VectorStore."""

    def __init__(self, vector_store: Optional[VectorStore] = None):
        # Lazily initialize VectorStore if not provided
        self.vector_store = vector_store or VectorStore()
        logger.info("Retriever initialized.")

    def retrieve_docs(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """Retrieve top-k relevant documents for the query."""
        if not query:
            return []
        return self.vector_store.similarity_search(query, k=k)