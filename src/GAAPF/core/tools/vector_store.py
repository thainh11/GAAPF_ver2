"""
Vector Store wrapper for ChromaDB with Vertex AI embeddings.

This module provides a high-level wrapper around ChromaDB for storing and retrieving
document embeddings, optimized for use with Vertex AI embedding models.
"""

import os
import json
import logging
from typing import Dict, List, Optional, Union, Any
from pathlib import Path

import chromadb
from langchain_google_vertexai import VertexAIEmbeddings
try:
	from langchain_chroma import Chroma as LCChroma
except Exception:
	from langchain_community.vectorstores import Chroma as LCChroma
	import warnings as _warnings
	_warnings.warn(
		"Using deprecated langchain_community Chroma. Install 'langchain-chroma' to remove this warning.",
		DeprecationWarning,
	)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VectorStore:
    """
    A high-level wrapper for ChromaDB with Vertex AI embeddings.
    
    This class provides:
    1. Simplified API for document storage and retrieval
    2. Consistent embedding generation with Vertex AI
    3. Metadata schema enforcement
    4. Persistence management
    """
    
    def __init__(self, 
                 persistent_dir: Union[str, Path] = Path('data/framework_cache/chroma_db'),
                 collection_name: str = "framework_docs",
                 embedding_model: str = "gemini-embedding-001",
                 project: str = None,
                 location: str = "us-central1"):
        """
        Initialize the Vector Store with ChromaDB backend.
        
        Args:
            persistent_dir: Directory for ChromaDB persistence
            collection_name: Name of the collection to use
            embedding_model: Vertex AI embedding model name
            project: GCP project ID
            location: GCP location
        """
        # Set up credentials path for Vertex AI (prefer env, fallback to local file)
        if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
            fallback_creds = os.path.join(project_root, "google-credentials.json")
            if os.path.exists(fallback_creds):
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = fallback_creds
        
        # Ensure directory exists
        self.persistent_dir = Path(persistent_dir) if isinstance(persistent_dir, str) else persistent_dir
        self.persistent_dir.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize embedding function
        effective_embedding_model = os.getenv("VERTEX_EMBEDDING_MODEL", embedding_model)
        # Resolve project id from env or credentials json
        effective_project = project or os.getenv("GOOGLE_CLOUD_PROJECT")
        if not effective_project:
            creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
            try:
                if creds_path and os.path.exists(creds_path):
                    with open(creds_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        pj = data.get("project_id")
                        if pj:
                            effective_project = pj
                            os.environ["GOOGLE_CLOUD_PROJECT"] = pj
            except Exception:
                pass
        effective_project = effective_project or "gen-lang-client-0305686287"
        effective_location = os.getenv("GOOGLE_CLOUD_LOCATION", location or "us-central1")
        self.embedding_function = VertexAIEmbeddings(
            model_name=effective_embedding_model,
            project=effective_project,
            location=effective_location
        )
        
        # Disable Chroma telemetry noise
        os.environ.setdefault("ANONYMIZED_TELEMETRY", "false")

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=str(self.persistent_dir))
        
        # Get or create collection
        self.collection_name = collection_name
        self._get_or_create_collection()
        
        logger.info(f"VectorStore initialized with collection '{collection_name}' at {persistent_dir}")
    
    def _get_or_create_collection(self):
        """
        Get or create ChromaDB collection.
        """
        try:
            self.collection = self.client.get_collection(name=self.collection_name)
            logger.info(f"Using existing collection: {self.collection_name}")
        except Exception as e:
            logger.info(f"Creating new collection: {self.collection_name}")
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": f"Framework documentation for {self.collection_name}"}
            )
        
        # Create LangChain Chroma wrapper for semantic search
        self.langchain_vectorstore = LCChroma(
            client=self.client,
            collection_name=self.collection_name,
            embedding_function=self.embedding_function
        )
    
    def add_documents(self, documents: List[Dict[str, Any]], ids: Optional[List[str]] = None) -> List[str]:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of document dictionaries with 'text' and 'metadata' keys
            ids: Optional list of IDs for the documents
            
        Returns:
            List of document IDs
        """
        texts = [doc.get('text', doc.get('page_content', '')) for doc in documents]
        metadatas = [doc.get('metadata', {}) for doc in documents]
        
        # Generate IDs if not provided
        if ids is None:
            from uuid import uuid4
            ids = [str(uuid4()) for _ in range(len(documents))]
        
        # Add documents to the collection
        self.collection.add(
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
        
        logger.info(f"Added {len(documents)} documents to collection {self.collection_name}")
        return ids
    
    def similarity_search(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """
        Perform similarity search using the vector store.
        
        Args:
            query: Search query text
            k: Number of results to return
            
        Returns:
            List of document dictionaries with 'text', 'metadata', and 'score' keys
        """
        # Use LangChain's similarity_search for better API
        results = self.langchain_vectorstore.similarity_search_with_score(query, k=k)
        
        # Format results
        formatted_results = []
        for doc, score in results:
            formatted_results.append({
                'text': doc.page_content,
                'metadata': doc.metadata,
                'score': score
            })
        
        return formatted_results
    
    def delete(self, ids: List[str]) -> bool:
        """
        Delete documents from the vector store.
        
        Args:
            ids: List of document IDs to delete
            
        Returns:
            True if successful
        """
        try:
            self.collection.delete(ids=ids)
            logger.info(f"Deleted {len(ids)} documents from collection {self.collection_name}")
            return True
        except Exception as e:
            logger.error(f"Error deleting documents: {e}")
            return False
    
    def count(self) -> int:
        """
        Get the number of documents in the collection.
        
        Returns:
            Document count
        """
        return self.collection.count()
    
    def get_collection_info(self) -> Dict[str, Any]:
        """
        Get information about the collection.
        
        Returns:
            Collection information dictionary
        """
        return {
            "name": self.collection_name,
            "count": self.collection.count(),
            "metadata": self.collection.metadata
        }