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
        from ..utils.credentials_helper import setup_google_credentials, get_vertex_embedding_config
        
        # Setup Google credentials
        setup_google_credentials()
        
        # Ensure directory exists
        self.persistent_dir = Path(persistent_dir) if isinstance(persistent_dir, str) else persistent_dir
        self.persistent_dir.parent.mkdir(parents=True, exist_ok=True)
        
        # Get embedding configuration
        embedding_config = get_vertex_embedding_config()
        self.embedding_function = VertexAIEmbeddings(**embedding_config)
        
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
            # Check if collection exists
            existing_collection = self.client.get_collection(name=self.collection_name)
            
            # Always check dimension compatibility, even for empty collections
            # Test current embedding function dimension
            test_vector = self.embedding_function.embed_query("test")
            current_dim = len(test_vector)
            
            # Check metadata first for dimension info
            metadata = existing_collection.metadata or {}
            stored_dim = metadata.get('embedding_dimension')
            
            # If metadata has correct dimension, use the collection
            if stored_dim == current_dim:
                logger.info(f"Collection has correct dimension in metadata: {stored_dim}")
            else:
                # Check if collection has any embeddings to verify dimension
                result = existing_collection.peek(limit=1)
                if result['embeddings'] is not None and len(result['embeddings']) > 0:
                    existing_dim = len(result['embeddings'][0])
                    if existing_dim != current_dim:
                        logger.info(f"Dimension mismatch: existing={existing_dim}, current={current_dim}. Recreating collection.")
                        self.client.delete_collection(name=self.collection_name)
                        raise Exception("Dimension mismatch - recreating collection")
                else:
                    # For empty collections without correct metadata, recreate
                    if stored_dim and stored_dim != current_dim:
                        logger.info(f"Dimension mismatch in metadata: stored={stored_dim}, current={current_dim}. Recreating collection.")
                        self.client.delete_collection(name=self.collection_name)
                        raise Exception("Dimension mismatch - recreating collection")
            
            self.collection = existing_collection
            logger.info(f"Using existing collection: {self.collection_name}")
        except Exception as e:
            logger.info(f"Creating new collection: {self.collection_name}")
            # Test current embedding function dimension for metadata
            test_vector = self.embedding_function.embed_query("test")
            current_dim = len(test_vector)
            
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={
                    "description": f"Framework documentation for {self.collection_name}",
                    "embedding_dimension": current_dim,
                    "embedding_model": getattr(self.embedding_function, 'model_name', 'unknown')
                }
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
        try:
            # Use direct ChromaDB query to avoid LangChain dimension issues
            query_embedding = self.embedding_function.embed_query(query)
            
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                include=['documents', 'metadatas', 'distances']
            )
            
            # Format results
            formatted_results = []
            if results['documents'] and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    metadata = results['metadatas'][0][i] if results['metadatas'] and results['metadatas'][0] else {}
                    distance = results['distances'][0][i] if results['distances'] and results['distances'][0] else 0.0
                    
                    formatted_results.append({
                        'text': doc,
                        'metadata': metadata,
                        'score': 1.0 - distance  # Convert distance to similarity score
                    })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error in similarity search: {e}")
            return []
    
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