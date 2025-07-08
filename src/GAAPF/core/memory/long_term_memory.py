from pathlib import Path
from typing import Optional, Union, List, Dict, Any
import json
import logging
import os
import chromadb
from langchain_google_vertexai import VertexAIEmbeddings
from chromadb import EmbeddingFunction
from .memory import Memory
import uuid

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VertexAIEmbeddingFunction(EmbeddingFunction):
    """A wrapper for LangChain's VertexAIEmbeddings to make it compatible with ChromaDB."""
    def __init__(self, embedding_func):
        self._embedding_func = embedding_func

    def __call__(self, input: List[str]) -> List[List[float]]:
        return self._embedding_func.embed_documents(input)

class LongTermMemory(Memory):
    """
    Long-term memory implementation using ChromaDB and Vertex AI embeddings.
    """
    def __init__(self, 
                memory_path: Optional[Union[Path, str]] = Path('templates/memory.json'),
                chroma_path: Optional[Union[Path, str]] = Path('memory/chroma_db'),
                collection_name: str = "long_term_memory",
                embedding_model: str = "textembedding-gecko@latest",
                is_reset_memory: bool = False,
                is_logging: bool = False,
                project: str = None,
                location: str = None,
                *args, **kwargs):
        """
        Initialize the long-term memory module.
        """
        super().__init__(
            memory_path=memory_path,
            is_reset_memory=is_reset_memory,
            is_logging=is_logging,
            *args, **kwargs
        )
        
        self.chroma_path = Path(chroma_path) if isinstance(chroma_path, str) else chroma_path
        self.chroma_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.client_db = chromadb.PersistentClient(path=str(self.chroma_path))
        
        # Use the LangChain VertexAIEmbeddings wrapped for ChromaDB
        langchain_vertex_embeddings = VertexAIEmbeddings(
            model_name=embedding_model,
            project=project,
            location=location
        )
        self.embedding_function = VertexAIEmbeddingFunction(langchain_vertex_embeddings)

        self._collection = self.client_db.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function
        )
    
    @property
    def collection(self):
        """Returns the ChromaDB collection."""
        return self._collection
    
    def save_short_term_memory(self, llm, message, user_id, agent_type=None):
        """
        Save short-term memory and also update the vector database.
        
        Args:
            llm: Language model for graph transformation
            message: Message to save
            user_id: User ID to associate with the memory
            agent_type: Type of agent creating this memory entry
        """
        # First use the standard memory saving process from the parent class
        graph = super().save_short_term_memory(llm, message, user_id, agent_type)
        
        # Then add to vector database for long-term recall
        if graph:
            self._add_to_vector_db(graph, user_id)
        
        return graph
    
    def _add_to_vector_db(self, graph: list, user_id: str):
        """
        Add memory graph entries to the vector database.
        
        Args:
            graph: List of memory graph entries
            user_id: User ID to associate with the memory
        """
        if not graph:
            return
        
        documents = []
        metadatas = []
        ids = []
        
        for i, entry in enumerate(graph):
            head = entry.get('head', '')
            relation = entry.get('relation', '')
            relation_properties = entry.get('relation_properties', '')
            tail = entry.get('tail', '')
            
            # Create a text representation of this memory entry
            if relation_properties:
                text = f"{head} -> {relation}[{relation_properties}] -> {tail}"
            else:
                text = f"{head} -> {relation} -> {tail}"
            
            documents.append(text)
            metadatas.append({
                "user_id": user_id,
                "head": head,
                "relation": relation,
                "tail": tail,
                "entry_type": "memory"
            })
            ids.append(f"{user_id}_{uuid.uuid4().hex}") # Use UUID for unique ID
        
        try:
            self.collection.upsert(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            if self.is_logging:
                logger.info(f"Added {len(documents)} entries to vector database for user {user_id}")
        except Exception as e:
            logger.error(f"Error adding to vector database: {e}")
    
    def query_similar_memories(self, query: str, user_id: str = None, n_results: int = 5, framework_id: str = None):
        """
        Query the vector database for memories similar to the query.
        
        Args:
            query: Query text
            user_id: Optional user ID to filter results
            n_results: Number of results to return
            framework_id: Optional framework ID to filter results
            
        Returns:
            List of similar memories
        """
        try:
            # Prepare filter if user_id is provided
            where_filter = {}
            if user_id:
                where_filter["user_id"] = user_id
            if framework_id:
                where_filter["framework"] = framework_id
            
            if not where_filter:
                where_filter = None

            # Query the vector database
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                where=where_filter
            )
            
            if self.is_logging:
                logger.info(f"Found {len(results['documents'][0])} similar memories for query: {query}")
            
            # Format results
            formatted_results = []
            for i, doc in enumerate(results['documents'][0]):
                metadata = results['metadatas'][0][i]
                distance = results['distances'][0][i]
                formatted_results.append({
                    "text": doc,
                    "metadata": metadata,
                    "distance": distance
                })
            
            return formatted_results
        except Exception as e:
            logger.error(f"Error querying vector database: {e}")
            return []
    
    def get_relevant_context(self, query: str, user_id: str = None, n_results: int = 5, framework_id: str = None):
        """
        Get relevant context from long-term memory for a given query.
        
        Args:
            query: Query text
            user_id: Optional user ID to filter results
            n_results: Number of results to return
            framework_id: Optional framework ID to filter results
            
        Returns:
            String containing relevant memories
        """
        memories = self.query_similar_memories(query, user_id, n_results, framework_id=framework_id)
        
        if not memories:
            return "No relevant memories found."
        
        context_lines = ["Relevant memories:"]
        for memory in memories:
            context_lines.append(f"- {memory['text']}")
        
        return "\n".join(context_lines)
    
    def delete_memories(self, user_id: str = None, framework_id: str = None):
        """
        Delete memories from the vector database.
        
        Args:
            user_id: Optional user ID to filter which memories to delete
            framework_id: Optional framework ID to filter which memories to delete
        """
        try:
            where_filter = {}
            if user_id:
                where_filter["user_id"] = user_id
            if framework_id:
                where_filter["framework"] = framework_id

            if where_filter:
                self.collection.delete(where=where_filter)
                if self.is_logging:
                    logger.info(f"Deleted memories with filter {where_filter} from vector database")
            else:
                # Delete all memories by recreating the collection
                logger.warning("No filter provided. Deleting all memories from collection.")
                self.client_db.delete_collection(name=self.collection_name)
                self._collection = self.client_db.create_collection(
                    name=self.collection_name,
                    embedding_function=self.embedding_function,
                    metadata={"hnsw:space": "cosine"}
                )
                if self.is_logging:
                    logger.info("Recreated collection to delete all memories.")
        except Exception as e:
            logger.error(f"Error deleting memories: {e}")
    
    def add_external_knowledge(self, text: str, user_id: str, source: str = "external", metadata: dict = None):
        """
        Add external knowledge to the vector database.
        
        Args:
            text: Text to add
            user_id: User ID to associate with the knowledge
            source: Source of the knowledge
            metadata: Additional metadata to store
        """
        if not text:
            return
        
        if metadata is None:
            metadata = {}
        
        metadata.update({
            "user_id": user_id,
            "source": source,
            "entry_type": "knowledge"
        })
        
        try:
            self.collection.upsert(
                documents=[text],
                metadatas=[metadata],
                ids=[f"{user_id}_knowledge_{source}_{uuid.uuid4().hex}"] # Use UUID for unique ID
            )
            if self.is_logging:
                logger.info(f"Added external knowledge for user {user_id} from source {source}")
        except Exception as e:
            logger.error(f"Error adding external knowledge: {e}") 