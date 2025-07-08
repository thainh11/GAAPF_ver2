#!/usr/bin/env python3
"""
Script to print the content of ChromaDB collections directly to the console.
"""

import os
import sys
from pathlib import Path
import chromadb

# Add the parent directory to the path so we can import the GAAPF package
sys.path.append(str(Path(__file__).parent / "GAAPF-main"))

# Now, we can import the necessary modules
from src.GAAPF.core.memory.long_term_memory import LongTermMemory

def print_collection(chroma_path, collection_name):
    """
    Print a ChromaDB collection to the console
    
    Args:
        chroma_path: Path to ChromaDB directory
        collection_name: Name of the collection to print
    """
    # Initialize ChromaDB client
    print(f"Initializing ChromaDB client at {chroma_path}...")
    client = chromadb.PersistentClient(path=str(chroma_path))
    
    try:
        # Get the collection
        print(f"Getting collection: {collection_name}")
        collection = client.get_collection(name=collection_name)
        
        # Get all items from the collection
        print("Retrieving items from collection...")
        results = collection.get(include=["documents", "metadatas", "embeddings"])
        
        # Print the results
        if results and "ids" in results and results["ids"]:
            print(f"\nFound {len(results['ids'])} items in collection {collection_name}:")
            print("-" * 80)
            
            for i, item_id in enumerate(results["ids"]):
                print(f"\nItem {i+1}:")
                print(f"  ID: {item_id}")
                
                if "documents" in results and i < len(results["documents"]):
                    doc = results["documents"][i]
                    print(f"  Document: {doc[:500]}..." if len(str(doc)) > 500 else f"  Document: {doc}")
                
                if "metadatas" in results and i < len(results["metadatas"]):
                    meta = results["metadatas"][i]
                    print(f"  Metadata: {meta}")
                
                if "embeddings" in results and i < len(results["embeddings"]):
                    emb = results["embeddings"][i]
                    print(f"  Embedding dimensions: {len(emb)}")
                    print(f"  First 5 embedding values: {emb[:5]}...")
                
                print("-" * 80)
        else:
            print(f"No items found in collection {collection_name}")
    
    except Exception as e:
        print(f"Error printing collection: {e}")

def print_all_collections(chroma_path):
    """
    Print all collections in a ChromaDB instance
    
    Args:
        chroma_path: Path to ChromaDB directory
    """
    # Initialize ChromaDB client
    print(f"Initializing ChromaDB client at {chroma_path}...")
    client = chromadb.PersistentClient(path=str(chroma_path))
    
    # Get all collections
    print("Listing collections...")
    collections = client.list_collections()
    
    if not collections:
        print("No collections found in the database.")
        return
    
    print(f"Found {len(collections)} collections:")
    for i, collection in enumerate(collections):
        print(f"{i+1}. {collection.name}")
    
    # Print each collection
    for collection in collections:
        print("\n" + "=" * 80)
        print(f"COLLECTION: {collection.name}")
        print("=" * 80)
        print_collection(chroma_path, collection.name)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python print_vectordb.py <chroma_path> [collection_name]")
        sys.exit(1)
    
    chroma_path = sys.argv[1]
    
    if len(sys.argv) > 2:
        collection_name = sys.argv[2]
        print_collection(chroma_path, collection_name)
    else:
        print_all_collections(chroma_path) 