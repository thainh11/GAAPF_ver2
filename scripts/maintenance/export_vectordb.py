#!/usr/bin/env python3
"""
Script to export ChromaDB content to a readable format.
This allows examining what's stored in the vector database.
"""

import os
import sys
import json
import argparse
from pathlib import Path
import chromadb
import pandas as pd

# Add the parent directory to the path so we can import the GAAPF package
sys.path.append(str(Path(__file__).parent / "GAAPF-main"))

def export_collection_to_json(chroma_path, collection_name, output_path=None):
    """
    Export a ChromaDB collection to a JSON file
    
    Args:
        chroma_path: Path to ChromaDB directory
        collection_name: Name of the collection to export
        output_path: Path to save the JSON file (defaults to collection_name.json)
    
    Returns:
        Path to the exported JSON file
    """
    # Initialize ChromaDB client
    client = chromadb.PersistentClient(path=str(chroma_path))
    
    try:
        # Get the collection
        collection = client.get_collection(name=collection_name)
        
        # Get all items from the collection
        results = collection.get(include=["documents", "metadatas", "embeddings"])
        
        # Convert to a more readable format
        readable_results = []
        
        if results and "ids" in results and results["ids"]:
            for i, item_id in enumerate(results["ids"]):
                entry = {
                    "id": item_id,
                    "document": results["documents"][i] if "documents" in results else None,
                    "metadata": results["metadatas"][i] if "metadatas" in results else None,
                    # Embeddings are typically large vectors, so we'll just include their dimensions
                    "embedding_dimensions": len(results["embeddings"][i]) if "embeddings" in results and results["embeddings"] else None
                }
                readable_results.append(entry)
        
        # Determine output path
        if output_path is None:
            output_path = f"{collection_name}_export.json"
        
        # Save to JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(readable_results, f, indent=2, ensure_ascii=False)
        
        print(f"Successfully exported collection '{collection_name}' to {output_path}")
        
        # Also create a more human-readable CSV for documents and metadata
        if readable_results:
            # Extract documents and metadata for CSV
            csv_data = []
            for item in readable_results:
                doc = item["document"]
                meta = item["metadata"]
                
                # Flatten document and metadata
                entry = {"id": item["id"]}
                
                if isinstance(doc, list):
                    for i, d in enumerate(doc):
                        entry[f"document_{i}"] = d
                else:
                    entry["document"] = doc
                
                if meta:
                    for k, v in meta.items():
                        entry[f"meta_{k}"] = v
                
                csv_data.append(entry)
            
            # Create DataFrame and save to CSV
            csv_path = output_path.replace('.json', '.csv')
            df = pd.DataFrame(csv_data)
            df.to_csv(csv_path, index=False)
            print(f"Also exported to CSV: {csv_path}")
        
        return output_path
    
    except Exception as e:
        print(f"Error exporting collection: {e}")
        return None

def export_all_collections(chroma_path, output_dir=None):
    """
    Export all collections in a ChromaDB instance
    
    Args:
        chroma_path: Path to ChromaDB directory
        output_dir: Directory to save the JSON files (defaults to current directory)
    
    Returns:
        List of paths to the exported JSON files
    """
    # Initialize ChromaDB client
    client = chromadb.PersistentClient(path=str(chroma_path))
    
    # Get all collections
    collections = client.list_collections()
    
    if not collections:
        print("No collections found in the database.")
        return []
    
    # Create output directory if it doesn't exist
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Export each collection
    exported_files = []
    for collection in collections:
        collection_name = collection.name
        
        if output_dir:
            output_path = os.path.join(output_dir, f"{collection_name}_export.json")
        else:
            output_path = f"{collection_name}_export.json"
        
        result = export_collection_to_json(chroma_path, collection_name, output_path)
        if result:
            exported_files.append(result)
    
    return exported_files

def main():
    parser = argparse.ArgumentParser(description='Export ChromaDB collections to JSON')
    parser.add_argument('--chroma-path', type=str, required=True, help='Path to ChromaDB directory')
    parser.add_argument('--collection', type=str, help='Name of the collection to export (if not specified, exports all collections)')
    parser.add_argument('--output', type=str, help='Output path for the exported JSON file(s)')
    
    args = parser.parse_args()
    
    if args.collection:
        export_collection_to_json(args.chroma_path, args.collection, args.output)
    else:
        export_all_collections(args.chroma_path, args.output)

if __name__ == "__main__":
    main() 