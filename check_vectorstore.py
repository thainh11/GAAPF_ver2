#!/usr/bin/env python3
"""
Script to check VectorStore data for each framework
"""

import sys
import os
from pathlib import Path

# Add src to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from GAAPF.core.tools.vector_store import VectorStore

def check_framework_vectorstore(framework_name: str):
    """Check if VectorStore for a framework has data"""
    framework_id = framework_name.lower().replace(" ", "_")
    collection_name = f"framework_docs_{framework_id}"
    vs_base_path = Path("data/frameworks/vectordb") / framework_id / framework_id
    
    print(f"\n=== Checking {framework_name} ===")
    print(f"Base directory: {vs_base_path}")
    print(f"Collection: {collection_name}")
    
    if not vs_base_path.exists():
        print("❌ Directory does not exist")
        return
    
    try:
        vs = VectorStore(
            persistent_dir=str(vs_base_path),
            collection_name=collection_name,
            project=None,
            location="us-central1"
        )
        
        count = vs.count()
        print(f"📊 Document count: {count}")
        
        if count > 0:
            print("✅ VectorStore has data")
            # Try a sample search
            try:
                results = vs.similarity_search(f"What is {framework_name}?", k=1)
                if results:
                    print(f"🔍 Sample search result: {results[0]['text'][:100]}...")
            except Exception as e:
                print(f"⚠️ Search failed: {e}")
        else:
            print("❌ VectorStore is empty")
            
    except Exception as e:
        print(f"❌ Error accessing VectorStore: {e}")

def main():
    frameworks = ["langchain", "langgraph", "crewai", "autogen", "haystack"]
    
    print("🔍 Checking VectorStore data for all frameworks...")
    
    for framework in frameworks:
        check_framework_vectorstore(framework)
    
    print("\n✅ VectorStore check complete")

if __name__ == "__main__":
    main()