
import asyncio
import json
from pathlib import Path
import os
import sys
import shutil

# Add project root to Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

from src.GAAPF.core.tools.framework_collector import FrameworkCollector
try:
    import importlib
    import src.GAAPF.core.tools.framework_collector as fc_mod
    print(f"Using FrameworkCollector module: {getattr(fc_mod, '__file__', 'unknown')}" )
except Exception:
    pass
from src.GAAPF.core.memory.long_term_memory import LongTermMemory
from src.GAAPF.core.utils.credentials_helper import setup_google_credentials, get_vertex_ai_config


async def main():
    """
    Initializes the framework knowledge base and then runs a test generation using Vertex AI.
    """
    # Setup Google credentials
    setup_google_credentials()
    
    # Get Vertex AI configuration
    vertex_config = get_vertex_ai_config()
    project = vertex_config['project']
    location = vertex_config['location']
    collection_name = "framework_knowledge"

    # Define the list of frameworks to be initialized
    framework_names = ["langchain", "langgraph", "crewai", "autogen", "haystack"]

    if not framework_names:
        print("No framework JSON files found in the 'frameworks' directory.")
        return

    print(f"Found frameworks to initialize: {', '.join(framework_names)}")

    # Initialize memory and collector
    # New organized vectordb base path
    db_path = Path(__file__).parent.parent.parent / "data" / "frameworks" / "vectordb"
    
    # Do not wipe existing DB; we want to preserve previous ingests.

    # The memory class will create the directory using Vertex AI embeddings
    memory = LongTermMemory(
        chroma_path=str(db_path),
        collection_name=collection_name,
        project=project,
        location=location
    )
    collector = FrameworkCollector(memory=memory, is_logging=True)
    user_id = "system_bootstrap"

    for framework_name in framework_names:
        print(f"--- Initializing knowledge for: {framework_name} ---")
        try:
            # Step 1: Ingest official docs into per-framework VectorStore collection
            if hasattr(collector, "ensure_ingested"):
                stats = collector.ensure_ingested(
                    framework_name=framework_name,
                    project=project,
                    location=location,
                    persistent_dir=str(db_path),
                )
                if stats.get("exists"):
                    print(f"Vector collection already exists for {framework_name}: {stats}")
                else:
                    print(f"Ingested docs for {framework_name}: {stats}")
            else:
                print("ensure_ingested not found on FrameworkCollector; skipping ingestion and proceeding with collection...")

            # Step 2: Collect info and store to memory/cache as before
            # Skip if raw cache already exists
            raw_cache_file = collector.raw_cache_dir / f"{framework_name.lower().replace(' ', '_')}.json"
            if raw_cache_file.exists():
                print(f"Raw cache already exists for {framework_name}: {raw_cache_file}")
                framework_info = None
            else:
                framework_info = await collector.collect_framework_info(
                    framework_name=framework_name,
                    user_id=user_id,
                    max_pages=8,
                    force_refresh=False
                )
                if framework_info:
                    # Save to new structured raw cache directory
                    cache_file = collector.raw_cache_dir / f"{framework_name.lower().replace(' ', '_')}.json"
                    with open(cache_file, "w", encoding="utf-8") as f:
                        json.dump(framework_info, f, indent=2)
                    print(f"Cached raw info at: {cache_file}")
                else:
                    print(f"Warning: could not collect info for {framework_name}")

        except Exception as e:
            print(f"An error occurred while processing {framework_name}: {e}")
        print("-" * (30 + len(framework_name)))
    
    print("\n\n--- Framework Initialization Complete ---")
    print("--- Starting Test Curriculum Generation using Vertex AI ---")

    print("\n=== Framework VectorStore Setup Complete ===")
    print(f"Successfully initialized VectorStore for {len(framework_names)} frameworks")
    print("VectorStore is ready for RAG operations")


if __name__ == "__main__":
    asyncio.run(main())