
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
from src.GAAPF.core.memory.long_term_memory import LongTermMemory
from src.GAAPF.core.curriculum.generator import CurriculumGenerator
import chromadb


async def main():
    """
    Initializes the framework knowledge base and then runs a test generation using Vertex AI.
    """
    # Define your Google Cloud project details here
    project = "gen-lang-client-0305686287"  # <-- IMPORTANT: Replace with your Google Cloud Project ID
    location = "us-central1"     # <-- IMPORTANT: Replace with your project's region if different
    collection_name = "framework_knowledge"

    # Define the list of frameworks to be initialized
    framework_names = ["langchain", "langgraph", "crewai", "autogen", "haystack"]

    if not framework_names:
        print("No framework JSON files found in the 'frameworks' directory.")
        return

    print(f"Found frameworks to initialize: {', '.join(framework_names)}")

    # Initialize memory and collector
    # Using a consistent path for the vectordb
    db_path = Path(__file__).parent.parent.parent / "data" / "framework_cache" / "chroma_db"
    
    # Clean the database directory before starting
    if db_path.exists():
        if db_path.is_dir():
            shutil.rmtree(db_path)
            print(f"Cleaned old database directory at {db_path}")
        else:
            db_path.unlink()
            print(f"Cleaned old database file at {db_path}")

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
            # Step 1: Collect comprehensive information about the framework.
            # This will use web searches and store the results in a cache file.
            framework_info = await collector.collect_framework_info(
                framework_name=framework_name,
                user_id=user_id,
                max_pages=10,  # Using a moderate number of pages for initial setup
                force_refresh=True # We want to ensure fresh data during setup
            )

            # Step 2: The collector already stores the info in memory via _store_in_memory
            # which is called by collect_framework_info. So no extra step is needed here.
            
            if framework_info:
                print(f"Successfully collected and stored information for {framework_name}.")
                # Save the collected info for potential direct use or inspection
                cache_file = collector.cache_dir / f"{framework_name.lower().replace(' ', '_')}.json"
                with open(cache_file, "w") as f:
                    json.dump(framework_info, f, indent=2)
                print(f"Cached raw info at: {cache_file}")

            else:
                print(f"Could not collect information for {framework_name}.")

        except Exception as e:
            print(f"An error occurred while processing {framework_name}: {e}")
        print("-" * (30 + len(framework_name)))
    
    print("\n\n--- Framework Initialization Complete ---")
    print("--- Starting Test Curriculum Generation using Vertex AI ---")

    try:
        # Use the same client that LongTermMemory created
        db_client = memory.client_db 
        generator = CurriculumGenerator(
            db_client=db_client,
            collection_name=collection_name,
            project=project,
            location=location
        )

        profile_path = Path(PROJECT_ROOT) / "user_profiles" / "beginner_user_001.json"
        with open(profile_path, 'r') as f:
            test_user_profile = json.load(f)

        framework_to_test = "langchain"
        generated_curriculum = generator.generate(framework_to_test, test_user_profile)

        if "error" not in generated_curriculum:
            output_dir = Path(PROJECT_ROOT) / "data" / "curriculums"
            output_dir.mkdir(exist_ok=True)
            output_path = output_dir / f"dynamic_curriculum_{framework_to_test}.json"
            with open(output_path, 'w') as f:
                json.dump(generated_curriculum, f, indent=2)
            print(f"\nSuccessfully generated and saved test curriculum to {output_path}")
        else:
            print("\nFailed to generate test curriculum.")
            print(generated_curriculum.get("raw_output", ""))

    except Exception as e:
        print(f"An error occurred during test curriculum generation: {e}")


if __name__ == "__main__":
    if "GOOGLE_APPLICATION_CREDENTIALS" not in os.environ:
        print("Warning: GOOGLE_APPLICATION_CREDENTIALS not set. The script will try to use default credentials.")
        # Attempt to set credentials from a local file as a fallback
        credential_path = Path(__file__).parent.parent / "google-credentials.json"
        if credential_path.exists():
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(credential_path)
            print(f"Loaded credentials from {credential_path}")
        else:
             print("Could not find local google-credentials.json.")
    
    asyncio.run(main()) 