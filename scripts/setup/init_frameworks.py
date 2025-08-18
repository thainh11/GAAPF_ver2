
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

    # Build dynamic list of frameworks to initialize
    # Priority order: env → collector.allowed_domains → cached files → fallback defaults
    frameworks_env = os.getenv("GAAPF_FRAMEWORKS", "").strip()
    frameworks_from_env = [fw.strip().lower() for fw in frameworks_env.split(",") if fw.strip()] if frameworks_env else []

    # Initialize collector (used also for discovering allowed frameworks)
    # The memory instance is created below; we can still instantiate collector to read allowed_domains
    tmp_collector = FrameworkCollector(is_logging=False)
    from_env_or_allowed = set(frameworks_from_env or list((tmp_collector.allowed_domains or {}).keys()))

    # Add frameworks present in raw cache directory
    cached_frameworks = set()
    try:
        raw_cache_dir = tmp_collector.raw_cache_dir
        for p in raw_cache_dir.glob("*.json"):
            name = p.stem.lower()
            if name:
                cached_frameworks.add(name)
    except Exception:
        pass

    # Fallback default list
    fallback_list = {"langchain", "langgraph", "crewai", "autogen", "haystack"}

    framework_names = sorted((from_env_or_allowed | cached_frameworks | fallback_list))

    if not framework_names:
        print("No frameworks discovered from environment, collector, or cache.")
        return

    print(f"Frameworks to initialize (final): {', '.join(framework_names)}")

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

    # Allow runtime tuning
    try:
        max_pages = int(os.getenv("GAAPF_COLLECT_MAX_PAGES", "12"))
    except Exception:
        max_pages = 12
    force_refresh = os.getenv("GAAPF_FORCE_REFRESH", "false").lower() in {"1", "true", "yes", "y"}

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
            # If raw cache exists and force_refresh is False, reuse it; otherwise refresh
            raw_cache_file = collector.raw_cache_dir / f"{framework_name.lower().replace(' ', '_')}.json"
            info_data = None
            if raw_cache_file.exists() and not force_refresh:
                print(f"Raw cache already exists for {framework_name}: {raw_cache_file}")
                try:
                    with open(raw_cache_file, 'r', encoding='utf-8') as rf:
                        info_data = json.load(rf)
                except Exception as _e:
                    print(f"Warning: failed reading cache for {framework_name}: {_e}")
            else:
                framework_info = await collector.collect_framework_info(
                    framework_name=framework_name,
                    user_id=user_id,
                    max_pages=max_pages,
                    force_refresh=force_refresh
                )
                if framework_info:
                    # Save to new structured raw cache directory
                    cache_file = collector.raw_cache_dir / f"{framework_name.lower().replace(' ', '_')}.json"
                    with open(cache_file, "w", encoding="utf-8") as f:
                        json.dump(framework_info, f, indent=2)
                    print(f"Cached raw info at: {cache_file}")
                    info_data = framework_info
                else:
                    print(f"Warning: could not collect info for {framework_name}")

            # Step 3: Generate lightweight curriculum JSON from collected info
            try:
                from pathlib import Path as _Path
                cur_dir = Path(__file__).parent.parent.parent / "data" / "curriculums"
                cur_dir.mkdir(parents=True, exist_ok=True)
                cur_path = cur_dir / f"dynamic_curriculum_{framework_name.lower().replace(' ', '_')}.json"

                curriculum = {
                    "framework": framework_name.title(),
                    "user_level": "beginner",
                    "modules": []
                }

                if info_data:
                    off = (info_data.get("official_docs", {}) or {})
                    tutorials = info_data.get("tutorials", []) or []
                    apis = (info_data.get("api_reference", {}) or {})

                    # Official Docs module
                    pages = (off.get("pages") or [])[:5]
                    if pages:
                        resources = []
                        for p in pages:
                            title = (p.get("title") or "").strip()
                            if title:
                                resources.append(f"Concept: {title}")
                        curriculum["modules"].append({
                            "title": "Official Docs",
                            "description": off.get("title") or "Core concepts from official documentation",
                            "topics": [{"resources": resources}]
                        })

                    # Tutorials module
                    if tutorials:
                        resources = []
                        for t in tutorials[:5]:
                            tt = (t.get("title") or "Tutorial").strip()
                            resources.append(f"Concept: {tt}")
                        curriculum["modules"].append({
                            "title": "Tutorials",
                            "description": "Hands-on guides and examples",
                            "topics": [{"resources": resources}]
                        })

                    # API Reference module
                    if apis:
                        resources = []
                        for name, a in list(apis.items())[:5]:
                            title = (a.get("title") or name).strip()
                            resources.append(f"Concept: {title}")
                        curriculum["modules"].append({
                            "title": "API Reference",
                            "description": "Key APIs relevant to development",
                            "topics": [{"resources": resources}]
                        })

                with open(cur_path, 'w', encoding='utf-8') as cf:
                    json.dump(curriculum, cf, indent=2, ensure_ascii=False)
                print(f"Generated curriculum: {cur_path}")
            except Exception as _e:
                print(f"Warning: failed to write curriculum for {framework_name}: {_e}")

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