
import asyncio
import json
from pathlib import Path
from datetime import datetime
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

    # Force deep crawl and disable intro-only behavior regardless of environment
    os.environ.setdefault("ONLY_INTRO_MODE", "false")
    os.environ.setdefault("ONLY_TUTORIALS_MODE", "false")
    os.environ.setdefault("TAVILY_MAP_LIMIT_DOCS", "400")
    os.environ.setdefault("TAVILY_MAX_DOC_URLS", "200")
    os.environ.setdefault("TAVILY_CONCURRENCY", "8")
    os.environ.setdefault("TAVILY_EXTRACT_BATCH", "5")
    os.environ.setdefault("HTTP_FALLBACK_CAP", "100")

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
    fallback_list = {"langchain", "langgraph", "crewai", "agno"}

    framework_names = sorted((from_env_or_allowed | cached_frameworks | fallback_list))

    # Filter strictly by frameworks present in config/intro_urls.json
    intro_cfg_path = Path(__file__).parent.parent.parent / "config" / "intro_urls.json"
    intro_map = {}
    allowed_intro = set()
    try:
        with open(intro_cfg_path, "r", encoding="utf-8") as _f:
            intro_map = json.load(_f) or {}
            allowed_intro = {k.strip().lower() for k, v in intro_map.items() if v}
    except Exception as _e:
        print(f"Warning: failed to load intro_urls.json: {_e}")
        allowed_intro = set()

    if allowed_intro:
        framework_names = [fw for fw in framework_names if fw in allowed_intro]
    else:
        framework_names = []

    if not framework_names:
        print("No frameworks discovered that are present in intro_urls.json.")
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
            # Step 1: Deep crawl strictly from configured base URL, then index
            base_url = (intro_map.get(framework_name, "") or "").strip()
            if not base_url:
                print(f"No base URL for {framework_name} in intro_urls.json; skipping.")
                continue

            # Discovery via Tavily Map + sitemap (strictly same domain as base_url)
            try:
                map_limit = int(os.getenv("TAVILY_MAP_LIMIT_DOCS", "400"))
            except Exception:
                map_limit = 400
            discovered_urls = []
            try:
                discovered = await collector._discover_with_tavily_map(
                    base_url=base_url.rstrip("/"),
                    max_depth=3,
                    max_breadth=60,
                    limit=map_limit,
                )
            except Exception:
                discovered = [base_url.rstrip("/")]
            try:
                sitemap_urls = collector._discover_sitemap_urls(base_url, cap=map_limit)
            except Exception:
                sitemap_urls = []
            discovered_urls = list(dict.fromkeys((discovered or []) + (sitemap_urls or [])))

            # Prioritize and cap, then extract content (advanced), fallback HTTP
            try:
                max_docs = int(os.getenv("TAVILY_MAX_DOC_URLS", "200"))
            except Exception:
                max_docs = 200
            prioritized = collector._prioritize_docs_urls(base_url.rstrip("/"), discovered_urls, max_docs)
            try:
                concurrency = int(os.getenv("TAVILY_CONCURRENCY", "8"))
            except Exception:
                concurrency = 8
            extracted = await collector._extract_urls(prioritized, max_concurrent=concurrency)

            pages_map = {}
            for e in extracted or []:
                u = e.get("url") or ""
                if not u:
                    continue
                pages_map[u] = {
                    "url": u,
                    "title": e.get("title", "") or u,
                    "content": e.get("content", "") or "",
                }
            # HTTP fallback for missing/empty
            try:
                http_cap = int(os.getenv("HTTP_FALLBACK_CAP", "100"))
            except Exception:
                http_cap = 100
            to_fill = [u for u in prioritized if (u not in pages_map) or (not pages_map[u]["content"])]
            if to_fill:
                http_results = collector._http_extract_batch(to_fill[:http_cap])
                for item in http_results:
                    u = item.get("url") or ""
                    if not u:
                        continue
                    pages_map[u] = {"url": u, "title": item.get("title", u), "content": item.get("content", "")}

            pages = []
            for u, v in pages_map.items():
                content = v.get("content", "") or ""
                if not content:
                    continue
                title = v.get("title", "") or u
                pages.append({
                    "url": u,
                    "title": title,
                    "content_summary": content[:800] + ("..." if len(content) > 800 else ""),
                    "is_api_reference": any(t in u.lower() for t in ["api", "reference", "class", "method", "function"]),
                    "content": content,
                })

            # Step 2: Write/refresh raw cache using our strict-base crawl
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
                # Coverage report
                try:
                    coverage = collector._coverage_report(discovered_urls, pages)
                except Exception:
                    coverage = {"discovered_urls": len(discovered_urls), "extracted_urls": len(pages), "coverage": 0.0}
                framework_info = {
                    "framework_name": framework_name,
                    "collection_timestamp": datetime.now().isoformat(),
                    "official_docs": {
                        "main_url": base_url,
                        "title": (pages[0].get("title") if pages else ""),
                        "pages": pages,
                        "coverage": coverage,
                    },
                    "github_info": {},
                    "tutorials": [],
                    "examples": [],
                    "api_reference": {},
                    "concepts": [],
                }
                cache_file = collector.raw_cache_dir / f"{framework_name.lower().replace(' ', '_')}.json"
                with open(cache_file, "w", encoding="utf-8") as f:
                    json.dump(framework_info, f, indent=2)
                print(f"Cached raw info at: {cache_file}")
                info_data = framework_info

            # Step 2.1: Ingest into per-framework VectorStore using official method (includes code blocks)
            try:
                ingest_stats = collector.ingest_official_docs(
                    framework_name=framework_name,
                    project=project,
                    location=location,
                    persistent_dir=str(db_path),
                    force_refresh=False  # reuse our cache
                )
                print(f"Ingested docs for {framework_name}: {ingest_stats}")
            except Exception as _e:
                print(f"Warning: ingest_official_docs failed for {framework_name}: {_e}")

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

            # Step 2.9: Emit Tavily-style JSON outputs (basic and with_code)
            try:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                out_dir = Path("results")
                out_dir.mkdir(parents=True, exist_ok=True)
                framework_id = framework_name.lower().replace(" ", "_")

                base_url = ((info_data or {}).get("official_docs", {}) or {}).get("main_url", "")
                # Use discovered URLs from our earlier step when available; otherwise derive from pages
                discovered_urls = []
                try:
                    # Reconstruct discovered from cache if present
                    if info_data and info_data.get("official_docs", {}).get("coverage", {}).get("discovered_urls"):
                        # We don't store the actual list, so rebuild a reasonable set from pages
                        discovered_urls = list({p.get("url") for p in (info_data.get("official_docs", {}).get("pages", []) or []) if p.get("url")})
                    else:
                        discovered_urls = list({p.get("url") for p in (info_data.get("official_docs", {}).get("pages", []) or []) if p.get("url")})
                except Exception:
                    discovered_urls = []

                pages_payload = []
                for p in ((info_data or {}).get("official_docs", {}).get("pages", []) or []):
                    u = p.get("url") or ""
                    if not u:
                        continue
                    pages_payload.append({
                        "url": u,
                        "title": p.get("title") or u,
                        "content": (p.get("content") or p.get("content_summary") or "")
                    })

                basic_output = {
                    "base_url": base_url,
                    "discovered_count": len(discovered_urls),
                    "extracted_count": len(pages_payload),
                    "discovered_urls": discovered_urls,
                    "pages": pages_payload,
                }
                out_path = out_dir / f"tavily_extract_{framework_id}_{ts}.json"
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(basic_output, f, ensure_ascii=False, indent=2)

                # With code blocks (attach to page entries)
                page_by_url = {p["url"]: p for p in pages_payload}
                total_blocks = 0
                try:
                    max_code_urls = int(os.getenv("TEST_MAX_CODE_URLS", "40"))
                except Exception:
                    max_code_urls = 40
                target_urls = (discovered_urls[:max_code_urls] if discovered_urls else [p["url"] for p in pages_payload][:max_code_urls])

                for u in target_urls:
                    try:
                        blocks = collector._extract_code_blocks_from_url(u) or []
                    except Exception:
                        blocks = []
                    if blocks:
                        total_blocks += len(blocks)
                        if u in page_by_url:
                            page_by_url[u]["code_blocks"] = blocks
                        else:
                            page_by_url[u] = {"url": u, "title": u, "content": "", "code_blocks": blocks}

                updated_pages = []
                seen = set()
                for p in pages_payload:
                    u = p["url"]
                    if u in page_by_url and u not in seen:
                        updated_pages.append(page_by_url[u])
                        seen.add(u)
                for u, p in page_by_url.items():
                    if u not in seen:
                        updated_pages.append(p)

                with_code_output = dict(basic_output)
                with_code_output["pages"] = updated_pages
                with_code_output["total_code_blocks"] = total_blocks

                code_path = out_dir / f"tavily_extract_{framework_id}_{ts}_with_code.json"
                with open(code_path, "w", encoding="utf-8") as f:
                    json.dump(with_code_output, f, ensure_ascii=False, indent=2)

                print({"saved": str(out_path), "saved_with_code": str(code_path), "discovered": len(discovered_urls), "extracted": len(pages_payload), "total_code_blocks": total_blocks})
            except Exception as _e:
                print(f"Tavily-style results write skipped for {framework_name}: {_e}")

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