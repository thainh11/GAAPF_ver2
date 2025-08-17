"""
Framework Information Collection Module for GAAPF Architecture

This module provides tools for collecting comprehensive information about
programming frameworks using web crawling and search technologies.
"""

import json
import logging
import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from datetime import datetime
import re

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import existing tools
from .websearch_tools import search_web
from ..memory.long_term_memory import LongTermMemory
from ..tools.vector_store import VectorStore
from ..utils.async_helpers import run_sync

# Try to import Tavily
try:
    from tavily import TavilyClient
    TAVILY_AVAILABLE = True
    logger.info("Tavily is available")
except ImportError:
    TAVILY_AVAILABLE = False
    logger.warning("Tavily not available. Using fallback implementation.")

class FrameworkCollector:
    """
    Framework information collection module that gathers comprehensive data
    about programming frameworks using web crawling and search technologies.
    """
    
    def __init__(
        self,
        memory: Optional[LongTermMemory] = None,
        cache_dir: Optional[Union[Path, str]] = Path("data/framework_cache"),
        is_logging: bool = False,
        tavily_api_key: Optional[str] = None
    ):
        """
        Initialize the framework collector.
        
        Args:
            memory: LongTermMemory instance for storing collected information
            cache_dir: Directory to cache framework information
            is_logging: Whether to enable detailed logging
            tavily_api_key: API key for Tavily if using direct client
        """
        self.memory = memory
        self.cache_dir = Path(cache_dir) if isinstance(cache_dir, str) else cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        # New: structured paths for easier control
        # Raw JSON cache directory (new location)
        self.raw_cache_dir = Path("data/frameworks/cache")
        self.raw_cache_dir.mkdir(parents=True, exist_ok=True)
        # Vector DB directory for per-framework docs (RAG)
        self.vectordb_dir = Path("data/frameworks/vectordb")
        self.vectordb_dir.mkdir(parents=True, exist_ok=True)
        self.is_logging = is_logging
        
        # Initialize Tavily client if API key is provided
        self.tavily_client = None
        if TAVILY_AVAILABLE and tavily_api_key:
            self.tavily_client = TavilyClient(api_key=tavily_api_key)
        
        if self.is_logging:
            logger.info(f"Initialized FrameworkCollector with cache at {self.cache_dir}")

        # Allowed documentation domains per framework to keep sources authoritative
        self.allowed_domains: Dict[str, List[str]] = {
            "langchain": [
                "python.langchain.com",
                "langchain.com",
            ],
            "langgraph": [
                "langchain-ai.github.io",
            ],
            "crewai": [
                "docs.crewai.com",
            ],
            "autogen": [
                "microsoft.github.io",
            ],
            "haystack": [
                "docs.haystack.deepset.ai",
            ],
        }
    
    async def collect_framework_info(
        self,
        framework_name: str,
        user_id: str,
        include_github: bool = True,
        include_docs: bool = True,
        max_pages: int = 50,
        force_refresh: bool = False
    ) -> Dict:
        """
        Collect comprehensive information about a framework.
        
        Args:
            framework_name: Name of the framework to collect info about
            user_id: User ID to associate with the collected information
            include_github: Whether to include GitHub repository information
            include_docs: Whether to include official documentation
            max_pages: Maximum number of pages to crawl
            force_refresh: Whether to force refresh cached information
            
        Returns:
            Dictionary containing collected framework information
        """
        # Check cache first if not forcing refresh (new path first, fallback to old)
        cache_file_new = self.raw_cache_dir / f"{framework_name.lower().replace(' ', '_')}.json"
        cache_file_old = self.cache_dir / f"{framework_name.lower().replace(' ', '_')}.json"
        if not force_refresh:
            if cache_file_new.exists():
                with open(cache_file_new, "r", encoding="utf-8") as f:
                    cached_data = json.load(f)
                    if self.is_logging:
                        logger.info(f"Using cached information (new path) for {framework_name}")
                    return cached_data
            if cache_file_old.exists():
                with open(cache_file_old, "r", encoding="utf-8") as f:
                    cached_data = json.load(f)
                    if self.is_logging:
                        logger.info(f"Using cached information (legacy path) for {framework_name}")
                    return cached_data
        
        # Initialize results dictionary
        results = {
            "framework_name": framework_name,
            "collection_timestamp": datetime.now().isoformat(),
            "official_docs": {},
            "github_info": {},
            "tutorials": [],
            "examples": [],
            "api_reference": {},
            "concepts": []
        }
        
        # Step 1: Use Tavily to find main resources
        search_results = search_web(f"{framework_name} official documentation github repository", num_results=5)
        
        # Extract official documentation URL and GitHub repository
        docs_url = None
        github_url = None
        
        for result in search_results.get("results", []):
            url = result.get("url", "")
            snippet = result.get("snippet", "")
            title = result.get("title", "")
            
            if "github.com" in url and not github_url and include_github:
                github_url = url
                results["github_info"]["url"] = url
                results["github_info"]["title"] = title
                results["github_info"]["snippet"] = snippet
                # Extract more github info immediately
                results["github_info"]["readme"] = snippet  # Use snippet as readme content
            elif (
                any(term in url.lower() for term in ["docs", "documentation", "guide", "tutorial"]) and
                not docs_url and include_docs and self._is_allowed_url(framework_name, url)
            ):
                docs_url = url
                results["official_docs"]["main_url"] = url
                results["official_docs"]["title"] = title
                results["official_docs"]["snippet"] = snippet
        
        # Step 2: Extract documentation content
        if docs_url:
            docs_info = await self._extract_documentation(docs_url, framework_name)
            results["official_docs"].update(docs_info)
        
        # Step 3: Extract GitHub repository if available
        if github_url:
            github_info = await self._extract_github_repo(github_url, framework_name)
            results["github_info"].update(github_info)
        
        # Step 2.5: Enhanced content extraction for existing data
        # Fill in missing content using search snippets
        for page in results["official_docs"].get("pages", []):
            if not page.get("content_summary"):
                # Search for specific content about this page
                page_search = search_web(f"{framework_name} {page.get('title', '')}", num_results=1)
                if page_search.get("results"):
                    page["content_summary"] = page_search["results"][0].get("snippet", "")
        
        # Ensure all tutorials have content
        for tutorial in results["tutorials"]:
            if not tutorial.get("snippet"):
                tutorial_search = search_web(f"{tutorial.get('title', '')} {framework_name}", num_results=1)
                if tutorial_search.get("results"):
                    tutorial["snippet"] = tutorial_search["results"][0].get("snippet", "")
        
        # Step 4: Find tutorials and examples
        tutorial_results = search_web(f"{framework_name} tutorial examples getting started", num_results=5)
        for result in tutorial_results.get("results", []):
            snippet = result.get("snippet", "")
            tutorial = {
                "title": result.get("title", ""),
                "url": result.get("url", ""),
                "snippet": snippet,
                "source": "tavily_search"
            }
            results["tutorials"].append(tutorial)
            
            # Also add examples if found in the snippet
            if any(term in snippet.lower() for term in ["example", "sample", "demo"]):
                results["examples"].append({
                    "title": result.get("title", ""),
                    "url": result.get("url", ""),
                    "snippet": snippet
                })
        
        # Step 5: Find API reference
        api_results = search_web(f"{framework_name} API reference documentation class methods", num_results=5)
        for result in api_results.get("results", []):
            url = result.get("url", "")
            snippet = result.get("snippet", "")
            if any(term in url.lower() for term in ["api", "reference", "class", "method"]) or any(term in snippet.lower() for term in ["api", "reference", "class", "method"]):
                api_entry = {
                    "title": result.get("title", ""),
                    "url": url,
                    "snippet": snippet,
                    "source": "tavily_search"
                }
                results["api_reference"][result.get("title", "API")] = api_entry
        
        # Step 6: Extract concepts from collected information
        results["concepts"] = self._extract_concepts_from_results(results, framework_name)
        
        # Step 7: Store in cache (new path)
        cache_file_new = self.raw_cache_dir / f"{framework_name.lower().replace(' ', '_')}.json"
        with open(cache_file_new, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        
        # Step 8: Store in long-term memory if available
        if self.memory:
            self._store_in_memory(results, user_id, framework_name)
        
        return results

    async def _extract_documentation(self, docs_url: str, framework_name: str) -> Dict:
        """Extract content from the framework's official documentation and discover key pages."""
        # Use Tavily extract API if available
        if self.tavily_client:
            try:
                extraction = self.tavily_client.extract(url=docs_url)
                return {
                    "content": extraction.get("content", ""),
                    "title": extraction.get("title", ""),
                    "pages": [{
                        "url": docs_url,
                        "title": extraction.get("title", ""),
                        "content_summary": extraction.get("content", "")[:500] + "..." if len(extraction.get("content", "")) > 500 else extraction.get("content", ""),
                    }]
                }
            except Exception as e:
                logger.error(f"Error using Tavily extract API: {e}")

        # Fallback to search for more documentation pages
        docs_info = {
            "main_url": docs_url,
            "pages": []
        }

        # Search for specific documentation pages
        search_terms = [
            f"{framework_name} getting started guide",
            f"{framework_name} core concepts documentation",
            f"{framework_name} API reference",
            f"{framework_name} examples documentation",
            f"{framework_name} advanced usage guide"
        ]

        for term in search_terms:
            search_results = search_web(term, num_results=2)
            for result in search_results.get("results", []):
                url = result.get("url", "")
                if not self._is_allowed_url(framework_name, url):
                    continue
                page_info = {
                    "url": url,
                    "title": result.get("title", ""),
                    "content_summary": result.get("snippet", ""),
                    "is_api_reference": any(t in url.lower() for t in ["api", "reference", "class", "method", "function"])
                }
                docs_info["pages"].append(page_info)

        return docs_info

    async def _extract_github_repo(self, github_url: str, framework_name: str) -> Dict:
        """Extract information from the framework's GitHub repository."""
        # Use Tavily extract API if available
        if self.tavily_client:
            try:
                extraction = self.tavily_client.extract(url=github_url)
                return {
                    "readme": extraction.get("content", ""),
                    "title": extraction.get("title", ""),
                    "examples": self._extract_example_links(extraction.get("content", ""))
                }
            except Exception as e:
                logger.error(f"Error using Tavily extract API: {e}")

        # Fallback to search for GitHub information
        github_info = {
            "url": github_url,
            "readme": "",
            "examples": []
        }

        # Search for README content
        readme_results = search_web(f"{framework_name} github readme", num_results=1)
        if readme_results.get("results"):
            github_info["readme"] = readme_results["results"][0].get("snippet", "")

        # Search for examples
        examples_results = search_web(f"{framework_name} github examples", num_results=3)
        for result in examples_results.get("results", []):
            if "github.com" in result.get("url", ""):
                github_info["examples"].append({
                    "url": result.get("url", ""),
                    "title": result.get("title", "")
                })

        return github_info
        
    def _extract_example_links(self, content: str) -> List[Dict]:
        """Extract example links from GitHub content"""
        examples = []
        
        # Look for markdown links that might be examples
        link_pattern = r"\[([^\]]+)\]\(([^)]+)\)"
        links = re.findall(link_pattern, content)
        
        for link_text, link_url in links:
            if any(term in link_text.lower() for term in ["example", "sample", "demo", "tutorial"]):
                examples.append({
                    "url": link_url,
                    "title": link_text
                })
        
        return examples
    
    def _extract_concepts_from_text(self, text: str, framework_name: str) -> List[Dict]:
        """Extract concepts from text using simple heuristics"""
        concepts = []
        
        # Look for capitalized terms that might be framework concepts
        # This is a simple heuristic and would need refinement for production
        words = re.findall(r"\b([A-Z][a-zA-Z0-9]+)\b", text)
        
        # Filter out common words
        common_words = ["The", "This", "That", "These", "Those", "When", "Where", "Why", "How"]
        filtered_words = [word for word in words if word not in common_words]
        
        # Add unique concepts
        unique_concepts = set(filtered_words)
        for concept in unique_concepts:
            # Try to find a sentence containing this concept
            concept_pattern = r"[^.!?]*\b" + re.escape(concept) + r"\b[^.!?]*[.!?]"
            concept_sentences = re.findall(concept_pattern, text)
            
            description = ""
            if concept_sentences:
                description = concept_sentences[0].strip()
            
            concepts.append({
                "name": concept,
                "description": description if description else f"Concept related to {framework_name}",
                "source": "text_extraction"
            })
        
        # Also look for terms in code blocks
        code_pattern = r"```[a-zA-Z]*\n(.*?)\n```"
        code_blocks = re.findall(code_pattern, text, re.DOTALL)
        
        for code in code_blocks:
            # Look for import statements or class/function definitions
            if framework_name.lower() in code.lower():
                import_pattern = r"(?:import|from)\s+([a-zA-Z0-9_.]+)"
                imports = re.findall(import_pattern, code)
                
                for imp in imports:
                    if framework_name.lower() in imp.lower():
                        concepts.append({
                            "name": imp,
                            "description": f"Module or package in {framework_name}",
                            "source": "code_extraction"
                        })
                
                # Look for class definitions
                class_pattern = r"class\s+([A-Za-z0-9_]+)"
                classes = re.findall(class_pattern, code)
                
                for cls in classes:
                    concepts.append({
                        "name": cls,
                        "description": f"Class in {framework_name}",
                        "source": "code_extraction"
                    })
        
        return concepts
    
    def _extract_concepts_from_results(self, results: Dict, framework_name: str) -> List[Dict]:
        """Extract concepts from all collected information"""
        all_concepts = []
        
        # Extract from official docs
        for page in results.get("official_docs", {}).get("pages", []):
            content = page.get("content_summary", "")
            if content:
                concepts = self._extract_concepts_from_text(content, framework_name)
                all_concepts.extend(concepts)
        
        # Extract from main official docs snippet
        main_snippet = results.get("official_docs", {}).get("snippet", "")
        if main_snippet:
            concepts = self._extract_concepts_from_text(main_snippet, framework_name)
            all_concepts.extend(concepts)
        
        # Extract from GitHub readme
        readme_content = results.get("github_info", {}).get("readme", "")
        if readme_content:
            concepts = self._extract_concepts_from_text(readme_content, framework_name)
            all_concepts.extend(concepts)
        
        # Extract from github snippet
        github_snippet = results.get("github_info", {}).get("snippet", "")
        if github_snippet:
            concepts = self._extract_concepts_from_text(github_snippet, framework_name)
            all_concepts.extend(concepts)
        
        # Extract from tutorials
        for tutorial in results.get("tutorials", []):
            snippet = tutorial.get("snippet", "")
            if snippet:
                concepts = self._extract_concepts_from_text(snippet, framework_name)
                all_concepts.extend(concepts)
        
        # Extract from API references
        for api_name, api_info in results.get("api_reference", {}).items():
            api_snippet = api_info.get("snippet", "")
            if api_snippet:
                concepts = self._extract_concepts_from_text(api_snippet, framework_name)
                all_concepts.extend(concepts)
        
        # Add some fallback concepts if none were found
        if not all_concepts:
            # Add basic framework concepts based on common patterns
            basic_concepts = [
                {"name": framework_name, "description": f"The main {framework_name} framework", "source": "fallback"},
                {"name": "Installation", "description": f"How to install {framework_name}", "source": "fallback"},
                {"name": "Getting Started", "description": f"Basic {framework_name} usage", "source": "fallback"},
                {"name": "API", "description": f"{framework_name} API interface", "source": "fallback"},
                {"name": "Examples", "description": f"{framework_name} usage examples", "source": "fallback"}
            ]
            all_concepts.extend(basic_concepts)
        
        # Deduplicate concepts
        return list({concept["name"]: concept for concept in all_concepts}.values())
    
    def _store_in_memory(self, framework_data: Dict, user_id: str, framework_name: str):
        """Store framework information in long-term memory"""
        if not self.memory:
            return
        
        # Store general framework information
        description = framework_data.get("official_docs", {}).get("snippet", "No description available")
        self.memory.add_external_knowledge(
            text=f"Framework: {framework_name}\nDescription: {description}",
            user_id=user_id,
            source=f"framework_collection_{framework_name}",
            metadata={"framework": framework_name, "type": "overview"}
        )
        
        # Store concepts
        for concept in framework_data.get("concepts", []):
            self.memory.add_external_knowledge(
                text=f"Concept: {concept['name']}\nDescription: {concept['description']}",
                user_id=user_id,
                source=f"framework_collection_{framework_name}",
                metadata={"framework": framework_name, "type": "concept", "concept_name": concept["name"]}
            )
        
        # Store documentation pages  
        for page in framework_data.get("official_docs", {}).get("pages", []):
            content = page.get("content_summary", "")
            if content:
                self.memory.add_external_knowledge(
                    text=f"Documentation: {page['title']}\nContent: {content}",
                    user_id=user_id,
                    source=f"framework_collection_{framework_name}",
                    metadata={"framework": framework_name, "type": "documentation", "page_title": page["title"]}
                )
        
        # Store tutorials
        for tutorial in framework_data.get("tutorials", []):
            snippet = tutorial.get("snippet", "")
            if snippet:
                self.memory.add_external_knowledge(
                    text=f"Tutorial: {tutorial['title']}\nContent: {snippet}",
                    user_id=user_id,
                    source=f"framework_collection_{framework_name}",
                    metadata={"framework": framework_name, "type": "tutorial", "tutorial_title": tutorial["title"]}
                )
        
        # Store API sections
        for api_name, api_info in framework_data.get("api_reference", {}).items():
            self.memory.add_external_knowledge(
                text=f"API: {api_name}\nDescription: {api_info.get('snippet', '')}",
                user_id=user_id,
                source=f"framework_collection_{framework_name}",
                metadata={"framework": framework_name, "type": "api", "api_name": api_name}
            )

    def ingest_official_docs(
        self,
        framework_name: str,
        project: Optional[str] = None,
        location: Optional[str] = None,
        persistent_dir: Optional[Union[str, Path]] = None,
        force_refresh: bool = False,
    ) -> Dict[str, Any]:
        """
        Ingest official documentation snippets into a per-framework VectorStore collection.

        Returns minimal stats: {"framework": str, "collection": str, "added": int}
        """
        framework_id = framework_name.lower().replace(" ", "_")
        collection_name = f"framework_docs_{framework_id}"

        # Collect docs info (uses cache unless force_refresh)
        info = run_sync(self.collect_framework_info(
            framework_name=framework_name,
            user_id="system_ingest",
            include_github=False,
            include_docs=True,
            max_pages=10,
            force_refresh=force_refresh,
        ))

        pages = (info.get("official_docs", {}) or {}).get("pages", [])
        main_snippet = (info.get("official_docs", {}) or {}).get("snippet", "")
        main_url = (info.get("official_docs", {}) or {}).get("main_url", "")

        documents: List[Dict[str, Any]] = []
        # Add main page snippet if present
        if main_snippet:
            documents.append({
                "text": main_snippet,
                "metadata": {
                    "framework": framework_id,
                    "url": main_url,
                    "title": info.get("official_docs", {}).get("title", ""),
                    "source": "official_docs",
                }
            })

        # Add each page's content_summary
        for p in pages:
            text = (p.get("content_summary") or "").strip()
            url = p.get("url", "")
            if not text or not self._is_allowed_url(framework_name, url):
                continue
            documents.append({
                "text": text,
                "metadata": {
                    "framework": framework_id,
                    "url": url,
                    "title": p.get("title", ""),
                    "source": "official_docs",
                    "is_api_reference": bool(p.get("is_api_reference")),
                }
            })

        if not documents:
            # Fallback: build minimal documents from tutorials/api_reference snippets to avoid empty DB
            for tutorial in info.get("tutorials", [])[:5]:
                snippet = (tutorial.get("snippet") or "").strip()
                url = tutorial.get("url", "")
                title = tutorial.get("title", "Tutorial")
                
                # Use title and URL as fallback content if snippet is empty
                content = snippet if snippet else f"Tutorial: {title}. URL: {url}"
                if content and url:
                    documents.append({
                        "text": content,
                        "metadata": {
                            "framework": framework_id,
                            "url": url,
                            "title": title,
                            "source": "tutorial"
                        }
                    })
                    
            for api_name, api in list(info.get("api_reference", {}).items())[:5]:
                snippet = (api.get("snippet") or "").strip()
                url = api.get("url", "")
                title = api.get("title", api_name)
                
                # Use title and URL as fallback content if snippet is empty
                content = snippet if snippet else f"API Reference: {title}. URL: {url}"
                if content and url:
                    documents.append({
                        "text": content,
                        "metadata": {
                            "framework": framework_id,
                            "url": url,
                            "title": title,
                            "source": "api_reference"
                        }
                    })
                    
            if not documents:
                return {"framework": framework_id, "collection": collection_name, "added": 0}

        # Upsert into per-framework VectorStore (separate folder per framework)
        vs_base = Path(persistent_dir) if isinstance(persistent_dir, (str, Path)) and persistent_dir else Path(self.vectordb_dir)
        vs_dir_path = vs_base / framework_id
        try:
            vs_dir_path.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        vs = VectorStore(
            persistent_dir=str(vs_dir_path),
            collection_name=collection_name,
            project=project,
            location=location or "us-central1",
        )
        ids = vs.add_documents(documents)
        return {"framework": framework_id, "collection": collection_name, "added": len(ids)}

    def ensure_ingested(
        self,
        framework_name: str,
        project: Optional[str] = None,
        location: Optional[str] = None,
        persistent_dir: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        """Ensure a per-framework collection exists with data; ingest if empty."""
        framework_id = framework_name.lower().replace(" ", "_")
        collection_name = f"framework_docs_{framework_id}"
        vs_base = Path(persistent_dir) if isinstance(persistent_dir, (str, Path)) and persistent_dir else Path(self.vectordb_dir)
        vs_dir_path = vs_base / framework_id
        try:
            vs_dir_path.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        vs = VectorStore(
            persistent_dir=str(vs_dir_path),
            collection_name=collection_name,
            project=project,
            location=location or "us-central1",
        )
        if vs.count() > 0:
            return {"framework": framework_id, "collection": collection_name, "added": 0, "exists": True}
        return self.ingest_official_docs(
            framework_name=framework_name,
            project=project,
            location=location,
            persistent_dir=str(vs_dir_path),
            force_refresh=True,
        )

    def _is_allowed_url(self, framework_name: str, url: str) -> bool:
        try:
            if not url:
                return False
            fw = framework_name.lower()
            domains = self.allowed_domains.get(fw, [])
            return any(domain in url for domain in domains) or (not domains)
        except Exception:
            return False

# Expose methods as module-level callables to align with tools.json entries
# These thin wrappers allow dynamic tool loader to find functions by name

def _extract_documentation(docs_url: str, framework_name: str) -> Dict:  # pyright: ignore[reportUnusedFunction]
    collector = FrameworkCollector(is_logging=False)
    return run_sync(collector._extract_documentation(docs_url, framework_name))

def _extract_github_repo(github_url: str, framework_name: str) -> Dict:  # pyright: ignore[reportUnusedFunction]
    collector = FrameworkCollector(is_logging=False)
    return run_sync(collector._extract_github_repo(github_url, framework_name))

def _extract_example_links(content: str) -> List[Dict]:  # pyright: ignore[reportUnusedFunction]
    collector = FrameworkCollector(is_logging=False)
    return collector._extract_example_links(content)

def _extract_concepts_from_text(text: str, framework_name: str) -> List[Dict]:  # pyright: ignore[reportUnusedFunction]
    collector = FrameworkCollector(is_logging=False)
    return collector._extract_concepts_from_text(text, framework_name)

def _extract_concepts_from_results(results: Dict, framework_name: str) -> List[Dict]:  # pyright: ignore[reportUnusedFunction]
    collector = FrameworkCollector(is_logging=False)
    return collector._extract_concepts_from_results(results, framework_name)

def _store_in_memory(framework_data: Dict, user_id: str, framework_name: str):  # pyright: ignore[reportUnusedFunction]
    collector = FrameworkCollector(is_logging=False)
    # Only store if a memory was provided in constructor; otherwise no-op
    try:
        collector._store_in_memory(framework_data, user_id, framework_name)
    except Exception:
        return None

async def collect_framework_info(
    framework_name: str,
    user_id: str,
    include_github: bool = True,
    include_docs: bool = True,
    max_pages: int = 50,
    force_refresh: bool = False,
):
    collector = FrameworkCollector(is_logging=True)
    return await collector.collect_framework_info(
        framework_name=framework_name,
        user_id=user_id,
        include_github=include_github,
        include_docs=include_docs,
        max_pages=max_pages,
        force_refresh=force_refresh,
    )

def FrameworkCollector_collect_framework_info(
    framework_name: str,
    user_id: str,
    include_github: bool = True,
    include_docs: bool = True,
    max_pages: int = 50,
    force_refresh: bool = False,
):
    """Sync wrapper to satisfy tools.json alias naming convention."""
    return run_sync(collect_framework_info(
        framework_name=framework_name,
        user_id=user_id,
        include_github=include_github,
        include_docs=include_docs,
        max_pages=max_pages,
        force_refresh=force_refresh,
    ))

async def initialize_framework_knowledge(
    framework_name: str,
    user_id: str,
    knowledge_graph,
    memory,
    is_quick_init: bool = True
):
    """Initialize framework knowledge with either quick or comprehensive collection"""
    collector = FrameworkCollector(memory=memory, is_logging=True)
    
    if is_quick_init:
        # Quick initialization during onboarding
        # Use limited pages and rely more on search results
        framework_info = await collector.collect_framework_info(
            framework_name=framework_name,
            user_id=user_id,
            max_pages=5,  # Limit initial crawling
            include_github=False  # Skip GitHub initially
        )
        
        # Create initial curriculum from limited information
        return create_initial_curriculum(framework_info, knowledge_graph)
    else:
        # Comprehensive background collection
        # This can run in a background task
        framework_info = await collector.collect_framework_info(
            framework_name=framework_name,
            user_id=user_id,
            max_pages=50,  # More comprehensive crawling
            include_github=True,
            force_refresh=True  # Get fresh data
        )
        
        # Update knowledge graph with comprehensive information
        update_knowledge_graph(framework_info, knowledge_graph, user_id)
        
        return framework_info

def create_initial_curriculum(framework_info: Dict, knowledge_graph) -> Dict:
    """Create an initial curriculum based on limited framework information"""
    curriculum = {
        "framework": framework_info["framework_name"],
        "description": framework_info.get("official_docs", {}).get("snippet", ""),
        "modules": [
            {
                "title": "Introduction",
                "concepts": [],
                "resources": []
            },
            {
                "title": "Core Concepts",
                "concepts": [],
                "resources": []
            },
            {
                "title": "Practical Application",
                "concepts": [],
                "resources": []
            }
        ]
    }
    
    # Add concepts to modules
    concepts = framework_info.get("concepts", [])
    
    # Sort concepts by relevance (using simple heuristic)
    concepts.sort(key=lambda x: len(x.get("description", "")), reverse=True)
    
    # Distribute concepts across modules
    for i, concept in enumerate(concepts):
        if i < 3:  # First few concepts go to Introduction
            curriculum["modules"][0]["concepts"].append(concept["name"])
        elif i < 10:  # Next set of concepts go to Core Concepts
            curriculum["modules"][1]["concepts"].append(concept["name"])
        else:  # Remaining concepts go to Practical Application
            curriculum["modules"][2]["concepts"].append(concept["name"])
    
    # Add resources
    for page in framework_info.get("official_docs", {}).get("pages", [])[:5]:
        curriculum["modules"][0]["resources"].append({
            "title": page.get("title", "Documentation"),
            "url": page.get("url", ""),
            "type": "documentation"
        })
    
    for tutorial in framework_info.get("tutorials", [])[:3]:
        curriculum["modules"][2]["resources"].append({
            "title": tutorial.get("title", "Tutorial"),
            "url": tutorial.get("url", ""),
            "type": "tutorial"
        })
    
    return curriculum

def update_knowledge_graph(framework_info: Dict, knowledge_graph, user_id: str):
    """Update knowledge graph with framework information"""
    # Add framework as a main concept
    framework_name = framework_info["framework_name"]
    
    # Add concepts and their relationships
    for concept in framework_info.get("concepts", []):
        concept_name = concept["name"]
        knowledge_graph._add_concept_if_not_exists(concept_name, framework_name, "core")
        
        # Connect concept to framework
        knowledge_graph._add_relationship(framework_name, concept_name, "has_concept")
    
    # Add API components
    for api_name, api_info in framework_info.get("api_reference", {}).items():
        knowledge_graph._add_concept_if_not_exists(api_name, framework_name, "api")
        knowledge_graph._add_relationship(framework_name, api_name, "has_api")
