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
import urllib.request
from urllib.parse import urlparse
import html as _html
import hashlib
import os

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
        # Intro URL config (optional): config/intro_urls.json
        self._intro_urls: Dict[str, str] = {}
        try:
            # Attempt to resolve project root and load config/intro_urls.json
            possible = [
                Path("config") / "intro_urls.json",
                Path(__file__).parent.parent.parent.parent / "config" / "intro_urls.json",
            ]
            for p in possible:
                if isinstance(p, Path) and p.exists():
                    with open(p, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        if isinstance(data, dict):
                            # normalize keys to lowercase
                            self._intro_urls = {str(k).lower(): str(v) for k, v in data.items()}
                    break
        except Exception as _e:
            if self.is_logging:
                logger.debug(f"intro_urls.json not loaded: {_e}")
        
        # Initialize Tavily client (prefer explicit arg, fallback to env)
        self.tavily_client = None
        if TAVILY_AVAILABLE:
            api_key = tavily_api_key or os.getenv("TAVILY_API_KEY")
            if api_key:
                self.tavily_client = TavilyClient(api_key=api_key)
        
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
        
        # Intro URL resolver: use configured URL if available, otherwise fallback to provided docs_url
        
    def _get_intro_url(self, framework_name: str, docs_url_fallback: str) -> str:
        try:
            key = (framework_name or "").strip().lower()
            if key in self._intro_urls and self._intro_urls[key]:
                return self._intro_urls[key]
        except Exception:
            pass
        return docs_url_fallback
    
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
        
        # Step 1: Prefer configured intro URL (config/intro_urls.json) to avoid unnecessary web search
        docs_url = None
        try:
            configured_intro = self._get_intro_url(framework_name, "")
            if configured_intro:
                docs_url = configured_intro
                results["official_docs"]["main_url"] = configured_intro
                results["official_docs"]["title"] = configured_intro
        except Exception:
            pass

        # If not configured, fall back to web search to locate main docs
        if not docs_url:
            search_results = search_web(f"{framework_name} official documentation", num_results=5)
        
        # Extract official documentation URL and GitHub repository
        github_url = None
        
        if not docs_url:
            for result in search_results.get("results", []):
                url = result.get("url", "")
                snippet = result.get("snippet", "")
                title = result.get("title", "")
                
                if (
                    any(term in url.lower() for term in ["docs", "documentation", "guide", "tutorial"]) and
                    not docs_url and include_docs and self._is_allowed_url(framework_name, url)
                ):
                    docs_url = url
                    results["official_docs"]["main_url"] = url
                    results["official_docs"]["title"] = title
                    results["official_docs"]["snippet"] = snippet
        
        # Optional: disable GitHub collection via environment override
        try:
            if os.getenv("DISABLE_GITHUB", "false").strip().lower() == "true":
                github_url = None
        except Exception:
            pass
        
        # Step 2: Extract documentation content
        if docs_url:
            docs_info = await self._extract_documentation(docs_url, framework_name)
            results["official_docs"].update(docs_info)
        
        # Step 3: GitHub processing removed
        
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
        # Anchor base to docs root when domain matches
        try:
            parsed = urlparse(docs_url)
            if parsed.netloc.endswith("python.langchain.com"):
                docs_url = f"{parsed.scheme}://{parsed.netloc}/docs/"
        except Exception:
            pass

        # Intro-only mode by default: fetch a single configured introduction hub page
        try:
            only_intro = os.getenv("ONLY_INTRO_MODE", "true").strip().lower() == "true"
        except Exception:
            only_intro = True
        if only_intro:
            intro_url = self._get_intro_url(framework_name, docs_url)
            pages: List[Dict[str, Any]] = []
            if intro_url:
                http_results = self._http_extract_batch([intro_url])
                if http_results:
                    item = http_results[0]
                    content = item.get("content", "") or ""
                    if content:
                        pages = [{
                            "url": intro_url,
                            "title": item.get("title", "") or intro_url,
                            "content_summary": content[:800] + ("..." if len(content) > 800 else ""),
                            "is_api_reference": False,
                            "content": content,
                        }]
            # Index and return coverage for intro-only
            self._index_docs(framework_name, pages)
            coverage = {"discovered_urls": 1 if intro_url else 0, "extracted_urls": len(pages), "coverage": 100.0 if pages else 0.0}
            return {
                "content": "",
                "title": pages[0].get("title", "") if pages else "",
                "pages": pages,
                "coverage": coverage,
            }
        # Prefer Tavily Map + Extract for broad coverage if available
        if self.tavily_client:
            try:
                base = docs_url.rstrip("/")
                # Optional tutorial-only mode: restrict to tutorial/how-to pages
                try:
                    only_tutorials = os.getenv("ONLY_TUTORIALS_MODE", "false").strip().lower() == "true"
                except Exception:
                    only_tutorials = False
                map_limit = int(os.getenv("TAVILY_MAP_LIMIT_DOCS", "200"))
                discovered = await self._discover_with_tavily_map(base_url=base, max_depth=3, max_breadth=60, limit=map_limit)
                # Merge sitemap for broader coverage
                sitemap_urls = self._discover_sitemap_urls(base, cap=map_limit)
                discovered = list(dict.fromkeys(discovered + sitemap_urls))
                if only_tutorials:
                    tutorial_like = [
                        u for u in discovered
                        if any(k in u.lower() for k in ["/tutorial", "/tutorials", "how_to", "how-to"]) and "/api" not in u.lower()
                    ]
                    if not tutorial_like:
                        try:
                            pu = urlparse(base)
                            seeds = [
                                f"{pu.scheme}://{pu.netloc}/docs/tutorials/",
                                f"{pu.scheme}://{pu.netloc}/docs/how_to/",
                                f"{pu.scheme}://{pu.netloc}/tutorials/",
                            ]
                            tutorial_like = list(dict.fromkeys(seeds))
                        except Exception:
                            tutorial_like = []
                    discovered = tutorial_like
                # Prioritize and cap URLs to respect usage budget
                max_docs = int(os.getenv("TAVILY_MAX_DOC_URLS", "120"))
                discovered = self._prioritize_docs_urls(base, discovered, max_docs)
                concurrency = int(os.getenv("TAVILY_CONCURRENCY", "6"))
                extracted = await self._extract_urls(discovered, max_concurrent=concurrency)
                # HTTP fallback for empty results or empty content pages
                pages_map: Dict[str, Dict[str, Any]] = {}
                for e in extracted:
                    u = e.get("url", "")
                    if not u:
                        continue
                    pages_map[u] = {
                        "url": u,
                        "title": e.get("title", "") or u,
                        "content": e.get("content", "") or "",
                    }
                # Determine which URLs still need content
                to_fill = [u for u in discovered[:max_docs] if (u not in pages_map) or (not pages_map[u]["content"])]
                if to_fill:
                    # Limit fallback batch size to keep budget in check
                    http_cap = int(os.getenv("HTTP_FALLBACK_CAP", "60"))
                    http_results = self._http_extract_batch(to_fill[:http_cap])
                    for item in http_results:
                        u = item["url"]
                        pages_map[u] = {"url": u, "title": item.get("title", u), "content": item.get("content", "")}
                # Build pages only for non-empty content
                pages: List[Dict[str, Any]] = []
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
                # If still empty, try single-page fallback via HTTP
                if not pages:
                    single_http = self._http_extract_batch([base])
                    if single_http:
                        item = single_http[0]
                        content = item.get("content", "") or ""
                        pages = [{
                            "url": base,
                            "title": item.get("title", "") or base,
                            "content_summary": content[:800] + ("..." if len(content) > 800 else ""),
                            "is_api_reference": False,
                            "content": content,
                        }]
                # Index
                self._index_docs(framework_name, pages)
                coverage = self._coverage_report(discovered, pages)
                return {
                    "content": "",
                    "title": pages[0].get("title", "") if pages else "",
                    "pages": pages,
                    "coverage": coverage,
                }
            except Exception as e:
                logger.error(f"Error using Tavily Map/Extract: {e}")

        # Fallback to search for more documentation pages (only if we have a docs_url)
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

        if docs_url:
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

    # GitHub repository extraction has been removed per configuration
        
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

    def _prioritize_docs_urls(self, base_url: str, urls: List[str], max_urls: int) -> List[str]:
        """Prioritize key documentation sections and cap the list for budget control."""
        if not urls:
            return [base_url]
        base_netloc = urlparse(base_url).netloc
        urls = [u for u in urls if urlparse(u).netloc == base_netloc]
        # Exclude heavy integrations pages for initial pass
        urls = [u for u in urls if "/docs/integrations" not in u.lower()]
        # Score by presence of high-value keywords
        def score(u: str) -> int:
            s = u.lower()
            pts = 0
            for kw in ["/docs/", "getting", "tutorial", "guide", "how_to", "how-to", "concept", "api", "reference", "introduction", "intro"]:
                if kw in s:
                    pts += 1
            # Prefer shallower paths
            depth = s.count('/')
            return pts * 10 - depth
        unique_urls = list(dict.fromkeys(urls))
        ranked = sorted(unique_urls, key=score, reverse=True)
        # Keep only positively scored URLs; fallback to shallow top if empty
        ranked_pos = [u for u in ranked if score(u) > 0]
        if not ranked_pos:
            ranked_pos = sorted(unique_urls, key=lambda u: u.count('/'))[:max_urls]
        # Ensure base is first
        if base_url in ranked_pos:
            ranked_pos.remove(base_url)
        ranked_pos.insert(0, base_url)
        return ranked_pos[:max_urls]

    # GitHub URL prioritization removed

    def _detect_version_lang(self, url: str) -> dict:
        """Detect version and language hints from URL path."""
        u = url.lower()
        version = None
        m = re.search(r"/v(\d+(?:\.\d+)*)/", u) or re.search(r"/(\d+\.\d+(?:\.\d+)*)/", u)
        if m:
            version = m.group(1)
        lang = None
        lm = re.search(r"/(en|zh|jp|ko|vi)/", u)
        if lm:
            lang = lm.group(1)
        return {"version": version, "lang": lang}

    async def _discover_with_tavily_map(self, base_url: str, max_depth: int = 3, max_breadth: int = 80, limit: int = 400) -> List[str]:
        """Discover documentation URLs using Tavily Map and normalize results."""
        if not self.tavily_client:
            return [base_url]
        try:
            resp = self.tavily_client.map(
                url=base_url,
                max_depth=max_depth,
                max_breadth=max_breadth,
                limit=limit,
                timeout=10000
            )
            candidates = resp.get("results", []) or resp.get("nodes", []) or []
            def _norm(u: str) -> str:
                u = u.split("#")[0]
                return re.sub(r"[?&](utm_[^=&]+|ref|source)=[^&]+", "", u)
            uniq = []
            seen = set()
            for x in candidates:
                u = x if isinstance(x, str) else x.get("url", "")
                if not u:
                    continue
                nu = _norm(u)
                if nu not in seen and urlparse(nu).netloc == urlparse(base_url).netloc:
                    seen.add(nu)
                    uniq.append(nu)
            if base_url not in seen:
                uniq.insert(0, base_url)
            return uniq
        except Exception as e:
            logger.warning(f"Tavily Map failed: {e}")
            return [base_url]

    def _discover_sitemap_urls(self, base_url: str, cap: int = 400) -> List[str]:
        """Discover documentation URLs using sitemap.xml when available."""
        try:
            from urllib.parse import urljoin
            import xml.etree.ElementTree as ET
            import requests  # type: ignore
        except Exception:
            return []
        try:
            root = f"{urlparse(base_url).scheme}://{urlparse(base_url).netloc}"
            sm_url = urljoin(root, "/sitemap.xml")
            r = requests.get(sm_url, timeout=15)
            r.raise_for_status()
            urls: List[str] = []
            tree = ET.fromstring(r.content)
            for loc in tree.iter("{*}loc"):
                u = (loc.text or "").strip()
                if u and urlparse(u).netloc == urlparse(root).netloc and "/docs/" in u:
                    urls.append(u.split("#")[0])
            # Dedup and cap
            return list(dict.fromkeys(urls))[:cap]
        except Exception:
            return []

    def _http_extract_batch(self, urls: List[str]) -> List[Dict[str, Any]]:
        """HTTP-based extraction fallback using trafilatura or BeautifulSoup."""
        out: List[Dict[str, Any]] = []
        try:
            import trafilatura  # type: ignore
        except Exception:
            trafilatura = None  # type: ignore
        try:
            import requests  # type: ignore
            from bs4 import BeautifulSoup  # type: ignore
        except Exception:
            return []
        for u in urls:
            text = ""
            try:
                # Avoid trafilatura for sites that commonly 403 (e.g., python.langchain.com)
                use_traf = trafilatura is not None and not urlparse(u).netloc.endswith("python.langchain.com")
                if use_traf:
                    fetched = trafilatura.fetch_url(u)
                    text = trafilatura.extract(fetched) or ""
                if not text:
                    headers = {
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
                        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
                        "Accept-Language": "en-US,en;q=0.9",
                        "Referer": f"{urlparse(u).scheme}://{urlparse(u).netloc}/",
                        "Cache-Control": "no-cache",
                    }
                    r = requests.get(u, timeout=20, headers=headers)
                    r.raise_for_status()
                    soup = BeautifulSoup(r.text, "lxml")
                    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
                        tag.decompose()
                    text = " ".join(soup.get_text("\n").split())
            except Exception:
                text = ""
            if text:
                out.append({"url": u, "title": u, "content": text})
        return out

    def _depth_for_url(self, url: str) -> str:
        """Heuristic to choose extraction depth per URL."""
        u = url.lower()
        return "advanced" if any(k in u for k in ["api", "reference", "class", "method", "function"]) else "basic"

    async def _extract_urls(self, urls: List[str], max_concurrent: int = 8) -> List[Dict[str, Any]]:
        """Extract a list of URLs using Tavily extract in batches with retry."""
        if not self.tavily_client or not urls:
            return []
        out: List[Dict[str, Any]] = []
        # Batch to reduce API calls and credits
        batch_size = int(os.getenv("TAVILY_EXTRACT_BATCH", "5"))
        batched = [urls[i:i + batch_size] for i in range(0, len(urls), batch_size)]
        sem = asyncio.Semaphore(max_concurrent)

        async def run(batch: List[str]):
            for i in range(5):
                try:
                    async with sem:
                        resp = await asyncio.to_thread(self.tavily_client.extract(extract_depth="advanced", timeout=120000), urls=batch)
                        results = resp.get("results", resp if isinstance(resp, list) else [])
                        for item in results or []:
                            u = item.get("url") or ""
                            if not u:
                                continue
                            title = item.get("title", "") or u
                            content = item.get("content", "") or ""
                            out.append({"url": u, "title": title, "content": content})
                        return
                except Exception as e:
                    if "exceeds your plan's set usage limit" in str(e).lower():
                        logger.warning("Tavily usage limit reached during extract; stopping further requests.")
                        return
                    logger.debug(f"Extract batch error (attempt {i+1}) for batch size {len(batch)}: {e}")
                    await asyncio.sleep((2 ** i) + 0.3)
            logger.warning("Failed to extract batch after retries")

        await asyncio.gather(*[run(b) for b in batched])
        return out

    def _ipynb_to_text(self, content: str) -> str:
        """Convert minimal .ipynb JSON to plain text by concatenating cell sources."""
        try:
            data = json.loads(content)
            cells = data.get("cells", [])
            parts: List[str] = []
            for c in cells:
                src = c.get("source", [])
                if isinstance(src, list):
                    parts.append("".join(src))
                elif isinstance(src, str):
                    parts.append(src)
            return "\n\n".join(parts)
        except Exception:
            return content

    def _classify_source_type(self, url: str) -> str:
        """Classify the source type by URL patterns for metadata and chunking choices."""
        u = url.lower()
        if any(x in u for x in ["/api", "/reference"]):
            return "api_ref"
        if any(x in u for x in ["tutorial", "guide", "getting-started"]):
            return "tutorial"
        if "example" in u:
            return "example"
        if "github.com" in u:
            if any(x in u for x in ["/blob/", "/raw.githubusercontent.com/"]):
                if u.endswith(".ipynb"):
                    return "notebook"
                return "code_file"
            return "readme"
        return "docs_page"

    def _prepare_content(self, url: str, content: str) -> str:
        st = self._classify_source_type(url)
        if st == "notebook":
            return self._ipynb_to_text(content)
        return content

    def _chunk_text(self, text: str, target_tokens: int = 600, overlap: int = 80) -> List[str]:
        """Lightweight chunking by paragraphs/blocks with approximate token sizing."""
        blocks = re.split(r"\n{2,}", text.strip())
        chunks: List[str] = []
        buf: List[str] = []
        size = 0
        for b in blocks:
            t = b.strip()
            if not t:
                continue
            tokens = max(1, len(t) // 4)
            if size + tokens > target_tokens and buf:
                chunks.append("\n\n".join(buf))
                buf = buf[-1:] if overlap > 0 else []
                size = sum(len(x) // 4 for x in buf)
            buf.append(t)
            size += tokens
        if buf:
            chunks.append("\n\n".join(buf))
        return chunks

    def _chunk_for_url(self, url: str, content: str) -> List[str]:
        st = self._classify_source_type(url)
        text = self._prepare_content(url, content)
        if st in ("code_file",):
            return self._chunk_text(text, target_tokens=400, overlap=60)
        return self._chunk_text(text, target_tokens=600, overlap=80)

    def _index_docs(self, framework_name: str, pages: List[Dict[str, Any]]):
        """Index pages into the per-framework vector store with metadata and dedup by hash."""
        vs_dir = self.vectordb_dir / framework_name.lower().replace(" ", "_")
        framework_id = framework_name.lower().replace(" ", "_")
        vs = VectorStore(persistent_dir=vs_dir, collection_name=f"framework_docs_{framework_id}")

        docs: List[Dict[str, Any]] = []
        ids: List[str] = []
        for p in pages:
            url = p["url"]
            title = p.get("title", "") or url
            content = p.get("content", "") or p.get("content_summary", "") or ""
            if not content:
                continue
            meta_vl = self._detect_version_lang(url)
            st = self._classify_source_type(url)
            for i, chunk in enumerate(self._chunk_for_url(url, content)):
                h = hashlib.sha256((url + str(i) + chunk).encode("utf-8")).hexdigest()
                metadata = {
                    "url": url,
                    "title": title,
                    "framework": framework_name,
                    "chunk_index": i,
                    "hash": h,
                    "source_type": st,
                    "version": meta_vl.get("version"),
                    "lang": meta_vl.get("lang"),
                }
                # Chroma metadata must be bool/int/float/str; drop None values
                metadata = {k: v for k, v in metadata.items() if v is not None}
                docs.append({
                    "text": chunk,
                    "metadata": metadata,
                })
                ids.append(h)
        if docs:
            try:
                vs.add_documents(docs, ids=ids)
            except Exception as e:
                logger.warning(f"Index add_documents error: {e}")

    # GitHub discovery removed

    # GitHub blob-to-raw conversion removed

    # GitHub batch extraction removed

    def _coverage_report(self, discovered: List[str], extracted_pages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute simple coverage metrics for discovered vs extracted URLs."""
        discovered_set = set([u for u in discovered if u])
        extracted_set = set([p.get("url") for p in extracted_pages if p.get("url")])
        return {
            "discovered_urls": len(discovered_set),
            "extracted_urls": len(extracted_set),
            "coverage": round(100.0 * len(extracted_set) / max(1, len(discovered_set)), 2),
        }
    
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
        ingested_at = datetime.utcnow().isoformat()
        # Add main page snippet if present
        if main_snippet:
            documents.append({
                "text": main_snippet,
                "metadata": {
                    "framework": framework_id,
                    "url": main_url,
                    "title": info.get("official_docs", {}).get("title", ""),
                    "source": "official_docs",
                    "canonical": True,
                    "source_domain": (urlparse(main_url).netloc if main_url else ""),
                    "doc_type": "page",
                    "ingested_at": ingested_at,
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
                    "canonical": True,
                    "source_domain": (urlparse(url).netloc if url else ""),
                    "doc_type": "page",
                    "ingested_at": ingested_at,
                }
            })

        # Attempt to extract code blocks directly from official pages to ensure canonical syntax
        try:
            code_docs: List[Dict[str, Any]] = []
            for p in pages[:8]:  # limit to first pages to keep ingestion fast
                url = p.get("url", "") or ""
                title = p.get("title", "") or ""
                if not url or not self._is_allowed_url(framework_name, url):
                    continue
                extracted = self._extract_code_blocks_from_url(url)
                for blk in extracted:
                    # Each blk: {"code": str, "language": str, optional "section_title", "section_anchor"}
                    code_text = (blk.get("code") or "").strip()
                    if not code_text:
                        continue
                    lang = (blk.get("language") or "").strip().lower() or None
                    section_title = (blk.get("section_title") or "").strip()
                    section_anchor = (blk.get("section_anchor") or "").strip()
                    source_domain = urlparse(url).netloc if url else ""
                    # Fingerprint for deduplication
                    try:
                        fingerprint = hashlib.sha1(code_text.encode("utf-8")).hexdigest()
                    except Exception:
                        fingerprint = code_text[:64]
                    # Simple relevance heuristic: contains framework name/import
                    fw_pat = re.compile(r"\b(?:import|from)\s+([a-zA-Z0-9_\.]+)")
                    imports = fw_pat.findall(code_text)
                    is_relevant = any(framework_id in imp.lower() for imp in imports) or (framework_id in code_text.lower())
                    priority = "official_code_high" if is_relevant else "official_code"
                    code_docs.append({
                        "text": code_text,
                        "metadata": {
                            "framework": framework_id,
                            "url": url,
                            "title": title,
                            "source": "official_docs",
                            "type": "code_block",
                            "language": lang,
                            "section_title": section_title,
                            "section_anchor": section_anchor,
                            "source_domain": source_domain,
                            "canonical": True,
                            "priority": priority,
                            "code_fingerprint": fingerprint,
                            "ingested_at": ingested_at,
                        }
                    })
            # De-duplicate short identical code blocks, keep up to 50 blocks
            if code_docs:
                seen: set = set()
                deduped: List[Dict[str, Any]] = []
                for d in code_docs:
                    key = (d["metadata"].get("url"), d["metadata"].get("code_fingerprint"))
                    if key in seen:
                        continue
                    seen.add(key)
                    deduped.append(d)
                documents.extend(deduped[:50])
        except Exception as _e:
            if self.is_logging:
                logger.warning(f"Code block extraction skipped: {_e}")

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
                            "source": "tutorial",
                            "canonical": False,
                            "source_domain": (urlparse(url).netloc if url else ""),
                            "doc_type": "tutorial",
                            "ingested_at": ingested_at,
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
                            "source": "api_reference",
                            "canonical": False,
                            "source_domain": (urlparse(url).netloc if url else ""),
                            "doc_type": "api_reference",
                            "ingested_at": ingested_at,
                        }
                    })
            
            if not documents:
                return {"framework": framework_id, "collection": collection_name, "added": 0}

        # Upsert into per-framework VectorStore. If persistent_dir is provided and already points
        # to the framework directory, use it as-is to avoid double-nesting.
        if isinstance(persistent_dir, (str, Path)) and persistent_dir:
            base_dir = Path(persistent_dir)
            if base_dir.name == framework_id:
                vs_dir_path = base_dir
            else:
                vs_dir_path = base_dir / framework_id
        else:
            base_dir = Path(self.vectordb_dir)
            vs_dir_path = base_dir / framework_id
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

    def _extract_code_blocks_from_url(self, url: str) -> List[Dict[str, str]]:
        """Fetch a URL and extract code blocks from HTML (<pre><code>) and fenced code.
        Returns a list of {"code": str, "language": Optional[str], "section_title": Optional[str], "section_anchor": Optional[str]}.
        Best-effort implementation without external dependencies.
        """
        blocks: List[Dict[str, str]] = []
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                raw = resp.read()
            # Try to decode
            try:
                html_text = raw.decode("utf-8", errors="ignore")
            except Exception:
                html_text = raw.decode(errors="ignore")

            # Helper: find nearest preceding heading within a sliding window
            def _nearest_heading(before_text: str) -> Dict[str, str]:
                window = before_text[-2000:] if len(before_text) > 2000 else before_text
                # Capture last h1/h2/h3
                heading_iter = list(re.finditer(r"<h([1-3])[^>]*>([\s\S]*?)</h\1>", window, re.IGNORECASE))
                if not heading_iter:
                    return {"section_title": "", "section_anchor": ""}
                last = heading_iter[-1]
                heading_html = last.group(0)
                # Extract text and optional id
                title_txt = re.sub(r"<[^>]+>", "", heading_html)
                id_match = re.search(r"id=\"([^\"]+)\"", heading_html)
                return {
                    "section_title": _html.unescape(title_txt).strip(),
                    "section_anchor": (id_match.group(1) if id_match else "").strip()
                }

            # HTML <pre><code class="language-xyz">...</code></pre> with positions
            pre_code_pattern = re.compile(r"<pre[^>]*>\s*<code([^>]*)>([\s\S]*?)</code>\s*</pre>", re.IGNORECASE)
            for m in pre_code_pattern.finditer(html_text):
                attrs = m.group(1) or ""
                inner = m.group(2) or ""
                code = _html.unescape(inner)
                # Strip HTML tags inside code if any lingering
                code = re.sub(r"<[^>]+>", "", code)
                lang = None
                cls_match = re.search(r"class=\"([^\"]+)\"", attrs)
                if cls_match:
                    classes = cls_match.group(1)
                    m2 = re.search(r"language-([a-z0-9_+-]+)", classes, re.IGNORECASE)
                    if m2:
                        lang = m2.group(1).lower()
                heading_meta = _nearest_heading(html_text[: m.start()])
                blocks.append({
                    "code": code.strip(),
                    "language": lang or "",
                    **heading_meta,
                })

            # Markdown-style fenced code blocks ```lang\n...\n``` with positions
            fence_pattern = re.compile(r"```([a-zA-Z0-9_+\-]*)\n([\s\S]*?)\n```", re.MULTILINE)
            for m in fence_pattern.finditer(html_text):
                lang = (m.group(1) or "").lower()
                body = m.group(2) or ""
                heading_meta = _nearest_heading(html_text[: m.start()])
                blocks.append({
                    "code": body.strip(),
                    "language": lang,
                    **heading_meta,
                })
        except Exception as e:
            if self.is_logging:
                logger.debug(f"Failed to fetch/extract code from {url}: {e}")
        # Filter obviously too short blocks
        out: List[Dict[str, str]] = []
        for b in blocks:
            code = (b.get("code") or "").strip()
            if len(code) < 12:
                continue
            out.append({
                "code": code,
                "language": (b.get("language") or "").strip(),
                "section_title": (b.get("section_title") or "").strip(),
                "section_anchor": (b.get("section_anchor") or "").strip(),
            })
        return out

    # GitHub README fallback removed

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
        # Upsert into per-framework VectorStore (separate folder per framework)
        # If persistent_dir is provided, treat it as the base directory; always append framework_id
        if isinstance(persistent_dir, (str, Path)) and persistent_dir:
            base_dir = Path(persistent_dir)
        else:
            base_dir = Path(self.vectordb_dir)
        vs_dir_path = base_dir / framework_id
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
        # Ingest into the computed per-framework directory without forcing refresh if cache exists
        return self.ingest_official_docs(
            framework_name=framework_name,
            project=project,
            location=location,
            persistent_dir=str(vs_dir_path),
            force_refresh=False,
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

# _extract_github_repo removed

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
