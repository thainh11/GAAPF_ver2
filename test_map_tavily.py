from tavily import TavilyClient
import os
import json
from urllib.parse import urlparse
from pathlib import Path
import time
from typing import List, Dict
import sys
import os as _os

# Ensure src/ is on sys.path to reuse existing collectors/utilities
CURRENT_DIR = _os.path.abspath(_os.path.dirname(__file__))
SRC_DIR = _os.path.join(CURRENT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
try:
    from GAAPF.core.tools.framework_collector import FrameworkCollector
except Exception:
    FrameworkCollector = None  # type: ignore

# Base URL to crawl
# BASE_URL = os.getenv("TEST_DOCS_BASE_URL", "https://docs.agno.com")
BASE_URL = "https://docs.agno.com"
# Get API key from environment to avoid hardcoding secrets
TAVILY_API_KEY = "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"
if not TAVILY_API_KEY:
    print("Missing TAVILY_API_KEY in environment. Aborting.")
    sys.exit(1)

# Initialize Tavily client
tavily_client = TavilyClient(api_key=TAVILY_API_KEY)

# Step 1: Discover documentation URLs
map_resp = tavily_client.map(
    url=BASE_URL,
    max_depth=3,
    max_breadth=50,
    limit=200,
    instructions="Find all CrewAI documentation pages including API reference, guides, tutorials",
    timeout=10000,
)

# Normalize discovered URLs (keep same domain, strip fragments, dedup)
root = urlparse(BASE_URL).netloc
raw_urls = map_resp.get("results", []) or map_resp.get("nodes", []) or []
urls: list[str] = []
for item in raw_urls:
    u = item if isinstance(item, str) else (item.get("url") or "")
    if not u:
        continue
    u = u.split("#")[0]
    if urlparse(u).netloc == root:
        urls.append(u)
discovered_urls = list(dict.fromkeys(urls))

# Step 2: Extract page contents in batches
extracted_pages: List[Dict] = []
batch_size = 10
for i in range(0, len(discovered_urls), batch_size):
    batch = discovered_urls[i:i + batch_size]
    try:
        # Prefer advanced depth for rich docs; increase timeout a bit
        er = tavily_client.extract(urls=batch, extract_depth="advanced", timeout=120000)
        items = er.get("results", er if isinstance(er, list) else [])
        for it in items or []:
            u = it.get("url") or ""
            title = it.get("title") or u
            # Try multiple possible fields depending on Tavily version
            content = (
                it.get("content")
                or it.get("markdown")
                or it.get("raw_content")
                or ""
            )
            if u:
                extracted_pages.append({
                    "url": u,
                    "title": title,
                    "content": content,
                })
    except Exception as e:
        print(f"Extract batch error: {e}")

# Step 2b: HTTP fallback for URLs with empty content
def http_extract_batch(urls: List[str]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    try:
        import requests  # type: ignore
        from bs4 import BeautifulSoup  # type: ignore
    except Exception:
        return out
    for u in urls:
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
                "Referer": f"{urlparse(u).scheme}://{urlparse(u).netloc}/",
                "Cache-Control": "no-cache",
            }
            r = requests.get(u, timeout=25, headers=headers)
            r.raise_for_status()
            soup = BeautifulSoup(r.text, "lxml")
            for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
                tag.decompose()
            text = " ".join(soup.get_text("\n").split())
            if text:
                out.append({"url": u, "title": u, "content": text})
        except Exception:
            pass
    return out

have_content = {p["url"] for p in extracted_pages if (p.get("content") or "").strip()}
missing_urls = [u for u in discovered_urls if u not in have_content]
if missing_urls:
    # limit fallback to keep it fast
    fallback_batch = missing_urls[:60]
    http_results = http_extract_batch(fallback_batch)
    url_to_page = {p["url"]: p for p in extracted_pages}
    for item in http_results:
        u = item["url"]
        if u in url_to_page and (url_to_page[u].get("content") or "").strip():
            continue
        extracted_pages.append({
            "url": u,
            "title": item.get("title") or u,
            "content": item.get("content") or "",
        })

# Step 3: Save to JSON file under results/
ts = time.strftime("%Y%m%d_%H%M%S")
out_dir = Path("results")
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / f"tavily_extract_crewai_{ts}.json"

output = {
    "base_url": BASE_URL,
    "discovered_count": len(discovered_urls),
    "extracted_count": len(extracted_pages),
    "discovered_urls": discovered_urls,
    "pages": extracted_pages,
}

with open(out_path, "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print({
    "saved": str(out_path),
    "discovered": len(discovered_urls),
    "extracted": len(extracted_pages)
})

# Optional: Extract code blocks for a subset of URLs and append to JSON (for learning guidance)
try:
    if FrameworkCollector is not None:
        collector = FrameworkCollector(is_logging=False)
        # Limit code extraction to avoid long runs
        max_code_urls = int(os.getenv("TEST_MAX_CODE_URLS", "40"))
        target_urls = discovered_urls[:max_code_urls]
        # Build index for quick lookup
        page_by_url = {p.get("url"): p for p in output.get("pages", [])}
        total_blocks = 0
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
        # Update output pages preserving order, then write a companion file
        updated_pages: List[Dict] = []
        seen = set()
        for p in output.get("pages", []):
            u = p.get("url")
            if u in page_by_url and u not in seen:
                updated_pages.append(page_by_url[u])
                seen.add(u)
        for u, p in page_by_url.items():
            if u not in seen:
                updated_pages.append(p)
        output["pages"] = updated_pages
        output["total_code_blocks"] = total_blocks
        code_path = out_dir / f"tavily_extract_crewai_{ts}_with_code.json"
        with open(code_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        print({"saved_with_code": str(code_path), "total_code_blocks": total_blocks})
except Exception as _e:
    print(f"Code block extraction step skipped: {_e}")