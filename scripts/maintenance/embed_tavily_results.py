import os
import re
import sys
import json
from pathlib import Path
from typing import Dict, List, Optional

# Add project root/src to import path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from GAAPF.core.tools.framework_collector import FrameworkCollector
from GAAPF.core.utils.credentials_helper import setup_google_credentials, get_vertex_ai_config
try:
    from langchain_google_vertexai import ChatVertexAI  # type: ignore
except Exception:  # pragma: no cover - optional dependency at runtime
    ChatVertexAI = None  # type: ignore


FILENAME_RE = re.compile(r"^tavily_extract_([a-zA-Z0-9_\-]+)_.*\\.json$")


def _infer_framework_from_filename(path_obj: Path) -> Optional[str]:
    match = FILENAME_RE.match(path_obj.name)
    if not match:
        return None
    return match.group(1).strip().lower().replace(" ", "_")


def _infer_framework_from_pages(pages: List[Dict], allowed: Dict[str, List[str]]) -> Optional[str]:
    # Fallback: infer framework by matching page URL domains against allowed_domains
    from urllib.parse import urlparse
    for page in pages or []:
        url = (page.get("url") or "").strip()
        if not url:
            continue
        netloc = urlparse(url).netloc.lower()
        for fw, domains in (allowed or {}).items():
            for domain in domains or []:
                if netloc.endswith(domain):
                    return fw
    return None


def _normalize_pages(raw_pages: List[Dict]) -> List[Dict]:
    normalized: List[Dict] = []
    for raw in raw_pages or []:
        url = (raw.get("url") or "").strip()
        if not url:
            continue
        title = (raw.get("title") or url).strip()
        # Prefer content; fallback to markdown/raw_content
        content = (
            (raw.get("content") or "").strip()
            or (raw.get("markdown") or "").strip()
            or (raw.get("raw_content") or "").strip()
        )
        summary = (raw.get("content_summary") or "").strip()
        if not content and not summary:
            # Skip empty page
            continue
        normalized.append({
            "url": url,
            "title": title,
            "content": content or summary,
            "content_summary": summary or (content[:800] + ("..." if len(content) > 800 else "")),
        })
    return normalized


def embed_from_results(results_dir: str = "results", only_frameworks: Optional[List[str]] = None) -> Dict:
    """Embed Tavily extract results into per-framework VectorStore collections.

    Args:
        results_dir: Directory containing tavily_extract_*.json files
        only_frameworks: Optional allowlist of framework names to process

    Returns:
        Summary dictionary of processing stats
    """
    setup_google_credentials()
    collector = FrameworkCollector(is_logging=True)

    base = Path(results_dir)
    if not base.exists():
        raise FileNotFoundError(f"Results directory not found: {base}")

    files = sorted([p for p in base.glob("tavily_extract_*.json") if p.is_file()])
    processed_files = 0
    framework_stats: Dict[str, Dict[str, int]] = {}

    only_fw_norm = {fw.strip().lower() for fw in (only_frameworks or [])}

    for path_obj in files:
        try:
            with open(path_obj, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"[skip] Failed to read {path_obj.name}: {e}")
            continue

        pages = _normalize_pages(data.get("pages") or [])
        if not pages:
            print(f"[skip] No pages with content in {path_obj.name}")
            continue

        framework = _infer_framework_from_filename(path_obj)
        if not framework:
            framework = _infer_framework_from_pages(pages, collector.allowed_domains)

        if not framework:
            print(f"[skip] Cannot infer framework for {path_obj.name}")
            continue

        if only_fw_norm and framework not in only_fw_norm:
            print(f"[skip] {path_obj.name}: framework {framework} not in filter {sorted(only_fw_norm)}")
            continue

        print(f"[embed] {path_obj.name} -> framework={framework}, pages={len(pages)}")
        collector._index_docs(framework, pages)

        processed_files += 1
        stats = framework_stats.setdefault(framework, {"files": 0, "pages": 0})
        stats["files"] += 1
        stats["pages"] += len(pages)

    return {"processed_files": processed_files, "frameworks": framework_stats}


# ---------------------------
# Curriculum generation (LLM)
# ---------------------------

def _init_vertex_llm():
    """Initialize ChatVertexAI using shared config.

    Returns
    -------
    ChatVertexAI
        Configured Vertex AI chat model instance
    """
    cfg = get_vertex_ai_config()
    if ChatVertexAI is None:
        raise RuntimeError("langchain-google-vertexai is not installed; cannot initialize Vertex AI LLM")
    return ChatVertexAI(
        model_name=cfg["model_name"],
        temperature=0.2,
        top_p=cfg["top_p"],
    )


def _synthesize_curriculum_from_discovered_urls(llm, framework: str, discovered_urls: List[str]) -> Dict:
    """Create a concise curriculum using discovered URLs as context (no sources in output)."""
    from datetime import datetime
    steps_hint = """
    You are a curriculum designer. Your job is take a list of URLs, you need to order them in a way that is most useful for learning the framework.
    Then create a curriculum with short titles and one-line descriptions per step. Avoid listing sources or links. Return markdown or bullets only.
    This is a example of a curriculum: 
    Here's the learning path I recommend:
    ## Step 1: Core Concepts (very short)

    - What is LangChain?

    - What is an Agent in LangChain?

    - What are Tools?

    - What is an LLMChain?

    ## Step 2: Build a Simple Agent

    -Let’s create a basic ReAct-style agent using OpenAI and Python functions as tools.

    ## Step 3: Add More Tools

    - Integrate search, math, or file-reading tools.

    -Learn how agents decide which tool to use.

    ## Step 4: Memory & Customization

    - Add conversation memory (like a chatbot that remembers).

    - Customize how the agent behaves or thinks.
    """
    # Build a simple chat-style payload compatible with ChatVertexAI
    user = "Now build a curriculum for the following discovered URLs:\n" + "\n".join(f"- {u}" for u in (discovered_urls or [])[:50])
    try:
        resp = llm.invoke([
            {"role": "system", "content": steps_hint},
            {"role": "user", "content": user},
        ])
        content = getattr(resp, "content", str(resp)) or ""
        # Sanitize: drop any 'Sources:' block if model still emitted it
        try:
            parts = content.split("\n\n")
            filtered = [p for p in parts if not p.strip().lower().startswith("sources:")]
            content = "\n\n".join(filtered).strip()
        except Exception:
            pass
    except Exception:
        content = (
            "Core modules:\n"
            "1) Basics\n2) Key components\n3) Minimal app\n4) Extensions\n5) Practice"
        )

    return {
        "framework": framework,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "discovered_urls_sample": (discovered_urls or [])[:50],
        "curriculum": content,
    }


def _write_curriculum(framework: str, payload: Dict) -> Path:
    """Persist curriculum payload to data/curriculums/<framework>.json"""
    out_dir = Path("data/curriculums")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{framework}.json"
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


def build_curriculums_from_results(results_dir: str = "results", only_frameworks: Optional[List[str]] = None) -> Dict:
    """Build per-framework curriculum files from Tavily result discovered_urls only.

    This function does NOT embed documents. It only reads tavily_extract_*.json,
    extracts discovered_urls, uses Vertex LLM to synthesize a concise curriculum,
    and writes to data/curriculums/<framework>.json.
    """
    setup_google_credentials()
    llm = _init_vertex_llm()

    base = Path(results_dir)
    if not base.exists():
        raise FileNotFoundError(f"Results directory not found: {base}")

    files = sorted([p for p in base.glob("tavily_extract_*.json") if p.is_file()])
    processed_files = 0
    framework_stats: Dict[str, Dict[str, int]] = {}

    only_fw_norm = {fw.strip().lower() for fw in (only_frameworks or [])}

    for path_obj in files:
        try:
            with open(path_obj, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"[skip] Failed to read {path_obj.name}: {e}")
            continue

        framework = _infer_framework_from_filename(path_obj)
        if not framework:
            # Attempt to infer from pages/domains as a fallback if needed
            pages = _normalize_pages(data.get("pages") or [])
            try:
                collector = FrameworkCollector(is_logging=False)
                framework = _infer_framework_from_pages(pages, collector.allowed_domains)
            except Exception:
                framework = None

        if not framework:
            print(f"[skip] Cannot infer framework for {path_obj.name}")
            continue

        if only_fw_norm and framework not in only_fw_norm:
            print(f"[skip] {path_obj.name}: framework {framework} not in filter {sorted(only_fw_norm)}")
            continue

        discovered_urls = data.get("discovered_urls") or []
        if not discovered_urls:
            print(f"[skip] No discovered_urls in {path_obj.name}")
            continue

        payload = _synthesize_curriculum_from_discovered_urls(llm, framework, discovered_urls)
        out_path = _write_curriculum(framework, payload)
        print(f"[curriculum] {path_obj.name} -> framework={framework}, urls={len(discovered_urls)} -> {out_path}")

        processed_files += 1
        stats = framework_stats.setdefault(framework, {"files": 0, "urls": 0})
        stats["files"] += 1
        stats["urls"] += len(discovered_urls)

    return {"processed_files": processed_files, "frameworks": framework_stats}


if __name__ == "__main__":
    results_dir = os.getenv("RESULTS_DIR", "results")
    only_fw_env = os.getenv("ONLY_FW", "")
    only_frameworks = [s.strip() for s in only_fw_env.split(",") if s.strip()]
    summary = build_curriculums_from_results(results_dir, only_frameworks=only_frameworks)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


