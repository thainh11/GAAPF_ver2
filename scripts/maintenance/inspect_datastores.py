#!/usr/bin/env python3
"""
Inspector utility for GAAPF data stores (JSON memories and ChromaDB vector stores).

It prints:
- Short-term memory JSONs in templates (memory.json, memory_*.json, memory_*_lt.json)
- ChromaDB collections in data/frameworks/vectordb (root and per-framework)
- ChromaDB collections in memory/chroma_db (root and per-framework)

Usage (PowerShell on Windows):
  conda activate ver5
  python scripts/maintenance/inspect_datastores.py
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, Any, List

def print_header(title: str) -> None:
    line = "=" * 80
    print(f"\n{line}\n{title}\n{line}")

def safe_read_json(path: Path) -> Any:
    try:
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        return {"error": str(e)}

def list_memory_jsons(base: Path) -> List[Path]:
    files: List[Path] = []
    for pattern in [
        "memory.json",
        "memory_*.json",
        "memory_*_lt.json",
    ]:
        files.extend(sorted((base / "templates").glob(pattern)))
    # De-duplicate
    out = []
    seen = set()
    for p in files:
        if p.resolve() not in seen:
            seen.add(p.resolve())
            out.append(p)
    return out

def inspect_memory_jsons(project_root: Path) -> None:
    print_header("Short-term and framework LT memory JSON files (templates)")
    files = list_memory_jsons(project_root)
    if not files:
        print("No memory files found in templates/")
        return
    for p in files:
        data = safe_read_json(p)
        print(f"\n- File: {p}")
        if data is None:
            print("  Status: (missing)")
            continue
        if isinstance(data, dict) and "error" in data:
            print(f"  Error reading file: {data['error']}")
            continue
        if isinstance(data, dict):
            keys = list(data.keys())
            print(f"  Type: dict with {len(keys)} top-level key(s) -> {keys[:10]}")
            # Show brief sample for first user_id
            if keys:
                uid = keys[0]
                try:
                    val = data.get(uid)
                    if isinstance(val, list):
                        print(f"  Sample user_id='{uid}' entries: {len(val)} (showing up to 3)")
                        for i, item in enumerate(val[:3], 1):
                            preview = str(item)[:200].replace("\n", " ")
                            print(f"[{i}] {preview}")
                    else:
                        print(f"  Sample user_id='{uid}' value type: {type(val).__name__}")
                except Exception as e:
                    print(f"  Error sampling user_id='{uid}': {e}")
        else:
            print(f"  Type: {type(data).__name__}")

def safe_import_chromadb():
    try:
        import chromadb  # type: ignore
        return chromadb
    except Exception as e:
        print(f"chromadb import failed: {e}")
        return None

def list_chroma_collections(client) -> List[Any]:
    try:
        return client.list_collections()
    except Exception as e:
        print(f"  list_collections error: {e}")
        return []

def print_collection_samples(collection, n: int = 3) -> None:
    try:
        cnt = collection.count()
        print(f"  Count: {cnt}")
        if cnt == 0:
            return
        m = min(n, cnt)
        res = collection.get(
            include=["documents", "metadatas"],
            limit=m,
        )
        ids = res.get("ids", [])
        docs = res.get("documents", [])
        metas = res.get("metadatas", [])
        for i in range(min(len(ids), m)):
            doc = (docs[i] if i < len(docs) else "") or ""
            meta = metas[i] if i < len(metas) else {}
            doc_str = str(doc)
            doc_preview = doc_str[:200].replace("\n", " ")
            print(f"    [{i+1}] {doc_preview}")
            print(f"        meta={json.dumps(meta, ensure_ascii=False)[:200]}")
    except Exception as e:
        print(f"  Error sampling collection: {e}")

def inspect_chromadb_root(root: Path, title: str) -> None:
    chromadb = safe_import_chromadb()
    if not chromadb:
        return
    print_header(title)
    if not root.exists():
        print(f"Path missing: {root}")
        return
    try:
        client = chromadb.PersistentClient(path=str(root))
    except Exception as e:
        print(f"Failed to open Chroma client at {root}: {e}")
        return
    colls = list_chroma_collections(client)
    if not colls:
        print("No collections found")
        return
    for c in colls:
        try:
            print(f"- Collection: {c.name}")
            print_collection_samples(c)
        except Exception as e:
            print(f"- Collection: (error) {e}")

def inspect_chromadb_per_framework(base: Path, frameworks: List[str], collection_prefix: str, title: str) -> None:
    chromadb = safe_import_chromadb()
    if not chromadb:
        return
    print_header(title)
    for fw in frameworks:
        fw_path = base / fw
        if not fw_path.exists():
            print(f"- {fw}: path missing {fw_path}")
            continue
        try:
            client = chromadb.PersistentClient(path=str(fw_path))
            coll_name = f"{collection_prefix}{fw}"
            coll = client.get_or_create_collection(name=coll_name)
            print(f"- {fw}: {coll_name}")
            print_collection_samples(coll)
        except Exception as e:
            print(f"- {fw}: error {e}")

def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    os.chdir(project_root)
    print(f"Project root: {project_root}")

    # 1) JSON memories
    inspect_memory_jsons(project_root)

    # 2) Vector DB: data/frameworks/vectordb (root collections)
    inspect_chromadb_root(project_root / "data" / "frameworks" / "vectordb", "ChromaDB at data/frameworks/vectordb (root)")

    # 3) Vector DB per-framework (framework_docs_<fw>)
    frameworks = ["langchain", "langgraph", "crewai", "autogen", "haystack"]
    inspect_chromadb_per_framework(
        project_root / "data" / "frameworks" / "vectordb",
        frameworks,
        collection_prefix="framework_docs_",
        title="Per-framework ChromaDB (data/frameworks/vectordb/<fw>)"
    )

    # 4) Vector DB: memory/chroma_db (root collections)
    inspect_chromadb_root(project_root / "memory" / "chroma_db", "ChromaDB at memory/chroma_db (root)")

    # 5) Vector DB per-framework LT used by SimpleLearningHub.set_framework()
    inspect_chromadb_per_framework(
        project_root / "memory" / "chroma_db",
        frameworks,
        collection_prefix="lt_",
        title="Per-framework LT ChromaDB (memory/chroma_db/<fw>)"
    )

if __name__ == "__main__":
    main()


