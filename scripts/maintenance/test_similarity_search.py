import sys
from pathlib import Path

# Ensure src/ is on path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from GAAPF.core.tools.vector_store import VectorStore


def main():
    base_dir = PROJECT_ROOT / "data" / "frameworks" / "vectordb"
    if not base_dir.exists():
        print(f"Base vectordb directory not found: {base_dir}")
        return

    frameworks = [p.name for p in base_dir.iterdir() if p.is_dir()]
    if not frameworks:
        print("No framework directories found under vectordb")
        return

    print(f"Found frameworks: {', '.join(frameworks)}")

    for fw in frameworks:
        fw_dir = base_dir / fw
        collection_name = f"framework_docs_{fw}"
        print("\n" + "=" * (12 + len(fw)))
        print(f"Testing: {fw}")
        print(f"Path: {fw_dir}")
        print(f"Collection: {collection_name}")
        try:
            vs = VectorStore(persistent_dir=str(fw_dir), collection_name=collection_name)
            cnt = vs.count()
            print(f"Count: {cnt}")
            if cnt > 0:
                q = f"What is {fw}?"
                results = vs.similarity_search(q, k=1)
                if results:
                    text = results[0].get("text", "")
                    print(f"Top-1 snippet: {text[:160]}{'...' if len(text) > 160 else ''}")
                else:
                    print("No results returned by similarity_search")
            else:
                print("Collection is empty")
        except Exception as e:
            print(f"Error testing {fw}: {e}")


if __name__ == "__main__":
    main()


