
import shutil
from pathlib import Path
import os

def clean_chromadb():
    """
    Deletes the ChromaDB directory to ensure a clean start.
    """
    db_path = Path(__file__).parent.parent.parent / "data" / "framework_cache" / "chroma_db"
    
    if db_path.exists() and db_path.is_dir():
        try:
            shutil.rmtree(db_path)
            print(f"Successfully deleted directory: {db_path}")
        except OSError as e:
            print(f"Error deleting directory {db_path}: {e.strerror}")
    else:
        print(f"Directory {db_path} does not exist, no need to clean.")

if __name__ == "__main__":
    clean_chromadb() 