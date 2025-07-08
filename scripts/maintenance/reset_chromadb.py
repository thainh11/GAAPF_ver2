import chromadb
import shutil
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def reset_chromadb():
    """Reset ChromaDB to fix dimension mismatch issues"""
    
    # Default ChromaDB path used by LongTermMemory
    chroma_path = Path('memory/chroma_db')
    
    try:
        if chroma_path.exists():
            logger.info(f"Removing existing ChromaDB directory: {chroma_path}")
            shutil.rmtree(chroma_path)
            logger.info("ChromaDB directory removed successfully")
        else:
            logger.info("No existing ChromaDB directory found")
        
        # Create the directory structure
        chroma_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("ChromaDB reset complete. The collection will be recreated with correct dimensions on next run.")
        
    except Exception as e:
        logger.error(f"Error resetting ChromaDB: {e}")
        raise

if __name__ == "__main__":
    reset_chromadb() 