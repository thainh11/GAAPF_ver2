"""Google Cloud credentials helper module.

This module provides standardized credential handling for Google Cloud services
across the GAAPF system, ensuring consistent authentication patterns.
"""

import os
import json
import logging
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

def setup_google_credentials(project_root: Optional[Path] = None) -> Tuple[str, str]:
    """
    Set up Google Cloud credentials with standardized fallback logic.
    
    This function implements the standard credential resolution pattern:
    1. Check GOOGLE_APPLICATION_CREDENTIALS environment variable
    2. Fallback to config/google-credentials.json if it exists
    3. Extract project_id from credentials if GOOGLE_CLOUD_PROJECT is not set
    
    Args:
        project_root: Root directory of the project. If None, will attempt to detect.
        
    Returns:
        Tuple of (project_id, location) with resolved values
        
    Raises:
        None - This function is designed to be fault-tolerant
    """
    # Determine project root if not provided
    if project_root is None:
        # Try to find project root by looking for common markers
        current_path = Path(__file__).parent
        while current_path.parent != current_path:
            if (current_path / "config").exists() or (current_path / "run_cli.py").exists():
                project_root = current_path
                break
            current_path = current_path.parent
        else:
            # Fallback to a reasonable default
            project_root = Path(__file__).parent.parent.parent.parent
    
    # Step 1: Check if GOOGLE_APPLICATION_CREDENTIALS is already set
    creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    
    # Step 2: Fallback to local credentials file if env var not set
    if not creds_path:
        fallback_creds = project_root / "config" / "google-credentials.json"
        if fallback_creds.exists():
            creds_path = str(fallback_creds)
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = creds_path
            logger.info(f"Using fallback credentials: {fallback_creds}")
    
    # Step 3: Resolve project_id from environment or credentials file
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
    
    if not project_id and creds_path and Path(creds_path).exists():
        try:
            with open(creds_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                extracted_project = data.get("project_id")
                if extracted_project:
                    project_id = extracted_project
                    os.environ["GOOGLE_CLOUD_PROJECT"] = project_id
                    logger.info(f"Extracted project_id from credentials: {project_id}")
        except Exception as e:
            logger.warning(f"Failed to extract project_id from credentials file: {e}")
    
    # Step 4: Final fallback for development (should be configurable)
    if not project_id:
        project_id = os.getenv("GAAPF_DEFAULT_PROJECT", "gen-lang-client-0305686287")
        logger.warning(f"Using fallback project_id: {project_id}")
    
    # Step 5: Resolve location
    location = os.getenv("VERTEX_AI_LOCATION", "us-central1")
    
    return project_id, location

def get_vertex_ai_config() -> dict:
    """
    Get standardized Vertex AI configuration with environment variable overrides.
    
    Returns:
        Dictionary containing Vertex AI configuration parameters
    """
    project_id, location = setup_google_credentials()
    
    return {
        "project": project_id,
        "location": location,
        "model_name": os.getenv("VERTEX_AI_MODEL", "gemini-2.5-flash"),
        "temperature": float(os.getenv("VERTEX_AI_TEMPERATURE", "0.3")),
        "top_p": float(os.getenv("VERTEX_AI_TOP_P", "0.95")),
    }

def get_vertex_embedding_config() -> dict:
    """
    Get standardized Vertex AI embedding configuration.
    
    Returns:
        Dictionary containing Vertex AI embedding configuration parameters
    """
    project_id, location = setup_google_credentials()
    
    return {
        "project": project_id,
        "location": location,
        "model_name": os.getenv("VERTEX_EMBEDDING_MODEL", "gemini-embedding-001"),
    }