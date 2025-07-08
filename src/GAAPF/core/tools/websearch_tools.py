"""
Websearch tools for agents.
"""

import logging
import os
import json
from typing import Dict, List, Any

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from dotenv import load_dotenv
    from tavily import TavilyClient
    TAVILY_AVAILABLE = True
except ImportError:
    TAVILY_AVAILABLE = False
    logger.warning("Tavily search API not available. Using stub implementation.")

# Try to load environment variables
try:
    load_dotenv()
except:
    logger.warning("Failed to load environment variables from .env file")

def search_web(query: str, num_results: int = 5) -> Dict:
    """
    Search the web for information.
    
    Parameters:
    ----------
    query : str
        The search query
    num_results : int, optional
        Number of results to return
        
    Returns:
    -------
    Dict
        Search results
    """
    logger.info(f"Web search called with query: {query}")
    
    # Try to use Tavily if available
    if TAVILY_AVAILABLE:
        try:
            tavily_api_key = os.getenv("TAVILY_API_KEY")
            if tavily_api_key:
                client = TavilyClient(api_key=tavily_api_key)
                search_result = client.search(
                    query=query,
                    search_depth="basic",
                    max_results=num_results
                )
                return {
                    "results": search_result.get("results", []),
                    "source": "tavily"
                }
        except Exception as e:
            logger.error(f"Error using Tavily search: {e}")
    
    # Fallback to stub implementation
    return {
        "results": [
            {
                "title": "Stub search result",
                "snippet": "This is a stub search result. The actual search functionality is not implemented.",
                "url": "https://example.com"
            }
        ],
        "source": "stub"
    }

def deep_search(query: str, num_results: int = 3) -> Dict:
    """
    Perform a deeper web search with more comprehensive results.
    
    Parameters:
    ----------
    query : str
        The search query
    num_results : int, optional
        Number of results to return
        
    Returns:
    -------
    Dict
        Search results
    """
    logger.info(f"Deep search called with query: {query}")
    
    # Use standard search with more results as fallback
    return search_web(query, num_results=num_results)
