"""GAAPF Core Tools Module"""

from .framework_collector import FrameworkCollector
from .deepsearch import DeepSearch
from .trending_news import TrendingTopics

# Import functions from websearch_tools
from .websearch_tools import search_web, deep_search

__all__ = [
    'FrameworkCollector',
    'DeepSearch',
    'TrendingTopics',
    'search_web',
    'deep_search'
]