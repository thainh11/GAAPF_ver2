import logging
from typing import Dict, List, Optional, Union, Any
from pathlib import Path

from . import SpecializedAgent
from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.tools import BaseTool
from src.GAAPF.prompts.research_assistant import generate_system_prompt

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ResearchAssistantAgent(SpecializedAgent):
    """
    Specialized agent focused on finding additional learning resources.
    
    The ResearchAssistantAgent is responsible for:
    1. Finding relevant articles, tutorials, and documentation
    2. Discovering community resources and discussions
    3. Identifying learning materials at appropriate skill levels
    4. Curating resources to supplement the learning experience
    """
    
    def __init__(
        self,
        llm: BaseLanguageModel,
        tools: List[Union[str, BaseTool]] = [],
        memory_path: Optional[Path] = None,
        config: Dict = None,
        is_logging: bool = False,
        *args, **kwargs
    ):
        """
        Initialize the ResearchAssistantAgent.
        
        Parameters:
        ----------
        llm : BaseLanguageModel
            Language model to use for this agent
        tools : List[Union[str, BaseTool]], optional
            Tools available to this agent
        memory_path : Path, optional
            Path to agent memory file
        config : Dict, optional
            Agent-specific configuration
        is_logging : bool, optional
            Flag to enable detailed logging
        """
        # Set default config if not provided
        if config is None:
            config = {
                "resource_types": ["articles", "tutorials", "documentation", "videos", "community"],
                "prioritize_official": True,
                "include_community": True,
                "recency_importance": "high",  # low, medium, high
                "depth_vs_breadth": "balanced"  # depth, breadth, balanced
            }
        
        # Set default tools if not provided
        if not tools:
            tools = []
        
        # Initialize the base specialized agent
        super().__init__(
            llm=llm,
            tools=tools,
            memory_path=memory_path,
            config=config,
            agent_type="research_assistant",
            description="Expert in finding additional learning resources",
            is_logging=is_logging,
            *args, **kwargs
        )
        
        if self.is_logging:
            logger.info(f"Initialized ResearchAssistantAgent with config: {self.config}")
    
    def _generate_system_prompt(self, learning_context: Dict = None) -> str:
        """
        Generate a system prompt for this agent.
        
        Returns:
        -------
        str
            System prompt for the agent
        """
        return generate_system_prompt(self.config, learning_context)
    
    def _enhance_query_with_context(self, query: str, learning_context: Dict) -> str:
        """
        Enhance a user query with learning context specific to the research assistant role.
        
        Parameters:
        ----------
        query : str
            Original user query
        learning_context : Dict
            Current learning context
            
        Returns:
        -------
        str
            Enhanced query with context
        """
        return query
    
    def _process_response(self, response: Any, learning_context: Dict) -> Dict:
        """
        Process and structure the research assistant's response.
        
        Parameters:
        ----------
        response : Any
            Raw response from the agent
        learning_context : Dict
            Current learning context
            
        Returns:
        -------
        Dict
            Processed and structured response
        """
        # Get base processed response
        processed = super()._process_response(response, learning_context)
        
        # Add research assistant-specific metadata
        processed["resources"] = self._extract_resources(processed["content"])
        processed["resource_count"] = len(processed["resources"])
        processed["resource_types"] = self._categorize_resources(processed["resources"])
        
        return processed
    
    def _extract_resources(self, response_content: str) -> List[Dict]:
        """
        Extract resources from the response.
        
        Parameters:
        ----------
        response_content : str
            Content of the response
            
        Returns:
        -------
        List[Dict]
            List of extracted resources
        """
        # In a real implementation, this would use more sophisticated parsing
        # For now, we'll use a simple approach to identify resources
        
        resources = []
        import re
        
        # Find markdown links
        link_pattern = r"\[([^\]]+)\]\(([^)]+)\)"
        link_matches = re.findall(link_pattern, response_content)
        
        for idx, (text, url) in enumerate(link_matches):
            resources.append({
                "id": idx,
                "title": text,
                "url": url,
                "description": self._find_description_for_link(response_content, text, url)
            })
        
        # Find numbered or bulleted resources
        resource_pattern = r"(?:^|\n)[\d*-][\d*\.\s-]*([^:\n]+):\s*([^\n]+)(?:\n|$)"
        resource_matches = re.findall(resource_pattern, response_content)
        
        for idx, (title, description) in enumerate(resource_matches):
            # Skip if already found as a link
            if any(r["title"] == title.strip() for r in resources):
                continue
                
            resources.append({
                "id": len(resources),
                "title": title.strip(),
                "url": self._extract_url_from_text(description),
                "description": description.strip()
            })
        
        return resources
    
    def _find_description_for_link(self, content: str, link_text: str, link_url: str) -> str:
        """
        Find a description for a link in the content.
        
        Parameters:
        ----------
        content : str
            Full content text
        link_text : str
            Text of the link
        link_url : str
            URL of the link
            
        Returns:
        -------
        str
            Description for the link
        """
        # Try to find a description after the link
        import re
        
        # Escape special characters in link_text for regex
        escaped_text = re.escape(link_text)
        
        # Look for description after the link
        pattern = f"\\[{escaped_text}\\]\\([^)]+\\)[^\\n.]*[\\n.]+([^\\n]+)"
        match = re.search(pattern, content)
        
        if match:
            return match.group(1).strip()
        
        # If no description found, return empty string
        return ""
    
    def _extract_url_from_text(self, text: str) -> str:
        """
        Extract a URL from text if present.
        
        Parameters:
        ----------
        text : str
            Text to extract URL from
            
        Returns:
        -------
        str
            Extracted URL or empty string
        """
        import re
        
        # Find URL in text
        url_pattern = r"https?://[^\s)>]+"
        match = re.search(url_pattern, text)
        
        if match:
            return match.group(0)
        
        return ""
    
    def _categorize_resources(self, resources: List[Dict]) -> Dict[str, int]:
        """
        Categorize resources by type.
        
        Parameters:
        ----------
        resources : List[Dict]
            List of resources
            
        Returns:
        -------
        Dict[str, int]
            Count of resources by type
        """
        categories = {
            "article": 0,
            "tutorial": 0,
            "documentation": 0,
            "video": 0,
            "community": 0,
            "other": 0
        }
        
        for resource in resources:
            title = resource.get("title", "").lower()
            url = resource.get("url", "").lower()
            description = resource.get("description", "").lower()
            
            # Categorize based on title, URL, and description
            if "tutorial" in title or "tutorial" in description or "how to" in title:
                categories["tutorial"] += 1
            elif "doc" in title or "documentation" in description or "docs" in url:
                categories["documentation"] += 1
            elif "video" in title or "youtube" in url or "watch" in url:
                categories["video"] += 1
            elif "forum" in url or "community" in url or "reddit" in url or "stack" in url:
                categories["community"] += 1
            elif "article" in title or "blog" in url:
                categories["article"] += 1
            else:
                categories["other"] += 1
        
        return categories 