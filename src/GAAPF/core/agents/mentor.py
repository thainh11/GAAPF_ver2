import logging
from typing import Dict, List, Optional, Union, Any
from pathlib import Path

from . import SpecializedAgent
from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.tools import BaseTool
from src.GAAPF.prompts.mentor import generate_system_prompt

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MentorAgent(SpecializedAgent):
    """
    Specialized agent focused on providing learning guidance and support.
    
    The MentorAgent is responsible for:
    1. Offering personalized learning advice and strategies
    2. Providing encouragement and motivation
    3. Helping users overcome learning challenges
    4. Suggesting learning paths and next steps
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
        Initialize the MentorAgent.
        
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
                "mentoring_style": "supportive",  # supportive, challenging, balanced
                "personalization_level": "high",  # low, medium, high
                "focus_on_strengths": True,
                "provide_motivation": True,
                "suggest_resources": True
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
            agent_type="mentor",
            description="Expert in providing learning guidance and support",
            is_logging=is_logging,
            *args, **kwargs
        )
        
        if self.is_logging:
            logger.info(f"Initialized MentorAgent with config: {self.config}")
    
    def _generate_system_prompt(self, learning_context: Dict = None) -> str:
        """
        Generate a dynamic system prompt for the Mentor agent.
        """
        return generate_system_prompt(self.config, learning_context)
    
    def _enhance_query_with_context(self, query: str, learning_context: Dict) -> str:
        """
        Enhance a user query with learning context specific to the mentor role.
        """
        return query
    
    def _process_response(self, response: Any, learning_context: Dict) -> Dict:
        """
        Process and structure the mentor's response.
        
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
        
        # Add mentor-specific metadata
        processed["guidance_type"] = self._determine_guidance_type(processed["content"])
        processed["suggested_resources"] = self._extract_suggested_resources(processed["content"])
        processed["motivational_content"] = self._contains_motivational_content(processed["content"])
        
        return processed
    
    def _determine_guidance_type(self, response_content: str) -> str:
        """
        Determine the type of guidance provided in the response.
        
        Parameters:
        ----------
        response_content : str
            Content of the response
            
        Returns:
        -------
        str
            Type of guidance
        """
        # In a real implementation, this would use NLP to classify the content
        # For now, we'll use a simple keyword-based approach
        
        content_lower = response_content.lower()
        
        if "next steps" in content_lower or "learning path" in content_lower:
            return "learning_path"
        elif "strategy" in content_lower or "approach" in content_lower:
            return "learning_strategy"
        elif "challenge" in content_lower or "overcome" in content_lower:
            return "challenge_support"
        elif "resource" in content_lower or "material" in content_lower:
            return "resource_suggestion"
        else:
            return "general_guidance"
    
    def _extract_suggested_resources(self, response_content: str) -> List[Dict]:
        """
        Extract suggested resources from the response.
        
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
                "type": "link"
            })
        
        return resources
    
    def _contains_motivational_content(self, response_content: str) -> bool:
        """
        Check if the response contains motivational content.
        
        Parameters:
        ----------
        response_content : str
            Content of the response
            
        Returns:
        -------
        bool
            True if motivational content is detected
        """
        motivational_keywords = [
            "encourage", "motivate", "progress", "achievement", "success",
            "confidence", "believe", "capable", "improve", "growth"
        ]
        
        content_lower = response_content.lower()
        return any(keyword in content_lower for keyword in motivational_keywords) 