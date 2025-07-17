import logging
from typing import Dict, List, Optional, Union, Any
from pathlib import Path

from . import SpecializedAgent
from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.tools import BaseTool
from GAAPF.prompts.documentation_expert import generate_system_prompt

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentationExpertAgent(SpecializedAgent):
    """
    Specialized agent focused on documentation navigation and explanation.
    
    The DocumentationExpertAgent is responsible for:
    1. Finding and providing relevant documentation
    2. Explaining documentation content and structure
    3. Translating technical documentation into understandable explanations
    4. Guiding users through framework documentation resources
    """
    
    # Class attributes for agent registry support
    DESCRIPTION = "Expert in documentation navigation and explanation"
    CAPABILITIES = [
        "documentation_search",
        "content_explanation",
        "technical_translation",
        "resource_guidance",
        "reference_finding"
    ]
    PRIORITY = 6  # Medium-high priority for documentation tasks
    
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
        Initialize the DocumentationExpertAgent.
        
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
                "detail_level": "balanced",  # concise, balanced, comprehensive
                "include_examples": True,
                "include_related_docs": True,
                "simplify_technical_terms": True,
                "prioritize_official_docs": True
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
            agent_type="documentation_expert",
            description="Expert in documentation navigation and explanation",
            is_logging=is_logging,
            *args, **kwargs
        )
        
        if self.is_logging:
            logger.info(f"Initialized DocumentationExpertAgent with config: {self.config}")
    
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
        Enhance a user query with learning context specific to the documentation expert role.
        
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
        Process and structure the documentation expert's response.
        
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
        
        # Add documentation expert-specific metadata
        processed["documentation_references"] = self._extract_documentation_references(processed["content"])
        processed["documentation_type"] = self._determine_documentation_type(processed["content"])
        
        return processed
    
    def _extract_documentation_references(self, response_content: str) -> List[Dict]:
        """
        Extract documentation references from the response.
        
        Parameters:
        ----------
        response_content : str
            Content of the response
            
        Returns:
        -------
        List[Dict]
            List of extracted documentation references
        """
        # In a real implementation, this would parse markdown links and references
        # For now, we'll use a simple approach
        
        doc_references = []
        import re
        
        # Find markdown links
        link_pattern = r"\[([^\]]+)\]\(([^)]+)\)"
        link_matches = re.findall(link_pattern, response_content)
        
        for idx, (text, url) in enumerate(link_matches):
            doc_references.append({
                "id": idx,
                "text": text,
                "url": url,
                "type": "link"
            })
        
        return doc_references
    
    def _determine_documentation_type(self, response_content: str) -> str:
        """
        Determine the type of documentation provided in the response.
        
        Parameters:
        ----------
        response_content : str
            Content of the response
            
        Returns:
        -------
        str
            Type of documentation (api, guide, tutorial, reference, etc.)
        """
        # In a real implementation, this would use NLP to classify the content
        # For now, we'll use a simple keyword-based approach
        
        content_lower = response_content.lower()
        
        if "api" in content_lower and ("reference" in content_lower or "documentation" in content_lower):
            return "api_reference"
        elif "tutorial" in content_lower or "step by step" in content_lower:
            return "tutorial"
        elif "guide" in content_lower or "how to" in content_lower:
            return "guide"
        elif "example" in content_lower or "sample" in content_lower:
            return "example"
        else:
            return "general_documentation"