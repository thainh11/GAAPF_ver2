import logging
from typing import Dict, List, Optional, Union, Any
from pathlib import Path

from . import SpecializedAgent
from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.tools import BaseTool
from src.GAAPF.prompts.knowledge_synthesizer import generate_system_prompt

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KnowledgeSynthesizerAgent(SpecializedAgent):
    """
    Specialized agent focused on concept integration and knowledge synthesis.
    
    The KnowledgeSynthesizerAgent is responsible for:
    1. Connecting related concepts across different modules
    2. Integrating knowledge into a cohesive mental model
    3. Summarizing complex information into digestible insights
    4. Identifying patterns and relationships between concepts
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
        Initialize the KnowledgeSynthesizerAgent.
        
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
                "synthesis_depth": "comprehensive",  # basic, moderate, comprehensive
                "visual_aids": True,
                "use_analogies": True,
                "cross_module_connections": True,
                "summarization_style": "conceptual"  # factual, conceptual, applied
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
            agent_type="knowledge_synthesizer",
            description="Expert in concept integration and knowledge synthesis",
            is_logging=is_logging,
            *args, **kwargs
        )
        
        if self.is_logging:
            logger.info(f"Initialized KnowledgeSynthesizerAgent with config: {self.config}")
    
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
        Enhance a user query with learning context specific to the knowledge synthesizer role.
        
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
        Process and structure the knowledge synthesizer's response.
        
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
        
        # Add knowledge synthesizer-specific metadata
        processed["synthesis_type"] = self._determine_synthesis_type(processed["content"])
        processed["concepts_connected"] = self._extract_concepts_connected(processed["content"], learning_context)
        processed["has_visual_aid"] = self._contains_visual_aid(processed["content"])
        
        return processed
    
    def _determine_synthesis_type(self, response_content: str) -> str:
        """
        Determine the type of synthesis provided in the response.
        
        Parameters:
        ----------
        response_content : str
            Content of the response
            
        Returns:
        -------
        str
            Type of synthesis
        """
        # In a real implementation, this would use NLP to classify the content
        # For now, we'll use a simple keyword-based approach
        
        content_lower = response_content.lower()
        
        if "compare" in content_lower or "contrast" in content_lower or "versus" in content_lower:
            return "comparative_synthesis"
        elif "integrate" in content_lower or "combine" in content_lower or "together" in content_lower:
            return "integrative_synthesis"
        elif "build" in content_lower or "foundation" in content_lower or "progression" in content_lower:
            return "progressive_synthesis"
        elif "pattern" in content_lower or "common" in content_lower or "recurring" in content_lower:
            return "pattern_recognition"
        elif "framework" in content_lower or "model" in content_lower or "structure" in content_lower:
            return "framework_synthesis"
        else:
            return "general_synthesis"
    
    def _extract_concepts_connected(self, response_content: str, learning_context: Dict) -> List[str]:
        """
        Extract concepts that are connected in the response.
        
        Parameters:
        ----------
        response_content : str
            Content of the response
        learning_context : Dict
            Current learning context
            
        Returns:
        -------
        List[str]
            List of concepts that are connected in the response
        """
        # In a real implementation, this would use NLP to extract concepts
        # For now, we'll use a simple approach based on known concepts
        
        # Get all concepts from framework config
        framework_config = learning_context.get("framework_config", {})
        modules = framework_config.get("modules", {})
        
        all_concepts = []
        for module_info in modules.values():
            all_concepts.extend(module_info.get("concepts", []))
        
        # Check which concepts are mentioned in the response
        mentioned_concepts = []
        content_lower = response_content.lower()
        
        for concept in all_concepts:
            if concept.lower() in content_lower:
                mentioned_concepts.append(concept)
        
        return mentioned_concepts
    
    def _contains_visual_aid(self, response_content: str) -> bool:
        """
        Check if the response contains a visual aid.
        
        Parameters:
        ----------
        response_content : str
            Content of the response
            
        Returns:
        -------
        bool
            True if a visual aid is detected
        """
        # Check for common indicators of visual aids
        visual_indicators = [
            "```mermaid",
            "```graphviz",
            "```dot",
            "```plantuml",
            "| --- |",  # Markdown table indicator
            "+-----------+",  # ASCII table indicator
            "diagram",
            "flowchart",
            "graph",
            "chart"
        ]
        
        return any(indicator in response_content for indicator in visual_indicators) 