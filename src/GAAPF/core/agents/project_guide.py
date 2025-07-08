import logging
from typing import Dict, List, Optional, Union, Any
from pathlib import Path

from . import SpecializedAgent
from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.tools import BaseTool
from src.GAAPF.prompts.project_guide import generate_system_prompt

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProjectGuideAgent(SpecializedAgent):
    """
    Specialized agent focused on project-based learning.
    
    The ProjectGuideAgent is responsible for:
    1. Designing practical projects to apply framework concepts
    2. Breaking down projects into manageable steps
    3. Providing guidance during project implementation
    4. Suggesting enhancements and extensions to projects
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
        Initialize the ProjectGuideAgent.
        
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
                "project_complexity": "adaptive",  # simple, moderate, complex, adaptive
                "guidance_level": "balanced",  # minimal, balanced, comprehensive
                "real_world_focus": True,
                "include_best_practices": True,
                "suggest_extensions": True
            }
        
        # Set default tools if not provided
        if not tools:
            tools = [
                "websearch_tools",
                "computer_tools",
                "terminal_tools"
            ]
        
        # Initialize the base specialized agent
        super().__init__(
            llm=llm,
            tools=tools,
            memory_path=memory_path,
            config=config,
            agent_type="project_guide",
            description="Expert in project-based learning",
            is_logging=is_logging,
            *args, **kwargs
        )
        
        if self.is_logging:
            logger.info(f"Initialized ProjectGuideAgent with config: {self.config}")
    
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
        Enhance a user query with learning context specific to the project guide role.
        
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
        Process and structure the project guide's response.
        
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
        
        # Add project guide-specific metadata
        processed["project_type"] = self._determine_project_type(processed["content"])
        processed["project_steps"] = self._extract_project_steps(processed["content"])
        processed["complexity_level"] = self._determine_complexity_level(processed["content"])
        
        return processed
    
    def _determine_project_type(self, response_content: str) -> str:
        """
        Determine the type of project described in the response.
        
        Parameters:
        ----------
        response_content : str
            Content of the response
            
        Returns:
        -------
        str
            Type of project
        """
        # In a real implementation, this would use NLP to classify the content
        # For now, we'll use a simple keyword-based approach
        
        content_lower = response_content.lower()
        
        if "web" in content_lower or "website" in content_lower or "frontend" in content_lower:
            return "web_application"
        elif "api" in content_lower or "backend" in content_lower or "server" in content_lower:
            return "backend_service"
        elif "mobile" in content_lower or "app" in content_lower:
            return "mobile_application"
        elif "data" in content_lower or "analysis" in content_lower or "visualization" in content_lower:
            return "data_project"
        elif "game" in content_lower:
            return "game_development"
        else:
            return "general_project"
    
    def _extract_project_steps(self, response_content: str) -> List[Dict]:
        """
        Extract project implementation steps from the response.
        
        Parameters:
        ----------
        response_content : str
            Content of the response
            
        Returns:
        -------
        List[Dict]
            List of project steps
        """
        # In a real implementation, this would use more sophisticated parsing
        # For now, we'll use a simple approach to identify numbered steps
        
        steps = []
        import re
        
        # Find numbered steps (1. Step description)
        step_pattern = r"(?:^|\n)(?:Step\s*)?(\d+)[\.:\)]\s*([^\n]+)"
        matches = re.findall(step_pattern, response_content, re.MULTILINE)
        
        for step_num, step_text in matches:
            steps.append({
                "number": int(step_num),
                "description": step_text.strip(),
                "details": self._find_step_details(response_content, step_num, step_text)
            })
        
        # Sort steps by number
        steps.sort(key=lambda x: x["number"])
        
        return steps
    
    def _find_step_details(self, content: str, step_num: str, step_text: str) -> str:
        """
        Find detailed description for a project step.
        
        Parameters:
        ----------
        content : str
            Full content text
        step_num : str
            Step number
        step_text : str
            Step brief description
            
        Returns:
        -------
        str
            Detailed description for the step
        """
        import re
        
        # Escape special characters in step_text for regex
        escaped_text = re.escape(step_text.strip())
        
        # Try to find details after the step heading
        next_step_pattern = r"(?:Step\s*)?(?:\d+)[\.:\)]"
        
        # Find the position of this step
        step_match = re.search(f"(?:Step\\s*)?{step_num}[\\.:)]\\s*{escaped_text}", content)
        if not step_match:
            return ""
        
        start_pos = step_match.end()
        
        # Find the position of the next step
        next_step_match = re.search(next_step_pattern, content[start_pos:])
        
        if next_step_match:
            end_pos = start_pos + next_step_match.start()
            return content[start_pos:end_pos].strip()
        else:
            # If no next step, get text until the next major section or end
            section_pattern = r"\n\s*(?:[A-Z][a-z]+\s*:|\d+\.|\#|\*\*)"
            section_match = re.search(section_pattern, content[start_pos:])
            
            if section_match:
                end_pos = start_pos + section_match.start()
                return content[start_pos:end_pos].strip()
            else:
                # If no section found, return the rest of the content
                return content[start_pos:].strip()
    
    def _determine_complexity_level(self, response_content: str) -> str:
        """
        Determine the complexity level of the project.
        
        Parameters:
        ----------
        response_content : str
            Content of the response
            
        Returns:
        -------
        str
            Complexity level (beginner, intermediate, advanced)
        """
        # In a real implementation, this would use more sophisticated analysis
        # For now, we'll use a simple keyword-based approach
        
        content_lower = response_content.lower()
        
        # Check for explicit complexity indicators
        if "advanced" in content_lower or "complex" in content_lower or "challenging" in content_lower:
            return "advanced"
        elif "intermediate" in content_lower or "moderate" in content_lower:
            return "intermediate"
        elif "beginner" in content_lower or "simple" in content_lower or "basic" in content_lower:
            return "beginner"
        
        # If no explicit indicators, analyze based on number of steps and keywords
        steps = self._extract_project_steps(response_content)
        
        if len(steps) > 10:
            return "advanced"
        elif len(steps) > 5:
            return "intermediate"
        else:
            return "beginner" 