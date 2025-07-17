import logging
from typing import Dict, List, Optional, Union, Any
from pathlib import Path

from . import SpecializedAgent
from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.tools import BaseTool
from GAAPF.prompts.troubleshooter import generate_system_prompt

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TroubleshooterAgent(SpecializedAgent):
    """
    Specialized agent focused on error resolution and debugging.
    
    The TroubleshooterAgent is responsible for:
    1. Diagnosing and resolving framework-related errors
    2. Explaining error messages and their causes
    3. Providing debugging strategies and techniques
    4. Helping users troubleshoot implementation issues
    """
    
    # Class attributes for agent registry support
    DESCRIPTION = "Expert in error resolution and debugging"
    CAPABILITIES = [
        "error_diagnosis",
        "debugging_assistance",
        "issue_resolution",
        "troubleshooting_guidance",
        "error_prevention"
    ]
    PRIORITY = 9  # High priority for troubleshooting tasks
    
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
        Initialize the TroubleshooterAgent.
        
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
                "explanation_depth": "comprehensive",  # basic, moderate, comprehensive
                "solution_approach": "step_by_step",  # direct, step_by_step, guided
                "include_prevention": True,
                "provide_alternatives": True,
                "error_categorization": True
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
            agent_type="troubleshooter",
            description="Expert in error resolution and debugging",
            is_logging=is_logging,
            *args, **kwargs
        )
        
        if self.is_logging:
            logger.info(f"Initialized TroubleshooterAgent with config: {self.config}")
    
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
        Enhance a user query with learning context specific to the troubleshooter role.
        
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
        Process and structure the troubleshooter's response.
        
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
        
        # Add troubleshooter-specific metadata
        processed["error_diagnosis"] = self._extract_error_diagnosis(processed["content"])
        processed["solution_steps"] = self._extract_solution_steps(processed["content"])
        processed["error_type"] = self._determine_error_type(processed["content"])
        
        return processed
    
    def _extract_error_diagnosis(self, response_content: str) -> Dict:
        """
        Extract error diagnosis information from the response.
        
        Parameters:
        ----------
        response_content : str
            Content of the response
            
        Returns:
        -------
        Dict
            Error diagnosis information
        """
        # In a real implementation, this would use more sophisticated parsing
        # For now, we'll use a simple approach to identify diagnosis sections
        
        import re
        
        diagnosis = {
            "identified_error": "",
            "cause": "",
            "severity": "medium"  # Default severity
        }
        
        # Try to find error identification
        error_pattern = r"(?:error|issue|problem)(?:\s+is|\s*:\s*|\s+appears to be)(.*?)(?:\n|$)"
        error_match = re.search(error_pattern, response_content, re.IGNORECASE)
        if error_match:
            diagnosis["identified_error"] = error_match.group(1).strip()
        
        # Try to find cause
        cause_pattern = r"(?:cause|reason|why)(?:\s+is|\s*:\s*)(.*?)(?:\n|$)"
        cause_match = re.search(cause_pattern, response_content, re.IGNORECASE)
        if cause_match:
            diagnosis["cause"] = cause_match.group(1).strip()
        
        # Try to determine severity
        content_lower = response_content.lower()
        if any(term in content_lower for term in ["critical", "severe", "major", "serious"]):
            diagnosis["severity"] = "high"
        elif any(term in content_lower for term in ["minor", "simple", "easy", "trivial"]):
            diagnosis["severity"] = "low"
        
        return diagnosis
    
    def _extract_solution_steps(self, response_content: str) -> List[Dict]:
        """
        Extract solution steps from the response.
        
        Parameters:
        ----------
        response_content : str
            Content of the response
            
        Returns:
        -------
        List[Dict]
            List of solution steps
        """
        # In a real implementation, this would use more sophisticated parsing
        # For now, we'll use a simple approach to identify numbered steps
        
        steps = []
        import re
        
        # Find solution section
        solution_section = response_content
        solution_headers = ["solution", "fix", "resolution", "how to fix", "steps to fix"]
        
        for header in solution_headers:
            pattern = f"(?:{header})(?:\\s+is|\\s*:\\s*)(.*?)(?=\\n\\s*(?:[A-Z][a-z]+\\s*:|$))"
            match = re.search(pattern, response_content, re.IGNORECASE | re.DOTALL)
            if match:
                solution_section = match.group(1)
                break
        
        # Find numbered steps (1. Step description)
        step_pattern = r"(?:^|\n)(?:Step\s*)?(\d+)[\.:\)]\s*([^\n]+)"
        matches = re.findall(step_pattern, solution_section, re.MULTILINE)
        
        for step_num, step_text in matches:
            steps.append({
                "number": int(step_num),
                "description": step_text.strip()
            })
        
        # If no numbered steps found, try to split by newlines
        if not steps and solution_section:
            lines = [line.strip() for line in solution_section.split("\n") if line.strip()]
            for idx, line in enumerate(lines):
                steps.append({
                    "number": idx + 1,
                    "description": line
                })
        
        # Sort steps by number
        steps.sort(key=lambda x: x["number"])
        
        return steps
    
    def _determine_error_type(self, response_content: str) -> str:
        """
        Determine the type of error described in the response.
        
        Parameters:
        ----------
        response_content : str
            Content of the response
            
        Returns:
        -------
        str
            Type of error
        """
        # In a real implementation, this would use NLP to classify the content
        # For now, we'll use a simple keyword-based approach
        
        content_lower = response_content.lower()
        
        if "syntax" in content_lower:
            return "syntax_error"
        elif "type" in content_lower and "error" in content_lower:
            return "type_error"
        elif "reference" in content_lower:
            return "reference_error"
        elif "import" in content_lower or "module" in content_lower:
            return "import_error"
        elif "runtime" in content_lower:
            return "runtime_error"
        elif "logic" in content_lower:
            return "logic_error"
        elif "configuration" in content_lower or "config" in content_lower:
            return "configuration_error"
        elif "permission" in content_lower or "access" in content_lower:
            return "permission_error"
        elif "network" in content_lower or "connection" in content_lower:
            return "network_error"
        else:
            return "general_error"