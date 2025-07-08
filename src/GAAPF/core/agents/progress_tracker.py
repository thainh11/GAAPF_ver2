# Creating the ProgressTrackerAgent

import logging
from typing import Dict, List, Optional, Union, Any
from pathlib import Path

from . import SpecializedAgent
from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.tools import BaseTool
from src.GAAPF.prompts.progress_tracker import generate_system_prompt

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProgressTrackerAgent(SpecializedAgent):
    """
    Specialized agent focused on monitoring learning progress.
    
    The ProgressTrackerAgent is responsible for:
    1. Tracking and analyzing user learning progress
    2. Identifying knowledge gaps and areas for improvement
    3. Recommending next steps in the learning journey
    4. Providing insights on learning patterns and effectiveness
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
        Initialize the ProgressTrackerAgent.
        
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
                "tracking_detail": "comprehensive",
                "visualization": True,
                "focus_on_improvement": True,
                "track_time_spent": True,
                "provide_comparisons": True
            }
        
        # Set default tools if not provided
        if not tools:
            tools = [
                "websearch_tools"
            ]
        
        # Initialize the base specialized agent
        super().__init__(
            llm=llm,
            tools=tools,
            memory_path=memory_path,
            config=config,
            agent_type="progress_tracker",
            description="Expert in monitoring learning progress",
            is_logging=is_logging,
            *args, **kwargs
        )
        
        if self.is_logging:
            logger.info(f"Initialized ProgressTrackerAgent with config: {self.config}")

    def _generate_system_prompt(self) -> str:
        """
        Generate a system prompt for this agent.
        
        Returns:
        -------
        str
            System prompt for the agent
        """
        return generate_system_prompt(self.config) 