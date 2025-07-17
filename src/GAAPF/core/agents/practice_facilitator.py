import logging
from typing import Dict, List, Optional, Union, Any
from pathlib import Path

from . import SpecializedAgent
from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.tools import BaseTool
from GAAPF.prompts.practice_facilitator import generate_system_prompt

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PracticeFacilitatorAgent(SpecializedAgent):
    """
    Specialized agent focused on creating exercises and practice activities.
    
    The PracticeFacilitatorAgent is responsible for:
    1. Creating practical exercises to reinforce learning
    2. Designing coding challenges of appropriate difficulty
    3. Providing hands-on activities to apply concepts
    4. Offering feedback on practice attempts
    """
    
    # Class attributes for agent registry support
    DESCRIPTION = "Expert in creating exercises and practice activities"
    CAPABILITIES = [
        "exercise_creation",
        "challenge_design",
        "hands_on_activities",
        "practice_feedback",
        "difficulty_adaptation"
    ]
    PRIORITY = 7  # High priority for practice tasks
    
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
        Initialize the PracticeFacilitatorAgent.
        
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
                "difficulty_adjustment": "adaptive",  # fixed, adaptive, progressive
                "exercise_style": "guided",  # guided, open-ended, project-based
                "provide_hints": True,
                "provide_solutions": True,
                "real_world_focus": "moderate"  # low, moderate, high
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
            agent_type="practice_facilitator",
            description="Expert in creating exercises and practice activities",
            is_logging=is_logging,
            *args, **kwargs
        )
        
        if self.is_logging:
            logger.info(f"Initialized PracticeFacilitatorAgent with config: {self.config}")
    
    def _generate_system_prompt(self) -> str:
        """
        Generate a system prompt for this agent.
        
        Returns:
        -------
        str
            System prompt for the agent
        """
        return generate_system_prompt(self.config)
    
    def _enhance_query_with_context(self, query: str, learning_context: Dict) -> str:
        """
        Enhance a user query with learning context specific to the practice facilitator role.
        
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
        # Get base context enhancement
        context_prefix = super()._enhance_query_with_context(query, learning_context)
        
        # Extract additional relevant context for practice activities
        framework_config = learning_context.get("framework_config", {})
        current_module = learning_context.get("current_module", "")
        user_profile = learning_context.get("user_profile", {})
        
        # Get completed exercises and skill level
        completed_exercises = user_profile.get("completed_exercises", [])
        skill_level = user_profile.get("skill_level", {}).get(current_module, "beginner")
        
        # Get module details if available
        module_info = {}
        if current_module and "modules" in framework_config:
            module_info = framework_config.get("modules", {}).get(current_module, {})
        
        # Get exercise types for this module
        exercise_types = module_info.get("exercise_types", [])
        exercises_str = ", ".join(exercise_types) if exercise_types else "None specified"
        
        # Add practice facilitator-specific context
        practice_context = f"""
Additional context for practice activities:
- Module skill level: {skill_level}
- Completed exercises: {len(completed_exercises)}
- Recommended exercise types: {exercises_str}

As a practice facilitator, create appropriate exercises or provide guidance on practice activities.
"""
        
        # Combine contexts
        enhanced_query = context_prefix + practice_context
        
        return enhanced_query
    
    def _process_response(self, response: Any, learning_context: Dict) -> Dict:
        """
        Process and structure the practice facilitator's response.
        
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
        
        # Add practice facilitator-specific metadata
        processed["exercise_type"] = self._determine_exercise_type(processed["content"])
        processed["difficulty_level"] = self._determine_difficulty_level(processed["content"], learning_context)
        processed["has_solution"] = "solution" in processed["content"].lower()
        
        return processed
    
    def _determine_exercise_type(self, response_content: str) -> str:
        """
        Determine the type of exercise provided in the response.
        
        Parameters:
        ----------
        response_content : str
            Content of the response
            
        Returns:
        -------
        str
            Type of exercise
        """
        # In a real implementation, this would use NLP to classify the content
        # For now, we'll use a simple keyword-based approach
        
        content_lower = response_content.lower()
        
        if "quiz" in content_lower or "multiple choice" in content_lower:
            return "quiz"
        elif "debug" in content_lower or "fix" in content_lower or "error" in content_lower:
            return "debugging_exercise"
        elif "implement" in content_lower or "create" in content_lower or "write" in content_lower:
            return "implementation_task"
        elif "project" in content_lower:
            return "project"
        elif "challenge" in content_lower:
            return "challenge"
        else:
            return "general_exercise"
    
    def _determine_difficulty_level(self, response_content: str, learning_context: Dict) -> str:
        """
        Determine the difficulty level of the exercise.
        
        Parameters:
        ----------
        response_content : str
            Content of the response
        learning_context : Dict
            Current learning context
            
        Returns:
        -------
        str
            Difficulty level (beginner, intermediate, advanced)
        """
        # In a real implementation, this would use more sophisticated analysis
        # For now, we'll use a simple keyword-based approach
        
        content_lower = response_content.lower()
        
        # Check for explicit difficulty indicators
        if "advanced" in content_lower or "challenging" in content_lower or "difficult" in content_lower:
            return "advanced"
        elif "intermediate" in content_lower or "moderate" in content_lower:
            return "intermediate"
        elif "beginner" in content_lower or "basic" in content_lower or "simple" in content_lower:
            return "beginner"
        
        # Fall back to user's experience level
        user_profile = learning_context.get("user_profile", {})
        current_module = learning_context.get("current_module", "")
        return user_profile.get("skill_level", {}).get(current_module, "beginner")