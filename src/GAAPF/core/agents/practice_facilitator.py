import logging
from typing import Dict, List, Optional, Union, Any
from pathlib import Path

from . import SpecializedAgent
from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.tools import BaseTool
from ...prompts.practice_facilitator import generate_system_prompt

# Enhanced imports for adaptive learning (silent fallback if unavailable)
try:
    from ..learning.bayesian_kt import BayesianKnowledgeTracker
    from ..gamification.achievement_system import AchievementSystem
    from ..config.adaptive_config import AdaptiveConfigManager
    ENHANCED_FEATURES_AVAILABLE = True
except ImportError:
    BayesianKnowledgeTracker = None
    AchievementSystem = None
    AdaptiveConfigManager = None
    ENHANCED_FEATURES_AVAILABLE = False

# Setup logging (avoid duplicate handlers)
if not logging.getLogger().handlers:
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
        
        # Initialize enhanced learning components
        self.bkt_tracker = None
        self.achievement_system = None
        self.adaptive_config_manager = None
        
        if ENHANCED_FEATURES_AVAILABLE:
            try:
                # Initialize BKT tracker
                if BayesianKnowledgeTracker:
                    self.bkt_tracker = BayesianKnowledgeTracker(
                        user_id="default",  # Will be updated per user
                        storage_path=memory_path.parent / "bkt_data" if memory_path else None
                    )
                    if self.is_logging:
                        logger.info("BKT tracker initialized for practice facilitator")
                
                # Initialize achievement system
                if AchievementSystem:
                    self.achievement_system = AchievementSystem(
                        storage_path=memory_path.parent / "achievements" if memory_path else None
                    )
                    if self.is_logging:
                        logger.info("Achievement system initialized for practice facilitator")
                
                # Initialize adaptive config manager
                if AdaptiveConfigManager:
                    self.adaptive_config_manager = AdaptiveConfigManager(
                        storage_path=memory_path.parent / "adaptive_config" if memory_path else None
                    )
                    if self.is_logging:
                        logger.info("Adaptive config manager initialized for practice facilitator")
                        
            except Exception as e:
                if self.is_logging:
                    logger.warning(f"Failed to initialize enhanced features: {e}")
        
        if self.is_logging:
            logger.info(f"Initialized PracticeFacilitatorAgent with config: {self.config}")
    
    def _generate_system_prompt(self, learning_context: Dict = None) -> str:
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
        # Fallback to hub-managed learning_state when current_module is not set
        if not current_module:
            current_module = (
                (learning_context.get("session", {}) or {}).get("learning_state", {}) or {}
            ).get("current_module", "") or current_module
        user_profile = learning_context.get("user_profile", {})
        
        # Get completed exercises and skill level
        # Prefer hub-tracked counter; fallback to user_profile list length if present
        _ls = ((learning_context.get("session", {}) or {}).get("learning_state", {}) or {})
        completed_exercises_count = int(_ls.get("completed_exercises", 0))
        if completed_exercises_count == 0:
            completed_exercises_count = len(user_profile.get("completed_exercises", []))
        skill_level = user_profile.get("skill_level", {}).get(current_module, "beginner")
        
        # Get module details if available
        module_info = {}
        if current_module and "modules" in framework_config:
            module_info = framework_config.get("modules", {}).get(current_module, {})
        
        # Get exercise types for this module
        exercise_types = module_info.get("exercise_types", [])
        exercises_str = ", ".join(exercise_types) if exercise_types else "None specified"
        
        # Get adaptive learning context if available
        user_id = learning_context.get("user_id", "unknown")
        adaptive_context = ""
        
        if self.bkt_tracker:
            try:
                knowledge_state = self.bkt_tracker.get_knowledge_probabilities(user_id)
                mastery_status = self.bkt_tracker.get_mastery_status(user_id)
                recommended_skills = self.bkt_tracker.recommend_skills(user_id)
                
                # Format knowledge state info for practice
                if knowledge_state:
                    weak_skills = sorted([(skill, prob) for skill, prob in knowledge_state.items() if prob < 0.6], 
                                        key=lambda x: x[1])
                    if weak_skills:
                        skill_info = ", ".join([f"{skill}: {prob:.2f}" for skill, prob in weak_skills[:3]])
                        adaptive_context += f"\n- Skills needing practice: {skill_info}"
                
                if mastery_status:
                    mastered_skills = [skill for skill, mastered in mastery_status.items() if mastered]
                    adaptive_context += f"\n- Mastered skills: {', '.join(mastered_skills) if mastered_skills else 'None'}"
                
                if recommended_skills:
                    adaptive_context += f"\n- Recommended practice areas: {', '.join(recommended_skills[:3])}"
                    
            except Exception as e:
                if self.is_logging:
                    logger.warning(f"Failed to get BKT context: {e}")
        
        if self.adaptive_config_manager:
            try:
                user_config = self.adaptive_config_manager.get_user_config(user_id)
                difficulty_level = user_config.get('difficulty', {}).get('base_level', 0.5)
                learning_style = user_config.get('personalization', {}).get('learning_style', 'balanced')
                exercise_preference = user_config.get('content', {}).get('exercise_types', [])
                adaptive_context += f"\n- Difficulty level: {difficulty_level:.2f}"
                adaptive_context += f"\n- Learning style: {learning_style}"
                if exercise_preference:
                    adaptive_context += f"\n- Preferred exercise types: {', '.join(exercise_preference)}"
            except Exception as e:
                if self.is_logging:
                    logger.warning(f"Failed to get adaptive config: {e}")
        
        # Add practice facilitator-specific context
        practice_context = f"""
Additional context for practice activities:
- Module skill level: {skill_level}
- Completed exercises: {completed_exercises_count}
- Recommended exercise types: {exercises_str}{adaptive_context}

As a practice facilitator, create adaptive exercises based on the user's knowledge state and learning preferences.
Focus on skills that need practice and adjust difficulty according to their current level.
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
        # Lightweight signals for hub state updates (Phase 5)
        try:
            signals = dict(processed.get("signals") or {})
            signals["exercise_issued"] = True
            signals["exercise_type"] = processed["exercise_type"]
            processed["signals"] = signals
        except Exception:
            pass
        
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
