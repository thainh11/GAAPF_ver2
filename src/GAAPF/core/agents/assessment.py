import logging
from typing import Dict, List, Optional, Union, Any
from pathlib import Path

from . import SpecializedAgent
from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.tools import BaseTool
from ...prompts.assessment import generate_system_prompt

# Enhanced imports for adaptive learning
try:
    from ..learning.bayesian_kt import BayesianKnowledgeTracker
    from ..gamification.achievement_system import AchievementSystem
    from ..config.adaptive_config import AdaptiveConfigManager
    ENHANCED_FEATURES_AVAILABLE = True
except ImportError:
    ENHANCED_FEATURES_AVAILABLE = False
    BayesianKnowledgeTracker = None
    AchievementSystem = None
    AdaptiveConfigManager = None

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AssessmentAgent(SpecializedAgent):
    """
    Specialized agent focused on evaluating user knowledge and progress.
    
    The AssessmentAgent is responsible for:
    1. Creating knowledge assessment questions and quizzes
    2. Evaluating user responses and providing feedback
    3. Identifying knowledge gaps and areas for improvement
    4. Tracking learning progress over time
    """
    
    # Class attributes for agent registry support
    DESCRIPTION = "Expert in evaluating user knowledge and progress through assessments"
    CAPABILITIES = [
        "knowledge_assessment",
        "quiz_creation",
        "progress_evaluation",
        "feedback_provision",
        "gap_analysis"
    ]
    PRIORITY = 8  # High priority for assessment tasks
    
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
        Initialize the AssessmentAgent.
        
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
                "assessment_style": "balanced",  # theoretical, practical, balanced
                "feedback_detail": "moderate",  # minimal, moderate, comprehensive
                "adaptive_difficulty": True,
                "track_progress": True,
                "question_types": ["multiple_choice", "short_answer", "coding"]
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
            agent_type="assessment",
            description="Expert in evaluating user knowledge and progress",
            is_logging=is_logging,
            *args, **kwargs
        )
        
        # Initialize enhanced learning components if available
        self.bkt_tracker = None
        self.achievement_system = None
        self.adaptive_config_manager = None
        
        if ENHANCED_FEATURES_AVAILABLE:
            try:
                self.bkt_tracker = BayesianKnowledgeTracker()
                self.achievement_system = AchievementSystem()
                self.adaptive_config_manager = AdaptiveConfigManager()
                if self.is_logging:
                    logger.info("Enhanced adaptive learning features initialized for AssessmentAgent")
            except Exception as e:
                if self.is_logging:
                    logger.warning(f"Failed to initialize enhanced features: {e}")
        
        if self.is_logging:
            logger.info(f"Initialized AssessmentAgent with config: {self.config}")
    
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
        Enhance a user query with learning context specific to the assessment role.
        
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
        
        # Extract additional relevant context for assessment
        framework_config = learning_context.get("framework_config", {})
        current_module = learning_context.get("current_module", "")
        user_profile = learning_context.get("user_profile", {})
        
        # Get learning history and assessment performance
        assessment_history = user_profile.get("assessment_history", {})
        current_module_assessments = assessment_history.get(current_module, [])
        
        # Calculate average score if available
        avg_score = "Not available"
        if current_module_assessments:
            scores = [assessment.get("score", 0) for assessment in current_module_assessments]
            if scores:
                avg_score = f"{sum(scores) / len(scores):.1f}%"
        
        # Get module details if available
        module_info = {}
        if current_module and "modules" in framework_config:
            module_info = framework_config.get("modules", {}).get(current_module, {})
        
        # Get key concepts for this module
        key_concepts = module_info.get("concepts", [])
        concepts_str = ", ".join(key_concepts) if key_concepts else "None specified"
        
        # Get adaptive learning context if available
        user_id = learning_context.get("user_id", "unknown")
        adaptive_context = ""
        
        if self.bkt_tracker:
            try:
                knowledge_state = self.bkt_tracker.get_knowledge_probabilities(user_id)
                mastery_status = self.bkt_tracker.get_mastery_status(user_id)
                recommended_skills = self.bkt_tracker.recommend_skills(user_id)
                
                # Format knowledge state info
                if knowledge_state:
                    top_skills = sorted(knowledge_state.items(), key=lambda x: x[1], reverse=True)[:3]
                    skill_info = ", ".join([f"{skill}: {prob:.2f}" for skill, prob in top_skills])
                    adaptive_context += f"\n- Top knowledge areas: {skill_info}"
                
                if mastery_status:
                    mastered_skills = [skill for skill, mastered in mastery_status.items() if mastered]
                    adaptive_context += f"\n- Mastered skills: {', '.join(mastered_skills) if mastered_skills else 'None'}"
                
                if recommended_skills:
                    adaptive_context += f"\n- Recommended focus areas: {', '.join(recommended_skills[:3])}"
                    
            except Exception as e:
                if self.is_logging:
                    logger.warning(f"Failed to get BKT context: {e}")
        
        if self.adaptive_config_manager:
            try:
                user_config = self.adaptive_config_manager.get_user_config(user_id)
                difficulty_level = user_config.get('difficulty', {}).get('base_level', 0.5)
                learning_style = user_config.get('personalization', {}).get('learning_style', 'balanced')
                adaptive_context += f"\n- Difficulty level: {difficulty_level:.2f}"
                adaptive_context += f"\n- Learning style: {learning_style}"
            except Exception as e:
                if self.is_logging:
                    logger.warning(f"Failed to get adaptive config: {e}")
        
        # Add assessment-specific context
        assessment_context = f"""
Additional context for assessment:
- Previous assessments in this module: {len(current_module_assessments)}
- Average score in this module: {avg_score}
- Key concepts to assess: {concepts_str}{adaptive_context}

As an assessment agent, create adaptive assessments based on the user's knowledge state and learning preferences.
Adjust difficulty and question types according to their current level and recommended focus areas.
"""
        
        # Combine contexts
        enhanced_query = context_prefix + assessment_context
        
        return enhanced_query
    
    def _process_response(self, response: Any, learning_context: Dict) -> Dict:
        """
        Process and structure the assessment agent's response.
        
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
        
        # Add assessment-specific metadata
        processed["assessment_type"] = self._determine_assessment_type(processed["content"])
        processed["questions"] = self._extract_questions(processed["content"])
        
        return processed
    
    def _determine_assessment_type(self, response_content: str) -> str:
        """
        Determine the type of assessment provided in the response.
        
        Parameters:
        ----------
        response_content : str
            Content of the response
            
        Returns:
        -------
        str
            Type of assessment
        """
        # In a real implementation, this would use NLP to classify the content
        # For now, we'll use a simple keyword-based approach
        
        content_lower = response_content.lower()
        
        if "quiz" in content_lower:
            return "quiz"
        elif "test" in content_lower:
            return "test"
        elif "exercise" in content_lower:
            return "exercise"
        elif "feedback" in content_lower or "evaluation" in content_lower:
            return "feedback"
        else:
            return "general_assessment"
    
    def _extract_questions(self, response_content: str) -> List[Dict]:
        """
        Extract assessment questions from the response.
        
        Parameters:
        ----------
        response_content : str
            Content of the response
            
        Returns:
        -------
        List[Dict]
            List of extracted questions with metadata
        """
        # In a real implementation, this would use more sophisticated parsing
        # For now, we'll use a simple approach to identify numbered questions
        
        questions = []
        import re
        
        # Find numbered questions (1. Question text)
        question_pattern = r"(\d+\.|\w+\))\s+(.*?)(?=\n\d+\.|\n\w+\)|\Z)"
        matches = re.findall(question_pattern, response_content, re.DOTALL)
        
        for idx, (num, text) in enumerate(matches):
            # Try to determine if it's multiple choice
            is_multiple_choice = bool(re.search(r"[A-D]\.|\([A-D]\)", text))
            
            # Try to extract choices if multiple choice
            choices = []
            if is_multiple_choice:
                choice_pattern = r"([A-D]\.|\([A-D]\))\s+(.*?)(?=[A-D]\.|\([A-D]\)|\Z)"
                choice_matches = re.findall(choice_pattern, text, re.DOTALL)
                choices = [choice_text.strip() for _, choice_text in choice_matches]
            
            questions.append({
                "id": idx,
                "number": num.strip(),
                "text": text.strip(),
                "type": "multiple_choice" if is_multiple_choice else "open_ended",
                "choices": choices if is_multiple_choice else []
            })
        
        return questions
    
    def create_adaptive_assessment(self, user_id: str, skill: str, difficulty_level: float = None) -> Dict:
        """
        Create an adaptive assessment based on user's knowledge state.
        
        Parameters:
        ----------
        user_id : str
            User identifier
        skill : str
            Skill to assess
        difficulty_level : float, optional
            Override difficulty level (0.0-1.0)
            
        Returns:
        -------
        Dict
            Adaptive assessment configuration
        """
        assessment_config = {
            "user_id": user_id,
            "skill": skill,
            "timestamp": time.time(),
            "questions": [],
            "metadata": {}
        }
        
        if not self.bkt_tracker:
            # Fallback to basic assessment
            assessment_config["type"] = "basic"
            assessment_config["difficulty"] = difficulty_level or 0.5
            return assessment_config
        
        try:
            # Get user's current knowledge state
            knowledge_state = self.bkt_tracker.get_knowledge_probabilities(user_id)
            skill_probability = knowledge_state.get(skill, 0.5)
            
            # Determine adaptive difficulty
            if difficulty_level is None:
                if skill_probability < 0.3:
                    difficulty_level = 0.3  # Easy
                elif skill_probability < 0.7:
                    difficulty_level = 0.5  # Medium
                else:
                    difficulty_level = 0.8  # Hard
            
            # Get user preferences
            user_config = {}
            if self.adaptive_config_manager:
                user_config = self.adaptive_config_manager.get_user_config(user_id)
            
            learning_style = user_config.get('personalization', {}).get('learning_style', 'balanced')
            
            # Configure assessment based on learning style
            if learning_style == 'visual':
                question_types = ['diagram', 'multiple_choice', 'matching']
            elif learning_style == 'practical':
                question_types = ['coding', 'problem_solving', 'application']
            else:
                question_types = ['multiple_choice', 'short_answer', 'explanation']
            
            assessment_config.update({
                "type": "adaptive",
                "difficulty": difficulty_level,
                "skill_probability": skill_probability,
                "learning_style": learning_style,
                "question_types": question_types,
                "metadata": {
                    "bkt_state": knowledge_state,
                    "adaptive_config": user_config
                }
            })
            
        except Exception as e:
            if self.is_logging:
                logger.error(f"Failed to create adaptive assessment: {e}")
            # Fallback to basic assessment
            assessment_config["type"] = "basic"
            assessment_config["difficulty"] = difficulty_level or 0.5
        
        return assessment_config
    
    def evaluate_response_with_bkt(self, user_id: str, skill: str, response: str, 
                                   correct_answer: str = None, is_correct: bool = None) -> Dict:
        """
        Evaluate user response and update BKT knowledge state.
        
        Parameters:
        ----------
        user_id : str
            User identifier
        skill : str
            Skill being assessed
        response : str
            User's response
        correct_answer : str, optional
            The correct answer for comparison
        is_correct : bool, optional
            Whether the response is correct (if not provided, will be determined)
            
        Returns:
        -------
        Dict
            Evaluation results with updated knowledge state
        """
        evaluation = {
            "user_id": user_id,
            "skill": skill,
            "response": response,
            "timestamp": time.time(),
            "is_correct": is_correct,
            "feedback": "",
            "knowledge_update": {}
        }
        
        # Determine correctness if not provided
        if is_correct is None and correct_answer is not None:
            # Simple string comparison - in practice, this would be more sophisticated
            is_correct = response.strip().lower() == correct_answer.strip().lower()
            evaluation["is_correct"] = is_correct
        
        # Update BKT if available
        if self.bkt_tracker and is_correct is not None:
            try:
                # Update knowledge state
                old_probability = self.bkt_tracker.get_knowledge_probabilities(user_id).get(skill, 0.5)
                self.bkt_tracker.update_knowledge(user_id, skill, is_correct)
                new_probability = self.bkt_tracker.get_knowledge_probabilities(user_id).get(skill, 0.5)
                
                evaluation["knowledge_update"] = {
                    "old_probability": old_probability,
                    "new_probability": new_probability,
                    "change": new_probability - old_probability
                }
                
                # Generate adaptive feedback
                if new_probability > 0.8:
                    evaluation["feedback"] = f"Excellent! You've mastered {skill}. Ready for more advanced topics."
                elif new_probability > 0.6:
                    evaluation["feedback"] = f"Good progress on {skill}. Keep practicing to solidify your understanding."
                elif new_probability > 0.4:
                    evaluation["feedback"] = f"You're learning {skill}. Consider reviewing the fundamentals."
                else:
                    evaluation["feedback"] = f"Let's focus more on {skill}. Additional practice recommended."
                
            except Exception as e:
                if self.is_logging:
                    logger.error(f"Failed to update BKT: {e}")
                evaluation["feedback"] = "Response recorded. Continue practicing!"
        
        # Update achievements if available
        if self.achievement_system and is_correct:
            try:
                self.achievement_system.update_progress(user_id, 'assessment_completed', {
                    'skill': skill,
                    'correct': is_correct
                })
            except Exception as e:
                if self.is_logging:
                    logger.warning(f"Failed to update achievements: {e}")
        
        return evaluation
    
    def get_next_assessment_recommendation(self, user_id: str) -> Dict:
        """
        Get recommendation for the next assessment based on user's knowledge state.
        
        Parameters:
        ----------
        user_id : str
            User identifier
            
        Returns:
        -------
        Dict
            Assessment recommendation
        """
        recommendation = {
            "user_id": user_id,
            "recommended_skill": None,
            "difficulty": 0.5,
            "reason": "No specific recommendation available",
            "priority": "medium"
        }
        
        if not self.bkt_tracker:
            return recommendation
        
        try:
            # Get recommended skills from BKT
            recommended_skills = self.bkt_tracker.recommend_skills(user_id)
            
            if recommended_skills:
                skill = recommended_skills[0]  # Take the top recommendation
                knowledge_state = self.bkt_tracker.get_knowledge_probabilities(user_id)
                skill_probability = knowledge_state.get(skill, 0.5)
                
                # Determine priority and difficulty
                if skill_probability < 0.3:
                    priority = "high"
                    difficulty = 0.3
                    reason = f"Low proficiency in {skill} - needs immediate attention"
                elif skill_probability < 0.6:
                    priority = "medium"
                    difficulty = 0.5
                    reason = f"Moderate proficiency in {skill} - good for practice"
                else:
                    priority = "low"
                    difficulty = 0.7
                    reason = f"High proficiency in {skill} - ready for advanced assessment"
                
                recommendation.update({
                    "recommended_skill": skill,
                    "difficulty": difficulty,
                    "reason": reason,
                    "priority": priority,
                    "current_probability": skill_probability
                })
            
        except Exception as e:
            if self.is_logging:
                logger.error(f"Failed to get assessment recommendation: {e}")
        
        return recommendation