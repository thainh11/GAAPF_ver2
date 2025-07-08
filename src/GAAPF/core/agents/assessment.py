import logging
from typing import Dict, List, Optional, Union, Any
from pathlib import Path

from . import SpecializedAgent
from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.tools import BaseTool
from src.GAAPF.prompts.assessment import generate_system_prompt

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
        
        # Add assessment-specific context
        assessment_context = f"""
Additional context for assessment:
- Previous assessments in this module: {len(current_module_assessments)}
- Average score in this module: {avg_score}
- Key concepts to assess: {concepts_str}

As an assessment agent, evaluate knowledge or create appropriate assessment questions.
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