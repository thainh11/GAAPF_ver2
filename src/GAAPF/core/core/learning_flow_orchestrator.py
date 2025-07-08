"""
Learning Flow Orchestrator Module for GAAPF Architecture

This module provides LangGraph-based orchestration for the main learning flow,
managing the complete learning experience from curriculum execution to user interaction.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, TypedDict
from pathlib import Path
import json
from datetime import datetime

from langgraph.graph import StateGraph, END, START
from langgraph.graph.state import CompiledStateGraph

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LearningFlowState(TypedDict):
    """State for the learning flow orchestration"""
    # Session information
    session_id: str
    user_id: str
    framework_name: str
    
    # Current curriculum and progress
    curriculum: Dict[str, Any]
    current_module: Dict[str, Any]
    module_index: int
    progress_metrics: Dict[str, float]
    
    # Learning content and interactions
    theory_content: str
    code_examples: List[Dict[str, str]]
    practice_exercises: List[Dict[str, Any]]
    quiz_questions: List[Dict[str, Any]]
    
    # User responses and feedback
    user_responses: List[Dict[str, Any]]
    current_quiz_results: Dict[str, Any]
    performance_data: Dict[str, float]
    
    # Flow control
    next_action: str
    requires_adjustment: bool
    completion_status: str
    
    # Error and retry handling
    errors: List[str]
    retry_count: int


class LearningFlowOrchestrator:
    """
    Orchestrates the main learning flow using LangGraph for complex workflow management.
    
    This class coordinates all aspects of the learning experience including content delivery,
    user interaction, progress tracking, and adaptive adjustments.
    """
    
    def __init__(
        self,
        curriculum_manager,
        content_generator,
        quiz_generator,
        progress_tracker,
        learning_hub,
        is_logging: bool = False
    ):
        """
        Initialize the Learning Flow Orchestrator.
        
        Args:
            curriculum_manager: CurriculumManager instance
            content_generator: Content generation component
            quiz_generator: Quiz generation component  
            progress_tracker: Progress tracking component
            learning_hub: LearningHub instance
            is_logging: Whether to enable detailed logging
        """
        self.curriculum_manager = curriculum_manager
        self.content_generator = content_generator
        self.quiz_generator = quiz_generator
        self.progress_tracker = progress_tracker
        self.learning_hub = learning_hub
        self.is_logging = is_logging
        
        # Initialize the learning flow graph
        self.flow_graph = self._build_learning_flow_graph()
        
        if self.is_logging:
            logger.info("Initialized LearningFlowOrchestrator")
    
    def _build_learning_flow_graph(self) -> CompiledStateGraph:
        """
        Build the LangGraph state machine for learning flow orchestration.
        
        Returns:
            Compiled state graph for learning flow management
        """
        # Create state graph
        graph = StateGraph(LearningFlowState)
        
        # Add nodes for different learning phases
        graph.add_node("initialize_session", self._initialize_session_node)
        graph.add_node("load_current_module", self._load_current_module_node)
        graph.add_node("generate_theory_content", self._generate_theory_content_node)
        graph.add_node("generate_code_examples", self._generate_code_examples_node)
        graph.add_node("present_content", self._present_content_node)
        graph.add_node("generate_quiz", self._generate_quiz_node)
        graph.add_node("evaluate_responses", self._evaluate_responses_node)
        graph.add_node("update_progress", self._update_progress_node)
        graph.add_node("check_module_completion", self._check_module_completion_node)
        graph.add_node("adjust_difficulty", self._adjust_difficulty_node)
        graph.add_node("advance_to_next_module", self._advance_to_next_module_node)
        graph.add_node("complete_curriculum", self._complete_curriculum_node)
        graph.add_node("handle_error", self._handle_error_node)
        
        # Define flow transitions
        graph.add_edge(START, "initialize_session")
        graph.add_edge("initialize_session", "load_current_module")
        graph.add_edge("load_current_module", "generate_theory_content")
        graph.add_edge("generate_theory_content", "generate_code_examples")
        graph.add_edge("generate_code_examples", "present_content")
        graph.add_edge("present_content", "generate_quiz")
        graph.add_edge("generate_quiz", "evaluate_responses")
        graph.add_edge("evaluate_responses", "update_progress")
        graph.add_edge("update_progress", "check_module_completion")
        
        # Add conditional edges for flow control
        graph.add_conditional_edges(
            "check_module_completion",
            self._determine_next_action,
            {
                "adjust_difficulty": "adjust_difficulty",
                "next_module": "advance_to_next_module",
                "complete": "complete_curriculum",
                "retry": "generate_theory_content",
                "error": "handle_error"
            }
        )
        
        graph.add_edge("adjust_difficulty", "generate_theory_content")
        graph.add_edge("advance_to_next_module", "load_current_module")
        graph.add_edge("complete_curriculum", END)
        graph.add_edge("handle_error", END)
        
        return graph.compile()
    
    async def start_learning_session(
        self,
        user_id: str,
        framework_name: str,
        curriculum: Dict[str, Any],
        session_config: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Start a new learning session with the orchestrated flow.
        
        Args:
            user_id: User identifier
            framework_name: Framework being learned
            curriculum: Approved curriculum to follow
            session_config: Optional session configuration
            
        Returns:
            Session initiation result and status
        """
        session_id = f"session_{user_id}_{int(datetime.now().timestamp())}"
        
        try:
            if self.is_logging:
                logger.info(f"Learning session started: {session_id}")
            
            return {
                "session_id": session_id,
                "status": "started",
                "framework_name": framework_name,
                "curriculum": curriculum
            }
            
        except Exception as e:
            logger.error(f"Error starting learning session: {str(e)}")
            return {
                "session_id": session_id,
                "status": "error",
                "error": str(e)
            }
    
    async def continue_learning_session(
        self,
        session_id: str,
        user_responses: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Continue a learning session with user responses.
        
        Args:
            session_id: Session identifier
            user_responses: User's responses to questions/exercises
            
        Returns:
            Continuation result with next content or completion status
        """
        try:
            # Load existing session state
            state = await self._load_session_state(session_id)
            if not state:
                return {"error": "Session not found"}
            
            # Update state with user responses
            state["user_responses"].extend(user_responses)
            
            # Continue from evaluation step
            result = await self.flow_graph.ainvoke(state)
            
            # Save updated state
            await self._save_session_state(session_id, result)
            
            return {
                "session_id": session_id,
                "status": result.get("completion_status", "in_progress"),
                "current_module": result.get("current_module", {}),
                "theory_content": result.get("theory_content", ""),
                "code_examples": result.get("code_examples", []),
                "quiz_questions": result.get("quiz_questions", []),
                "progress_metrics": result.get("progress_metrics", {}),
                "performance_data": result.get("performance_data", {})
            }
            
        except Exception as e:
            logger.error(f"Error continuing learning session: {str(e)}")
            return {"error": str(e)}
    
    # Node implementations for the learning flow graph
    async def _initialize_session_node(self, state: LearningFlowState) -> LearningFlowState:
        """Initialize the learning session with user and curriculum data."""
        try:
            # Initialize progress metrics
            state["progress_metrics"] = {
                "overall_progress": 0.0,
                "module_progress": 0.0,
                "quiz_accuracy": 0.0,
                "time_spent": 0.0
            }
            
            # Initialize performance tracking
            state["performance_data"] = {
                "correct_answers": 0,
                "total_questions": 0,
                "average_response_time": 0.0,
                "difficulty_level": "beginner"
            }
            
            if self.is_logging:
                logger.info(f"Session initialized: {state['session_id']}")
            
            return state
            
        except Exception as e:
            state["errors"].append(f"Session initialization error: {str(e)}")
            state["next_action"] = "error"
            return state
    
    async def _load_current_module_node(self, state: LearningFlowState) -> LearningFlowState:
        """Load the current module from the curriculum."""
        try:
            curriculum = state["curriculum"]
            module_index = state["module_index"]
            
            if "modules" in curriculum and module_index < len(curriculum["modules"]):
                current_module = curriculum["modules"][module_index]
                state["current_module"] = current_module
                
                if self.is_logging:
                    logger.info(f"Loaded module {module_index}: {current_module.get('title', 'Unknown')}")
            else:
                state["completion_status"] = "completed"
                state["next_action"] = "complete"
            
            return state
            
        except Exception as e:
            state["errors"].append(f"Module loading error: {str(e)}")
            state["next_action"] = "error"
            return state
    
    async def _generate_theory_content_node(self, state: LearningFlowState) -> LearningFlowState:
        """Generate theory content for the current module."""
        try:
            current_module = state["current_module"]
            framework_name = state["framework_name"]
            user_id = state["user_id"]
            
            # Generate theory content based on module and user progress
            theory_content = await self._generate_theory_content(
                module=current_module,
                framework=framework_name,
                user_id=user_id,
                difficulty_level=state["performance_data"].get("difficulty_level", "beginner")
            )
            
            state["theory_content"] = theory_content
            
            if self.is_logging:
                logger.info(f"Generated theory content for module: {current_module.get('title', 'Unknown')}")
            
            return state
            
        except Exception as e:
            state["errors"].append(f"Theory generation error: {str(e)}")
            state["next_action"] = "error"
            return state
    
    async def _generate_code_examples_node(self, state: LearningFlowState) -> LearningFlowState:
        """Generate code examples for the current module."""
        try:
            current_module = state["current_module"]
            framework_name = state["framework_name"]
            
            # Generate code examples
            code_examples = await self._generate_code_examples(
                module=current_module,
                framework=framework_name,
                theory_content=state["theory_content"]
            )
            
            state["code_examples"] = code_examples
            
            if self.is_logging:
                logger.info(f"Generated {len(code_examples)} code examples")
            
            return state
            
        except Exception as e:
            state["errors"].append(f"Code generation error: {str(e)}")
            state["next_action"] = "error"
            return state
    
    async def _present_content_node(self, state: LearningFlowState) -> LearningFlowState:
        """Present theory content and code examples to the user."""
        try:
            # This node prepares content for presentation
            # The actual presentation happens in the UI layer
            state["next_action"] = "present_to_user"
            
            if self.is_logging:
                logger.info("Content prepared for presentation")
            
            return state
            
        except Exception as e:
            state["errors"].append(f"Content presentation error: {str(e)}")
            state["next_action"] = "error"
            return state
    
    async def _generate_quiz_node(self, state: LearningFlowState) -> LearningFlowState:
        """Generate quiz questions for the current module."""
        try:
            current_module = state["current_module"]
            theory_content = state["theory_content"]
            code_examples = state["code_examples"]
            
            # Generate quiz questions
            quiz_questions = await self._generate_quiz_questions(
                module=current_module,
                theory_content=theory_content,
                code_examples=code_examples,
                difficulty_level=state["performance_data"].get("difficulty_level", "beginner")
            )
            
            state["quiz_questions"] = quiz_questions
            
            if self.is_logging:
                logger.info(f"Generated {len(quiz_questions)} quiz questions")
            
            return state
            
        except Exception as e:
            state["errors"].append(f"Quiz generation error: {str(e)}")
            state["next_action"] = "error"
            return state
    
    async def _evaluate_responses_node(self, state: LearningFlowState) -> LearningFlowState:
        """Evaluate user responses to quiz questions."""
        try:
            quiz_questions = state["quiz_questions"]
            user_responses = state["user_responses"]
            
            # Evaluate responses
            quiz_results = await self._evaluate_quiz_responses(
                questions=quiz_questions,
                responses=user_responses
            )
            
            state["current_quiz_results"] = quiz_results
            
            # Update performance data
            correct_answers = quiz_results.get("correct_count", 0)
            total_questions = quiz_results.get("total_questions", 1)
            
            state["performance_data"]["correct_answers"] += correct_answers
            state["performance_data"]["total_questions"] += total_questions
            
            accuracy = correct_answers / max(total_questions, 1)
            state["performance_data"]["quiz_accuracy"] = accuracy
            
            if self.is_logging:
                logger.info(f"Quiz evaluation completed. Accuracy: {accuracy:.2%}")
            
            return state
            
        except Exception as e:
            state["errors"].append(f"Response evaluation error: {str(e)}")
            state["next_action"] = "error"
            return state
    
    async def _update_progress_node(self, state: LearningFlowState) -> LearningFlowState:
        """Update learning progress based on quiz results."""
        try:
            quiz_results = state["current_quiz_results"]
            module_index = state["module_index"]
            total_modules = len(state["curriculum"].get("modules", []))
            
            # Update module progress
            module_completion = quiz_results.get("score", 0.0)
            state["progress_metrics"]["module_progress"] = module_completion
            
            # Update overall progress
            overall_progress = (module_index + module_completion) / max(total_modules, 1)
            state["progress_metrics"]["overall_progress"] = overall_progress
            
            if self.is_logging:
                logger.info(f"Progress updated. Overall: {overall_progress:.1%}, Module: {module_completion:.1%}")
            
            return state
            
        except Exception as e:
            state["errors"].append(f"Progress update error: {str(e)}")
            state["next_action"] = "error"
            return state
    
    async def _check_module_completion_node(self, state: LearningFlowState) -> LearningFlowState:
        """Check if current module is completed and determine next action."""
        try:
            quiz_accuracy = state["performance_data"]["quiz_accuracy"]
            module_progress = state["progress_metrics"]["module_progress"]
            
            # Define completion thresholds
            completion_threshold = 0.7  # 70% accuracy required
            mastery_threshold = 0.9    # 90% for excellent performance
            
            if quiz_accuracy >= mastery_threshold:
                state["next_action"] = "next_module"
            elif quiz_accuracy >= completion_threshold:
                state["next_action"] = "next_module"
            elif quiz_accuracy < 0.5:  # Below 50% needs difficulty adjustment
                state["requires_adjustment"] = True
                state["next_action"] = "adjust_difficulty"
            else:
                # Retry with same difficulty
                state["retry_count"] += 1
                if state["retry_count"] < 3:
                    state["next_action"] = "retry"
                else:
                    state["next_action"] = "adjust_difficulty"
            
            if self.is_logging:
                logger.info(f"Module completion check. Next action: {state['next_action']}")
            
            return state
            
        except Exception as e:
            state["errors"].append(f"Module completion check error: {str(e)}")
            state["next_action"] = "error"
            return state
    
    def _determine_next_action(self, state: LearningFlowState) -> str:
        """Determine the next action based on current state."""
        if state.get("errors"):
            return "error"
        
        if state["completion_status"] == "completed":
            return "complete"
        
        return state.get("next_action", "retry")
    
    async def _adjust_difficulty_node(self, state: LearningFlowState) -> LearningFlowState:
        """Adjust difficulty level based on performance."""
        try:
            current_accuracy = state["performance_data"]["quiz_accuracy"]
            current_difficulty = state["performance_data"]["difficulty_level"]
            
            # Adjust difficulty based on performance
            if current_accuracy < 0.3:
                new_difficulty = "beginner"
            elif current_accuracy < 0.6:
                new_difficulty = "intermediate" if current_difficulty == "advanced" else "beginner"
            else:
                new_difficulty = current_difficulty  # Keep same level
            
            state["performance_data"]["difficulty_level"] = new_difficulty
            state["requires_adjustment"] = False
            state["retry_count"] = 0
            
            if self.is_logging:
                logger.info(f"Difficulty adjusted to: {new_difficulty}")
            
            return state
            
        except Exception as e:
            state["errors"].append(f"Difficulty adjustment error: {str(e)}")
            state["next_action"] = "error"
            return state
    
    async def _advance_to_next_module_node(self, state: LearningFlowState) -> LearningFlowState:
        """Advance to the next module in the curriculum."""
        try:
            state["module_index"] += 1
            state["retry_count"] = 0
            state["user_responses"] = []  # Clear responses for new module
            
            # Check if we've completed all modules
            total_modules = len(state["curriculum"].get("modules", []))
            if state["module_index"] >= total_modules:
                state["completion_status"] = "completed"
                state["next_action"] = "complete"
            
            if self.is_logging:
                logger.info(f"Advanced to module {state['module_index']}")
            
            return state
            
        except Exception as e:
            state["errors"].append(f"Module advancement error: {str(e)}")
            state["next_action"] = "error"
            return state
    
    async def _complete_curriculum_node(self, state: LearningFlowState) -> LearningFlowState:
        """Complete the curriculum and finalize results."""
        try:
            state["completion_status"] = "completed"
            
            # Calculate final metrics
            final_accuracy = state["performance_data"]["quiz_accuracy"]
            overall_progress = state["progress_metrics"]["overall_progress"]
            
            if self.is_logging:
                logger.info(f"Curriculum completed. Final accuracy: {final_accuracy:.1%}, Progress: {overall_progress:.1%}")
            
            return state
            
        except Exception as e:
            state["errors"].append(f"Curriculum completion error: {str(e)}")
            state["next_action"] = "error"
            return state
    
    async def _handle_error_node(self, state: LearningFlowState) -> LearningFlowState:
        """Handle errors in the learning flow."""
        try:
            state["completion_status"] = "error"
            
            if self.is_logging:
                logger.error(f"Learning flow error: {state['errors']}")
            
            return state
            
        except Exception as e:
            state["errors"].append(f"Error handling error: {str(e)}")
            return state
    
    # Helper methods for content generation and evaluation
    async def _generate_theory_content(
        self,
        module: Dict[str, Any],
        framework: str,
        user_id: str,
        difficulty_level: str
    ) -> str:
        """Generate theory content for a module."""
        # This would integrate with your content generation system
        # For now, return a placeholder
        return f"Theory content for {module.get('title', 'Module')} at {difficulty_level} level"
    
    async def _generate_code_examples(
        self,
        module: Dict[str, Any],
        framework: str,
        theory_content: str
    ) -> List[Dict[str, str]]:
        """Generate code examples for a module."""
        # This would integrate with your code generation system
        # For now, return placeholder examples
        return [
            {
                "title": "Basic Example",
                "code": f"# Example code for {module.get('title', 'Module')}",
                "explanation": "This demonstrates the basic concepts"
            }
        ]
    
    async def _generate_quiz_questions(
        self,
        module: Dict[str, Any],
        theory_content: str,
        code_examples: List[Dict[str, str]],
        difficulty_level: str
    ) -> List[Dict[str, Any]]:
        """Generate quiz questions for a module."""
        # This would integrate with your quiz generation system
        # For now, return placeholder questions
        return [
            {
                "id": "q1",
                "question": f"What is the main concept in {module.get('title', 'this module')}?",
                "type": "multiple_choice",
                "options": ["Option A", "Option B", "Option C", "Option D"],
                "correct_answer": "Option A"
            }
        ]
    
    async def _evaluate_quiz_responses(
        self,
        questions: List[Dict[str, Any]],
        responses: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Evaluate user responses to quiz questions."""
        # This would integrate with your evaluation system
        # For now, return placeholder results
        correct_count = len(responses) // 2  # Assume 50% correct for placeholder
        total_questions = len(questions)
        
        return {
            "correct_count": correct_count,
            "total_questions": total_questions,
            "score": correct_count / max(total_questions, 1),
            "detailed_results": []
        }
    
    async def _load_session_state(self, session_id: str) -> Optional[LearningFlowState]:
        """Load session state from storage."""
        # This would integrate with your session storage system
        # For now, return None
        return None
    
    async def _save_session_state(self, session_id: str, state: LearningFlowState) -> None:
        """Save session state to storage."""
        # This would integrate with your session storage system
        pass 