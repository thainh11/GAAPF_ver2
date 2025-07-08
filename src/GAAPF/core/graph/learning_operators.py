"""
Learning Operators for GAAPF Architecture

This module provides specialized operators for learning workflows
in the GAAPF architecture.
"""

import logging
from typing import Dict, List, Optional, Any, Callable
import copy

from .operator import Operator

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LearningOperator(Operator):
    """Base class for learning operators"""
    
    def __init__(self, name: str = None, description: str = None):
        """
        Initialize a learning operator.
        
        Parameters:
        ----------
        name : str, optional
            Name of the operator
        description : str, optional
            Description of the operator
        """
        self.name = name or self.__class__.__name__
        self.description = description or "Learning operator"
    
    def __call__(self, state: Dict) -> Dict:
        """
        Apply the operator to the state.
        
        Parameters:
        ----------
        state : Dict
            Current state
            
        Returns:
        -------
        Dict
            Updated state
        """
        # Base implementation just returns the state unchanged
        return state

class ConceptExtractionOperator(LearningOperator):
    """
    Extracts concepts from agent responses.
    
    This operator analyzes agent responses to identify and extract
    key concepts mentioned in the content.
    """
    
    def __init__(self, concept_list: List[str] = None):
        """
        Initialize a concept extraction operator.
        
        Parameters:
        ----------
        concept_list : List[str], optional
            List of known concepts to look for
        """
        super().__init__(
            name="ConceptExtraction",
            description="Extracts concepts from agent responses"
        )
        self.concept_list = concept_list or []
    
    def __call__(self, state: Dict) -> Dict:
        """
        Extract concepts from agent responses.
        
        Parameters:
        ----------
        state : Dict
            Current state with agent responses
            
        Returns:
        -------
        Dict
            State updated with extracted concepts
        """
        # Create a copy of the state
        new_state = copy.deepcopy(state)
        
        # Initialize concepts if not present
        if "extracted_concepts" not in new_state:
            new_state["extracted_concepts"] = []
        
        # Get agent responses
        agent_responses = new_state.get("agent_responses", {})
        
        # Extract concepts from each response
        for agent_type, response in agent_responses.items():
            content = response.get("content", "")
            concepts = self._extract_concepts(content)
            
            # Add to extracted concepts
            for concept in concepts:
                if concept not in new_state["extracted_concepts"]:
                    new_state["extracted_concepts"].append(concept)
        
        return new_state
    
    def _extract_concepts(self, content: str) -> List[str]:
        """
        Extract concepts from content.
        
        Parameters:
        ----------
        content : str
            Content to extract concepts from
            
        Returns:
        -------
        List[str]
            Extracted concepts
        """
        extracted = []
        
        # Simple extraction based on known concepts
        content_lower = content.lower()
        for concept in self.concept_list:
            if concept.lower() in content_lower:
                extracted.append(concept)
        
        # TODO: Add more sophisticated concept extraction
        # This could use NLP techniques to identify concepts
        
        return extracted

class LearningProgressOperator(LearningOperator):
    """
    Updates learning progress based on interaction.
    
    This operator analyzes the interaction and updates
    the learning progress in the learning context.
    """
    
    def __init__(self):
        """Initialize a learning progress operator."""
        super().__init__(
            name="LearningProgress",
            description="Updates learning progress based on interaction"
        )
    
    def __call__(self, state: Dict) -> Dict:
        """
        Update learning progress.
        
        Parameters:
        ----------
        state : Dict
            Current state
            
        Returns:
        -------
        Dict
            State with updated learning context
        """
        # Create a copy of the state
        new_state = copy.deepcopy(state)
        
        # Get learning context
        learning_context = new_state.get("learning_context", {})
        
        # Get user profile
        user_profile = learning_context.get("user_profile", {})
        
        # Get current module and interaction data
        current_module = learning_context.get("current_module", "")
        interaction_data = new_state.get("interaction_data", {})
        interaction_type = interaction_data.get("type", "general")
        
        # Get agent responses
        agent_responses = new_state.get("agent_responses", {})
        
        # Update learning history
        learning_history = user_profile.get("learning_history", [])
        
        # Create a new history entry
        history_entry = {
            "module": current_module,
            "interaction_type": interaction_type,
            "timestamp": self._get_timestamp(),
            "concepts": new_state.get("extracted_concepts", [])
        }
        
        # Add to history
        learning_history.append(history_entry)
        
        # Update user profile
        user_profile["learning_history"] = learning_history
        
        # Update learning context
        learning_context["user_profile"] = user_profile
        new_state["learning_context"] = learning_context
        
        return new_state
    
    def _get_timestamp(self) -> int:
        """Get current timestamp"""
        import time
        return int(time.time())

class AssessmentOperator(LearningOperator):
    """
    Processes assessment interactions and updates learning context.
    
    This operator handles assessment-specific interactions and
    updates the learning context with assessment results.
    """
    
    def __init__(self):
        """Initialize an assessment operator."""
        super().__init__(
            name="Assessment",
            description="Processes assessment interactions"
        )
    
    def __call__(self, state: Dict) -> Dict:
        """
        Process assessment interaction.
        
        Parameters:
        ----------
        state : Dict
            Current state
            
        Returns:
        -------
        Dict
            State with updated learning context
        """
        # Create a copy of the state
        new_state = copy.deepcopy(state)
        
        # Check if this is an assessment interaction
        interaction_data = new_state.get("interaction_data", {})
        if interaction_data.get("type") != "assessment":
            # Not an assessment, return unchanged
            return new_state
        
        # Get learning context
        learning_context = new_state.get("learning_context", {})
        
        # Get user profile
        user_profile = learning_context.get("user_profile", {})
        
        # Get current module
        current_module = learning_context.get("current_module", "")
        
        # Get assessment data
        assessment_data = interaction_data.get("assessment_data", {})
        score = assessment_data.get("score")
        
        if score is not None:
            # Update assessment history
            assessment_history = user_profile.get("assessment_history", {})
            
            # Initialize module assessments if not present
            if current_module not in assessment_history:
                assessment_history[current_module] = []
            
            # Add assessment result
            assessment_entry = {
                "timestamp": self._get_timestamp(),
                "score": score,
                "details": assessment_data.get("details", {})
            }
            
            assessment_history[current_module].append(assessment_entry)
            
            # Update user profile
            user_profile["assessment_history"] = assessment_history
            
            # Check if module is completed
            if score >= 80:  # Assuming 80% is passing
                completed_modules = user_profile.get("completed_modules", [])
                if current_module not in completed_modules:
                    completed_modules.append(current_module)
                    user_profile["completed_modules"] = completed_modules
            
            # Update learning context
            learning_context["user_profile"] = user_profile
            new_state["learning_context"] = learning_context
        
        return new_state
    
    def _get_timestamp(self) -> int:
        """Get current timestamp"""
        import time
        return int(time.time())

class HandoffOperator(LearningOperator):
    """
    Determines if a handoff to another agent is needed.
    
    This operator analyzes agent responses and interaction
    to determine if a handoff to another agent is needed.
    """
    
    def __init__(self, agent_types: List[str]):
        """
        Initialize a handoff operator.
        
        Parameters:
        ----------
        agent_types : List[str]
            List of available agent types
        """
        super().__init__(
            name="Handoff",
            description="Determines if a handoff to another agent is needed"
        )
        self.agent_types = agent_types
    
    def __call__(self, state: Dict) -> Dict:
        """
        Determine if handoff is needed.
        
        Parameters:
        ----------
        state : Dict
            Current state
            
        Returns:
        -------
        Dict
            State updated with handoff information
        """
        # Create a copy of the state
        new_state = copy.deepcopy(state)
        
        # Get current agent and response
        current_agent = new_state.get("current_agent", "")
        agent_responses = new_state.get("agent_responses", {})
        response = agent_responses.get(current_agent, {})
        
        # Check for explicit handoff request
        handoff_to = response.get("handoff_to")
        if handoff_to and handoff_to in self.agent_types and handoff_to != current_agent:
            new_state["handoff_needed"] = True
            new_state["handoff_to"] = handoff_to
            new_state["handoff_reason"] = response.get("handoff_reason", "Agent requested handoff")
            return new_state
        
        # Check for implicit handoff based on content
        content = response.get("content", "").lower()
        
        # Define handoff rules
        handoff_rules = [
            {
                "keywords": ["code", "implementation", "example", "syntax"],
                "agent": "code_assistant",
                "reason": "Code implementation needed"
            },
            {
                "keywords": ["error", "bug", "fix", "issue", "problem"],
                "agent": "troubleshooter",
                "reason": "Error resolution needed"
            },
            {
                "keywords": ["document", "reference", "api", "specification"],
                "agent": "documentation_expert",
                "reason": "Documentation reference needed"
            },
            {
                "keywords": ["practice", "exercise", "challenge", "try"],
                "agent": "practice_facilitator",
                "reason": "Practice exercises needed"
            },
            {
                "keywords": ["assess", "evaluate", "test", "quiz"],
                "agent": "assessment",
                "reason": "Assessment needed"
            },
            {
                "keywords": ["connect", "integrate", "synthesize", "relate"],
                "agent": "knowledge_synthesizer",
                "reason": "Knowledge synthesis needed"
            }
        ]
        
        # Check each rule
        for rule in handoff_rules:
            agent_type = rule["agent"]
            # Skip if this is the current agent or not available
            if agent_type == current_agent or agent_type not in self.agent_types:
                continue
            
            # Check if any keywords match
            if any(keyword in content for keyword in rule["keywords"]):
                new_state["handoff_needed"] = True
                new_state["handoff_to"] = agent_type
                new_state["handoff_reason"] = rule["reason"]
                return new_state
        
        # No handoff needed
        new_state["handoff_needed"] = False
        
        return new_state

def create_learning_operator_chain(
    agent_types: List[str],
    concept_list: List[str] = None
) -> Callable:
    """
    Create a chain of learning operators.
    
    Parameters:
    ----------
    agent_types : List[str]
        List of available agent types
    concept_list : List[str], optional
        List of known concepts
        
    Returns:
    -------
    Callable
        Function that applies the chain of operators
    """
    # Create operators
    concept_extraction = ConceptExtractionOperator(concept_list)
    learning_progress = LearningProgressOperator()
    assessment = AssessmentOperator()
    handoff = HandoffOperator(agent_types)
    
    def operator_chain(state: Dict) -> Dict:
        """Apply the chain of operators"""
        # Apply each operator in sequence
        state = concept_extraction(state)
        state = assessment(state)
        state = learning_progress(state)
        state = handoff(state)
        
        return state
    
    return operator_chain 