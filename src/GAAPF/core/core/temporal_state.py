# Creating the Temporal State Manager

"""
Temporal State Manager for GAAPF Architecture

This module provides the TemporalState class that tracks learning
effectiveness over time and optimizes constellation selection.
"""

import logging
from typing import Dict, List, Optional, Any
import time
import copy

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TemporalState:
    """
    Tracks learning effectiveness and optimizes constellation selection.
    
    The TemporalState class is responsible for:
    1. Tracking patterns in learning effectiveness
    2. Optimizing constellation selection based on historical data
    3. Detecting changes in learning context that require constellation updates
    4. Providing temporal insights to improve the learning experience
    """
    
    def __init__(self, is_logging: bool = False, *args, **kwargs):
        """
        Initialize the TemporalState.
        
        Parameters:
        ----------
        is_logging : bool, optional
            Flag to enable detailed logging
        """
        self.is_logging = is_logging
        
        # Store historical effectiveness data
        self.effectiveness_history = {}
        
        # Store learning stage transition patterns
        self.stage_transitions = {}
        
        # Store activity transition patterns
        self.activity_transitions = {}
        
        if self.is_logging:
            logger.info("TemporalState initialized")
    
    def process_interaction(self, learning_context: Dict, interaction_data: Dict) -> Dict:
        """
        Process an interaction to update temporal state.
        
        Parameters:
        ----------
        learning_context : Dict
            Current learning context
        interaction_data : Dict
            Data about the interaction
            
        Returns:
        -------
        Dict
            Updates to apply to learning context
        """
        # Create a copy of the learning context
        context_updates = {}
        
        # Extract user ID
        user_id = learning_context.get("user_id", "default_user")
        
        # Check for learning stage transitions
        stage_update = self._check_stage_transition(learning_context, interaction_data)
        if stage_update:
            context_updates["learning_stage"] = stage_update
            context_updates["learning_stage_changed"] = True
            
            # Record transition
            self._record_stage_transition(
                user_id,
                learning_context.get("learning_stage"),
                stage_update
            )
        
        # Check for activity transitions
        activity_update = self._check_activity_transition(learning_context, interaction_data)
        if activity_update:
            context_updates["current_activity"] = activity_update
            context_updates["current_activity_changed"] = True
            
            # Record transition
            self._record_activity_transition(
                user_id,
                learning_context.get("current_activity"),
                activity_update
            )
        
        # Check for module transitions
        module_update = self._check_module_transition(learning_context, interaction_data)
        if module_update:
            context_updates["current_module"] = module_update
            context_updates["current_module_changed"] = True
        
        # Add temporal insights
        insights = self._generate_temporal_insights(learning_context, interaction_data)
        if insights:
            context_updates["temporal_insights"] = insights
        
        return context_updates
    
    def get_optimal_constellation_type(self, user_id: str, learning_context: Dict) -> str:
        """
        Get the optimal constellation type based on historical effectiveness.
        
        Parameters:
        ----------
        user_id : str
            Identifier for the user
        learning_context : Dict
            Current learning context
            
        Returns:
        -------
        str
            Optimal constellation type
        """
        # Get current learning stage and activity
        learning_stage = learning_context.get("learning_stage", "exploration")
        current_activity = learning_context.get("current_activity", "introduction")
        
        # Default mappings if no history
        stage_to_constellation = {
            "exploration": "learning",
            "concept": "learning",
            "practice": "practice",
            "assessment": "assessment",
            "project": "project",
            "troubleshooting": "troubleshooting"
        }
        
        activity_to_constellation = {
            "introduction": "learning",
            "concept_learning": "learning",
            "coding": "practice",
            "practice": "practice",
            "assessment": "assessment",
            "project": "project",
            "debugging": "troubleshooting"
        }
        
        # Try activity mapping first, then stage mapping
        if current_activity in activity_to_constellation:
            return activity_to_constellation[current_activity]
        elif learning_stage in stage_to_constellation:
            return stage_to_constellation[learning_stage]
        else:
            return "learning"  # Default
    
    def record_effectiveness(
        self,
        user_id: str,
        learning_context: Dict,
        constellation_type: str,
        effectiveness_score: float
    ) -> None:
        """
        Record the effectiveness of a constellation for a learning context.
        
        Parameters:
        ----------
        user_id : str
            Identifier for the user
        learning_context : Dict
            Learning context
        constellation_type : str
            Type of constellation
        effectiveness_score : float
            Score indicating how effective the constellation was (0-1)
        """
        # Get current learning stage and activity
        learning_stage = learning_context.get("learning_stage", "exploration")
        current_activity = learning_context.get("current_activity", "introduction")
        
        # Initialize user history if not present
        if user_id not in self.effectiveness_history:
            self.effectiveness_history[user_id] = {}
        
        user_history = self.effectiveness_history[user_id]
        
        # Initialize stage history if not present
        if learning_stage not in user_history:
            user_history[learning_stage] = {}
        
        # Initialize activity history if not present
        if current_activity not in user_history:
            user_history[current_activity] = {}
        
        # Record effectiveness
        timestamp = int(time.time())
        effectiveness_entry = {
            "constellation_type": constellation_type,
            "effectiveness_score": effectiveness_score,
            "timestamp": timestamp,
            "learning_context": copy.deepcopy(learning_context)
        }
        
        # Store under both stage and activity
        if constellation_type not in user_history[learning_stage]:
            user_history[learning_stage][constellation_type] = []
        if constellation_type not in user_history[current_activity]:
            user_history[current_activity][constellation_type] = []
        
        user_history[learning_stage][constellation_type].append(effectiveness_entry)
        user_history[current_activity][constellation_type].append(effectiveness_entry)
        
        if self.is_logging:
            logger.info(f"Recorded effectiveness score {effectiveness_score} for {constellation_type} constellation")
    
    def _check_stage_transition(self, learning_context: Dict, interaction_data: Dict) -> Optional[str]:
        """
        Check if a learning stage transition should occur.
        
        Parameters:
        ----------
        learning_context : Dict
            Current learning context
        interaction_data : Dict
            Data about the interaction
            
        Returns:
        -------
        Optional[str]
            New learning stage if transition should occur
        """
        current_stage = learning_context.get("learning_stage", "exploration")
        interaction_type = interaction_data.get("type", "general")
        query = interaction_data.get("query", "").lower()
        
        # Simple stage transition logic
        if current_stage == "exploration":
            if any(keyword in query for keyword in ["practice", "exercise", "code", "implement"]):
                return "practice"
            elif any(keyword in query for keyword in ["test", "quiz", "assessment", "evaluate"]):
                return "assessment"
            elif any(keyword in query for keyword in ["project", "build", "create", "develop"]):
                return "project"
        
        elif current_stage == "concept":
            if any(keyword in query for keyword in ["practice", "exercise", "try", "do"]):
                return "practice"
            elif any(keyword in query for keyword in ["help", "stuck", "error", "problem", "debug"]):
                return "troubleshooting"
        
        elif current_stage == "practice":
            if any(keyword in query for keyword in ["concept", "theory", "explain", "understand"]):
                return "concept"
            elif any(keyword in query for keyword in ["test", "assess", "check", "evaluate"]):
                return "assessment"
            elif any(keyword in query for keyword in ["error", "bug", "problem", "help"]):
                return "troubleshooting"
        
        elif current_stage == "assessment":
            if any(keyword in query for keyword in ["practice", "more", "again", "exercise"]):
                return "practice"
            elif any(keyword in query for keyword in ["review", "concept", "theory"]):
                return "concept"
        
        return None  # No transition
    
    def _check_activity_transition(self, learning_context: Dict, interaction_data: Dict) -> Optional[str]:
        """
        Check if an activity transition should occur.
        
        Parameters:
        ----------
        learning_context : Dict
            Current learning context
        interaction_data : Dict
            Data about the interaction
            
        Returns:
        -------
        Optional[str]
            New activity if transition should occur
        """
        current_activity = learning_context.get("current_activity", "introduction")
        query = interaction_data.get("query", "").lower()
        
        # Simple activity transition logic
        if current_activity == "introduction":
            if any(keyword in query for keyword in ["concept", "learn", "understand", "explain"]):
                return "concept_learning"
            elif any(keyword in query for keyword in ["code", "practice", "exercise", "implement"]):
                return "coding"
        
        elif current_activity == "concept_learning":
            if any(keyword in query for keyword in ["practice", "code", "implement", "try"]):
                return "coding"
            elif any(keyword in query for keyword in ["test", "quiz", "assess"]):
                return "assessment"
        
        elif current_activity == "coding":
            if any(keyword in query for keyword in ["error", "bug", "problem", "debug", "help"]):
                return "debugging"
            elif any(keyword in query for keyword in ["concept", "theory", "explain"]):
                return "concept_learning"
            elif any(keyword in query for keyword in ["project", "build", "create"]):
                return "project"
        
        elif current_activity == "debugging":
            if any(keyword in query for keyword in ["practice", "continue", "more"]):
                return "coding"
            elif any(keyword in query for keyword in ["concept", "understand", "explain"]):
                return "concept_learning"
        
        return None  # No transition
    
    def _check_module_transition(self, learning_context: Dict, interaction_data: Dict) -> Optional[str]:
        """
        Check if a module transition should occur.
        
        Parameters:
        ----------
        learning_context : Dict
            Current learning context
        interaction_data : Dict
            Data about the interaction
            
        Returns:
        -------
        Optional[str]
            New module if transition should occur
        """
        # This is a placeholder - in a real implementation, this would
        # use more sophisticated logic to detect module transitions
        query = interaction_data.get("query", "").lower()
        
        # Check for explicit module mentions
        modules = ["introduction", "basics", "intermediate", "advanced", "project"]
        for module in modules:
            if module in query:
                return module
        
        return None  # No transition
    
    def _record_stage_transition(self, user_id: str, from_stage: str, to_stage: str) -> None:
        """
        Record a learning stage transition.
        
        Parameters:
        ----------
        user_id : str
            Identifier for the user
        from_stage : str
            Previous learning stage
        to_stage : str
            New learning stage
        """
        if user_id not in self.stage_transitions:
            self.stage_transitions[user_id] = []
        
        transition_entry = {
            "from_stage": from_stage,
            "to_stage": to_stage,
            "timestamp": int(time.time())
        }
        
        self.stage_transitions[user_id].append(transition_entry)
        
        if self.is_logging:
            logger.info(f"Recorded stage transition: {from_stage} -> {to_stage}")
    
    def _record_activity_transition(self, user_id: str, from_activity: str, to_activity: str) -> None:
        """
        Record an activity transition.
        
        Parameters:
        ----------
        user_id : str
            Identifier for the user
        from_activity : str
            Previous activity
        to_activity : str
            New activity
        """
        if user_id not in self.activity_transitions:
            self.activity_transitions[user_id] = []
        
        transition_entry = {
            "from_activity": from_activity,
            "to_activity": to_activity,
            "timestamp": int(time.time())
        }
        
        self.activity_transitions[user_id].append(transition_entry)
        
        if self.is_logging:
            logger.info(f"Recorded activity transition: {from_activity} -> {to_activity}")
    
    def _generate_temporal_insights(self, learning_context: Dict, interaction_data: Dict) -> Dict:
        """
        Generate temporal insights based on patterns.
        
        Parameters:
        ----------
        learning_context : Dict
            Current learning context
        interaction_data : Dict
            Data about the interaction
            
        Returns:
        -------
        Dict
            Temporal insights
        """
        insights = {}
        
        # Get user ID
        user_id = learning_context.get("user_id", "default_user")
        
        # Analyze stage transition patterns
        if user_id in self.stage_transitions:
            transitions = self.stage_transitions[user_id]
            if len(transitions) > 1:
                # Find most common transitions
                transition_counts = {}
                for transition in transitions:
                    key = f"{transition['from_stage']}->{transition['to_stage']}"
                    transition_counts[key] = transition_counts.get(key, 0) + 1
                
                if transition_counts:
                    most_common = max(transition_counts, key=transition_counts.get)
                    insights["common_stage_transition"] = most_common
        
        # Analyze activity transition patterns
        if user_id in self.activity_transitions:
            transitions = self.activity_transitions[user_id]
            if len(transitions) > 1:
                # Find most common transitions
                transition_counts = {}
                for transition in transitions:
                    key = f"{transition['from_activity']}->{transition['to_activity']}"
                    transition_counts[key] = transition_counts.get(key, 0) + 1
                
                if transition_counts:
                    most_common = max(transition_counts, key=transition_counts.get)
                    insights["common_activity_transition"] = most_common
        
        return insights 