"""State Extractor for RL System

This module provides state representation extraction for the GAAPF RL system,
converting complex agent and task information into structured state vectors.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Union
from enum import Enum
import json

class StateType(Enum):
    """Types of state representations."""
    VECTOR = "vector"
    DICT = "dict"
    GRAPH = "graph"
    SEQUENCE = "sequence"

class FeatureType(Enum):
    """Types of features to extract."""
    AGENT_CAPABILITIES = "agent_capabilities"
    TASK_REQUIREMENTS = "task_requirements"
    PERFORMANCE_METRICS = "performance_metrics"
    CONTEXT_INFORMATION = "context_information"
    TEMPORAL_FEATURES = "temporal_features"
    INTERACTION_HISTORY = "interaction_history"

@dataclass
class StateConfig:
    """Configuration for state extraction."""
    
    # Feature inclusion flags
    include_agent_states: bool = True
    include_task_context: bool = True
    include_performance_metrics: bool = True
    include_temporal_features: bool = False
    include_interaction_history: bool = False
    
    # State representation settings
    state_type: StateType = StateType.VECTOR
    vector_dim: int = 128
    normalize_features: bool = True
    
    # Feature extraction settings
    max_agents: int = 10
    max_history_length: int = 50
    capability_embedding_dim: int = 32
    task_embedding_dim: int = 32
    
    # Performance metrics settings
    performance_window: int = 10
    include_success_rate: bool = True
    include_efficiency_metrics: bool = True
    include_quality_metrics: bool = True
    
    # Normalization settings
    feature_scaling: str = "standard"  # "standard", "minmax", "robust"
    clip_outliers: bool = True
    outlier_threshold: float = 3.0

class StateExtractor:
    """Extracts state representations for RL agents."""
    
    def __init__(self, config: StateConfig):
        self.config = config
        self.feature_stats = {}
        self.initialized = False
        
    def initialize(self, sample_data: Optional[Dict[str, Any]] = None):
        """Initialize the state extractor with sample data for normalization."""
        if sample_data:
            self._compute_feature_stats(sample_data)
        self.initialized = True
        
    def extract_state(self, 
                     agents: List[Dict[str, Any]], 
                     task: Dict[str, Any],
                     context: Optional[Dict[str, Any]] = None,
                     history: Optional[List[Dict[str, Any]]] = None) -> Union[np.ndarray, Dict[str, Any]]:
        """Extract state representation from current situation."""
        
        features = {}
        
        # Extract agent features
        if self.config.include_agent_states:
            features.update(self._extract_agent_features(agents))
            
        # Extract task features
        if self.config.include_task_context:
            features.update(self._extract_task_features(task))
            
        # Extract performance features
        if self.config.include_performance_metrics and history:
            features.update(self._extract_performance_features(history))
            
        # Extract temporal features
        if self.config.include_temporal_features and history:
            features.update(self._extract_temporal_features(history))
            
        # Extract interaction features
        if self.config.include_interaction_history and history:
            features.update(self._extract_interaction_features(history))
            
        # Extract context features
        if context:
            features.update(self._extract_context_features(context))
            
        # Convert to desired format
        if self.config.state_type == StateType.VECTOR:
            return self._features_to_vector(features)
        else:
            return features
            
    def _extract_agent_features(self, agents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract features from agent information."""
        features = {}
        
        # Pad or truncate agent list
        padded_agents = agents[:self.config.max_agents]
        while len(padded_agents) < self.config.max_agents:
            padded_agents.append({})
            
        # Extract capabilities
        capabilities_matrix = []
        for i, agent in enumerate(padded_agents):
            agent_capabilities = self._encode_capabilities(
                agent.get('capabilities', [])
            )
            capabilities_matrix.append(agent_capabilities)
            
        features['agent_capabilities'] = np.array(capabilities_matrix)
        
        # Extract agent states
        agent_states = []
        for agent in padded_agents:
            state = [
                agent.get('load', 0.0),
                agent.get('performance_score', 0.0),
                agent.get('availability', 1.0),
                int(agent.get('active', False))
            ]
            agent_states.append(state)
            
        features['agent_states'] = np.array(agent_states)
        
        # Agent count and diversity
        features['num_agents'] = len(agents)
        features['agent_diversity'] = self._compute_agent_diversity(agents)
        
        return features
        
    def _extract_task_features(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Extract features from task information."""
        features = {}
        
        # Task requirements
        requirements = self._encode_requirements(
            task.get('requirements', [])
        )
        features['task_requirements'] = requirements
        
        # Task properties
        features['task_complexity'] = task.get('complexity', 0.5)
        features['task_priority'] = task.get('priority', 0.5)
        features['task_urgency'] = task.get('urgency', 0.5)
        features['estimated_duration'] = task.get('estimated_duration', 1.0)
        
        # Task type encoding
        task_type = task.get('type', 'general')
        features['task_type'] = self._encode_task_type(task_type)
        
        return features
        
    def _extract_performance_features(self, history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract performance-related features from history."""
        features = {}
        
        recent_history = history[-self.config.performance_window:]
        
        if self.config.include_success_rate:
            success_rates = [h.get('success_rate', 0.0) for h in recent_history]
            features['avg_success_rate'] = np.mean(success_rates) if success_rates else 0.0
            features['success_rate_trend'] = self._compute_trend(success_rates)
            
        if self.config.include_efficiency_metrics:
            efficiency_scores = [h.get('efficiency', 0.0) for h in recent_history]
            features['avg_efficiency'] = np.mean(efficiency_scores) if efficiency_scores else 0.0
            features['efficiency_trend'] = self._compute_trend(efficiency_scores)
            
        if self.config.include_quality_metrics:
            quality_scores = [h.get('quality', 0.0) for h in recent_history]
            features['avg_quality'] = np.mean(quality_scores) if quality_scores else 0.0
            features['quality_trend'] = self._compute_trend(quality_scores)
            
        return features
        
    def _extract_temporal_features(self, history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract temporal features from history."""
        features = {}
        
        if history:
            # Time since last action
            features['time_since_last_action'] = history[-1].get('timestamp', 0)
            
            # Action frequency
            timestamps = [h.get('timestamp', 0) for h in history]
            if len(timestamps) > 1:
                intervals = np.diff(timestamps)
                features['avg_action_interval'] = np.mean(intervals)
                features['action_frequency'] = 1.0 / np.mean(intervals) if np.mean(intervals) > 0 else 0.0
            else:
                features['avg_action_interval'] = 0.0
                features['action_frequency'] = 0.0
                
        return features
        
    def _extract_interaction_features(self, history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract interaction features from history."""
        features = {}
        
        recent_history = history[-self.config.max_history_length:]
        
        # Interaction patterns
        interaction_types = [h.get('interaction_type', 'unknown') for h in recent_history]
        features['interaction_diversity'] = len(set(interaction_types))
        
        # Collaboration patterns
        collaborations = [h.get('collaboration_score', 0.0) for h in recent_history]
        features['avg_collaboration'] = np.mean(collaborations) if collaborations else 0.0
        
        return features
        
    def _extract_context_features(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract features from context information."""
        features = {}
        
        # System load
        features['system_load'] = context.get('system_load', 0.0)
        features['resource_availability'] = context.get('resource_availability', 1.0)
        
        # Environmental factors
        features['time_of_day'] = context.get('time_of_day', 0.5)
        features['workload_pressure'] = context.get('workload_pressure', 0.0)
        
        return features
        
    def _encode_capabilities(self, capabilities: List[str]) -> np.ndarray:
        """Encode agent capabilities as a vector."""
        # Simple one-hot encoding for common capabilities
        common_capabilities = [
            'data_analysis', 'visualization', 'reporting', 'communication',
            'problem_solving', 'research', 'coding', 'testing', 'documentation',
            'project_management', 'quality_assurance', 'user_interface'
        ]
        
        encoding = np.zeros(len(common_capabilities))
        for i, cap in enumerate(common_capabilities):
            if cap in capabilities:
                encoding[i] = 1.0
                
        return encoding
        
    def _encode_requirements(self, requirements: List[str]) -> np.ndarray:
        """Encode task requirements as a vector."""
        # Similar to capabilities encoding
        return self._encode_capabilities(requirements)
        
    def _encode_task_type(self, task_type: str) -> np.ndarray:
        """Encode task type as a vector."""
        task_types = [
            'analysis', 'development', 'testing', 'documentation',
            'research', 'communication', 'planning', 'review'
        ]
        
        encoding = np.zeros(len(task_types))
        if task_type in task_types:
            encoding[task_types.index(task_type)] = 1.0
            
        return encoding
        
    def _compute_agent_diversity(self, agents: List[Dict[str, Any]]) -> float:
        """Compute diversity score for agent team."""
        if not agents:
            return 0.0
            
        all_capabilities = set()
        for agent in agents:
            all_capabilities.update(agent.get('capabilities', []))
            
        # Diversity based on unique capabilities
        return len(all_capabilities) / max(len(agents), 1)
        
    def _compute_trend(self, values: List[float]) -> float:
        """Compute trend (slope) of values."""
        if len(values) < 2:
            return 0.0
            
        x = np.arange(len(values))
        y = np.array(values)
        
        # Simple linear regression slope
        slope = np.polyfit(x, y, 1)[0] if len(values) > 1 else 0.0
        return slope
        
    def _features_to_vector(self, features: Dict[str, Any]) -> np.ndarray:
        """Convert feature dictionary to a single vector."""
        vector_parts = []
        
        for key, value in features.items():
            if isinstance(value, np.ndarray):
                vector_parts.append(value.flatten())
            elif isinstance(value, (int, float)):
                vector_parts.append(np.array([value]))
            elif isinstance(value, list):
                vector_parts.append(np.array(value))
                
        if vector_parts:
            full_vector = np.concatenate(vector_parts)
            
            # Pad or truncate to desired dimension
            if len(full_vector) > self.config.vector_dim:
                full_vector = full_vector[:self.config.vector_dim]
            elif len(full_vector) < self.config.vector_dim:
                padding = np.zeros(self.config.vector_dim - len(full_vector))
                full_vector = np.concatenate([full_vector, padding])
                
            # Normalize if requested
            if self.config.normalize_features:
                full_vector = self._normalize_vector(full_vector)
                
            return full_vector
        else:
            return np.zeros(self.config.vector_dim)
            
    def _normalize_vector(self, vector: np.ndarray) -> np.ndarray:
        """Normalize vector based on configuration."""
        if self.config.feature_scaling == "standard":
            mean = np.mean(vector)
            std = np.std(vector)
            if std > 0:
                vector = (vector - mean) / std
        elif self.config.feature_scaling == "minmax":
            min_val = np.min(vector)
            max_val = np.max(vector)
            if max_val > min_val:
                vector = (vector - min_val) / (max_val - min_val)
        elif self.config.feature_scaling == "robust":
            median = np.median(vector)
            mad = np.median(np.abs(vector - median))
            if mad > 0:
                vector = (vector - median) / mad
                
        # Clip outliers if requested
        if self.config.clip_outliers:
            vector = np.clip(vector, -self.config.outlier_threshold, self.config.outlier_threshold)
            
        return vector
        
    def _compute_feature_stats(self, sample_data: Dict[str, Any]):
        """Compute feature statistics for normalization."""
        # This would be used for more sophisticated normalization
        # based on training data statistics
        pass
        
    def get_state_info(self) -> Dict[str, Any]:
        """Get information about the state extractor."""
        return {
            'config': self.config.__dict__,
            'initialized': self.initialized,
            'state_type': self.config.state_type.value,
            'vector_dim': self.config.vector_dim
        }