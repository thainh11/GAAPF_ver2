"""RL-Enhanced Specialized Agent for GAAPF Dual-System Implementation

This module provides an RL-enhanced version of the SpecializedAgent that
integrates reinforcement learning capabilities while maintaining compatibility
with the existing GAAPF architecture.
"""

import numpy as np
import time
import uuid
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import logging
from dataclasses import asdict

# Import base classes
try:
    from src.GAAPF.core.agents import SpecializedAgent
    from langchain_core.language_models.base import BaseLanguageModel
    from langchain_core.tools import BaseTool
except ImportError:
    # Fallback for development
    from abc import ABC, abstractmethod
    
    class SpecializedAgent(ABC):
        def __init__(self, *args, **kwargs):
            pass
    
    class BaseLanguageModel(ABC):
        pass
    
    class BaseTool(ABC):
        pass

# Import RL components
from ..utils.reward_system import RewardCalculator, RewardType
from ..utils.experience_buffer import Experience, ExperienceBuffer

logger = logging.getLogger(__name__)

class RLSpecializedAgent(SpecializedAgent):
    """RL-enhanced specialized agent with learning capabilities"""
    
    def __init__(self,
                 llm: BaseLanguageModel,
                 tools: List[Union[str, BaseTool]] = [],
                 memory_path: Optional[Path] = None,
                 config: Dict = None,
                 agent_type: str = "rl_specialized",
                 description: str = "RL-enhanced specialized learning agent",
                 is_logging: bool = False,
                 rl_config: Optional[Dict] = None,
                 reward_calculator: Optional[RewardCalculator] = None,
                 experience_buffer: Optional[ExperienceBuffer] = None,
                 *args, **kwargs):
        """
        Initialize RL-enhanced specialized agent.
        
        Parameters:
        ----------
        llm : BaseLanguageModel
            Language model for the agent
        tools : List[Union[str, BaseTool]]
            Tools available to the agent
        memory_path : Optional[Path]
            Path to agent memory
        config : Dict
            Agent configuration
        agent_type : str
            Type of the agent
        description : str
            Agent description
        is_logging : bool
            Enable logging
        rl_config : Optional[Dict]
            RL-specific configuration
        reward_calculator : Optional[RewardCalculator]
            Reward calculation system
        experience_buffer : Optional[ExperienceBuffer]
            Experience storage system
        """
        # Initialize base agent
        super().__init__(
            llm=llm,
            tools=tools,
            memory_path=memory_path,
            config=config,
            agent_type=agent_type,
            description=description,
            is_logging=is_logging,
            *args, **kwargs
        )
        
        # RL-specific initialization
        self.rl_config = rl_config or {}
        self.reward_calculator = reward_calculator
        self.experience_buffer = experience_buffer
        
        # RL state management
        self.current_state = {}
        self.last_action = {}
        self.current_episode_id = None
        self.step_count = 0
        self.episode_start_time = None
        
        # Learning parameters
        self.learning_rate = self.rl_config.get('learning_rate', 0.001)
        self.exploration_rate = self.rl_config.get('exploration_rate', 0.1)
        self.exploration_decay = self.rl_config.get('exploration_decay', 0.995)
        self.min_exploration_rate = self.rl_config.get('min_exploration_rate', 0.01)
        
        # Performance tracking
        self.performance_metrics = {
            'total_interactions': 0,
            'successful_interactions': 0,
            'average_reward': 0.0,
            'learning_efficiency': 0.0,
            'adaptation_score': 0.0
        }
        
        # RL-specific capabilities
        self.rl_capabilities = [
            "reinforcement_learning",
            "adaptive_behavior",
            "experience_replay",
            "reward_optimization",
            "continuous_improvement"
        ]
        
        if self.is_logging:
            logger.info(f"Initialized RL-enhanced {agent_type} agent with RL config: {self.rl_config}")
    
    def start_episode(self, initial_state: Dict[str, Any]) -> str:
        """Start a new learning episode"""
        self.current_episode_id = str(uuid.uuid4())
        self.current_state = initial_state.copy()
        self.step_count = 0
        self.episode_start_time = time.time()
        
        if self.is_logging:
            logger.info(f"Started new episode {self.current_episode_id} for agent {self.agent_type}")
        
        return self.current_episode_id
    
    def _extract_state_features(self, learning_context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract state features from learning context"""
        user_profile = learning_context.get("user_profile", {})
        framework_config = learning_context.get("framework_config", {})
        progress_metrics = learning_context.get("progress_metrics", {})
        conversation_analysis = learning_context.get("conversation_analysis", {})
        
        state_features = {
            # User characteristics
            "user_level": self._encode_user_level(user_profile.get("experience_level", "beginner")),
            "user_engagement": conversation_analysis.get("engagement_score", 0.5),
            "user_satisfaction": user_profile.get("satisfaction_score", 0.5),
            
            # Learning context
            "current_module": learning_context.get("current_module", ""),
            "learning_stage": learning_context.get("learning_stage", "exploration"),
            "interaction_count": learning_context.get("interaction_count", 0),
            
            # Progress indicators
            "module_progress": progress_metrics.get("current_module_progress", 0.0),
            "overall_progress": progress_metrics.get("completion_percentage", 0.0),
            "knowledge_retention": progress_metrics.get("knowledge_retention", 0.5),
            
            # Conversation context
            "conversation_depth": conversation_analysis.get("conversation_depth", "surface"),
            "recent_topics": len(conversation_analysis.get("recent_interaction_topics", [])),
            "learning_momentum": conversation_analysis.get("learning_momentum", "starting"),
            
            # Framework context
            "framework_complexity": framework_config.get("complexity_level", 0.5),
            "available_tools": len(self.tools),
            
            # Agent state
            "agent_confidence": getattr(self, 'confidence_score', 0.5),
            "exploration_rate": self.exploration_rate,
            "step_count": self.step_count
        }
        
        return state_features
    
    def _encode_user_level(self, level: str) -> float:
        """Encode user experience level as numerical value"""
        level_mapping = {
            "beginner": 0.0,
            "intermediate": 0.5,
            "advanced": 1.0
        }
        return level_mapping.get(level.lower(), 0.0)
    
    def _select_action(self, state: Dict[str, Any], query: str) -> Dict[str, Any]:
        """Select action using epsilon-greedy strategy with RL enhancements"""
        # Exploration vs exploitation decision
        if np.random.random() < self.exploration_rate:
            # Exploration: try different approaches
            action = self._explore_action(state, query)
            action_type = "exploration"
        else:
            # Exploitation: use learned policy
            action = self._exploit_action(state, query)
            action_type = "exploitation"
        
        # Add action metadata
        action.update({
            "action_type": action_type,
            "exploration_rate": self.exploration_rate,
            "timestamp": time.time(),
            "step_count": self.step_count
        })
        
        return action
    
    def _explore_action(self, state: Dict[str, Any], query: str) -> Dict[str, Any]:
        """Exploration action selection"""
        # Different exploration strategies based on context
        user_level = state.get("user_level", 0.0)
        conversation_depth = state.get("conversation_depth", "surface")
        
        exploration_strategies = [
            "detailed_explanation",
            "interactive_questioning",
            "practical_examples",
            "conceptual_connections",
            "progressive_disclosure"
        ]
        
        # Select strategy based on context
        if user_level < 0.3:  # Beginner
            preferred_strategies = ["detailed_explanation", "practical_examples"]
        elif user_level > 0.7:  # Advanced
            preferred_strategies = ["conceptual_connections", "interactive_questioning"]
        else:  # Intermediate
            preferred_strategies = exploration_strategies
        
        selected_strategy = np.random.choice(preferred_strategies)
        
        return {
            "strategy": selected_strategy,
            "approach": "exploratory",
            "adaptation_level": np.random.uniform(0.3, 0.8),
            "interaction_style": np.random.choice(["supportive", "challenging", "collaborative"])
        }
    
    def _exploit_action(self, state: Dict[str, Any], query: str) -> Dict[str, Any]:
        """Exploitation action selection using learned policy"""
        # Use learned preferences and successful patterns
        user_level = state.get("user_level", 0.0)
        user_engagement = state.get("user_engagement", 0.5)
        learning_momentum = state.get("learning_momentum", "starting")
        
        # Select optimal strategy based on learned patterns
        if user_engagement > 0.7 and learning_momentum == "accelerating":
            strategy = "progressive_disclosure"
            approach = "challenging"
        elif user_level < 0.3:
            strategy = "detailed_explanation"
            approach = "supportive"
        elif user_engagement < 0.4:
            strategy = "interactive_questioning"
            approach = "engaging"
        else:
            strategy = "practical_examples"
            approach = "balanced"
        
        return {
            "strategy": strategy,
            "approach": "optimal",
            "adaptation_level": min(0.9, user_level + 0.3),
            "interaction_style": approach
        }
    
    def _calculate_reward(self, response: Any, learning_context: Dict[str, Any], 
                        action: Dict[str, Any]) -> float:
        """Calculate reward for the current interaction"""
        if not self.reward_calculator:
            # Simple reward calculation if no reward calculator available
            return self._simple_reward_calculation(response, learning_context, action)
        
        # Extract metrics for reward calculation
        metrics = self._extract_reward_metrics(response, learning_context, action)
        
        # Calculate total reward
        total_reward = self.reward_calculator.calculate_total_reward(metrics)
        
        return total_reward
    
    def _simple_reward_calculation(self, response: Any, learning_context: Dict[str, Any], 
                                 action: Dict[str, Any]) -> float:
        """Simple reward calculation when no reward calculator is available"""
        base_reward = 0.5  # Neutral baseline
        
        # Reward based on response quality (proxy measures)
        if hasattr(response, 'content'):
            content_length = len(response.content)
            if 50 <= content_length <= 500:  # Appropriate length
                base_reward += 0.2
            elif content_length > 500:
                base_reward += 0.1  # Slightly less for very long responses
        
        # Reward based on user engagement (if available)
        user_profile = learning_context.get("user_profile", {})
        if user_profile.get("satisfaction_score", 0) > 0.6:
            base_reward += 0.3
        
        # Penalty for exploration in inappropriate contexts
        if action.get("action_type") == "exploration":
            user_level = learning_context.get("user_profile", {}).get("experience_level", "beginner")
            if user_level == "beginner":
                base_reward -= 0.1  # Small penalty for over-exploration with beginners
        
        return np.clip(base_reward, -1.0, 1.0)
    
    def _extract_reward_metrics(self, response: Any, learning_context: Dict[str, Any], 
                              action: Dict[str, Any]) -> Dict[str, Any]:
        """Extract metrics for detailed reward calculation"""
        user_profile = learning_context.get("user_profile", {})
        progress_metrics = learning_context.get("progress_metrics", {})
        conversation_analysis = learning_context.get("conversation_analysis", {})
        
        # Task completion metrics
        task_completion = {
            "success": True,  # Assume success for response generation
            "completion_time": time.time() - (self.episode_start_time or time.time()),
            "expected_time": 30.0,  # Expected response time in seconds
            "quality_score": self._estimate_response_quality(response, action)
        }
        
        # User satisfaction metrics
        user_satisfaction = {
            "satisfaction_score": user_profile.get("satisfaction_score", 0.5),
            "feedback_sentiment": conversation_analysis.get("sentiment_score", 0.0),
            "interaction_quality": conversation_analysis.get("interaction_quality", 0.5)
        }
        
        # Learning efficiency metrics
        learning_efficiency = {
            "knowledge_gained": progress_metrics.get("knowledge_gained", 0.1),
            "learning_time": self.step_count * 30,  # Approximate time per step
            "retention_rate": progress_metrics.get("knowledge_retention", 0.5)
        }
        
        # Collaboration metrics (for multi-agent scenarios)
        collaboration = {
            "team_performance": 0.7,  # Default value
            "communication_quality": 0.8,
            "coordination_score": 0.6
        }
        
        return {
            "task_completion": task_completion,
            "user_satisfaction": user_satisfaction,
            "learning_efficiency": learning_efficiency,
            "collaboration": collaboration
        }
    
    def _estimate_response_quality(self, response: Any, action: Dict[str, Any]) -> float:
        """Estimate response quality based on various factors"""
        quality_score = 0.5  # Base quality
        
        if hasattr(response, 'content'):
            content = response.content
            
            # Length appropriateness
            if 100 <= len(content) <= 800:
                quality_score += 0.2
            
            # Strategy alignment
            strategy = action.get("strategy", "")
            if strategy == "detailed_explanation" and len(content) > 200:
                quality_score += 0.1
            elif strategy == "practical_examples" and "example" in content.lower():
                quality_score += 0.1
            elif strategy == "interactive_questioning" and "?" in content:
                quality_score += 0.1
        
        return np.clip(quality_score, 0.0, 1.0)
    
    def _store_experience(self, state: Dict[str, Any], action: Dict[str, Any], 
                         reward: float, next_state: Dict[str, Any], done: bool) -> None:
        """Store experience in the experience buffer"""
        if not self.experience_buffer:
            return
        
        experience = Experience(
            agent_id=f"{self.agent_type}_{id(self)}",
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
            timestamp=time.time(),
            episode_id=self.current_episode_id or "unknown",
            step_id=self.step_count
        )
        
        self.experience_buffer.add_experience(experience)
        
        if self.is_logging:
            logger.debug(f"Stored experience: reward={reward:.3f}, done={done}")
    
    def _update_exploration_rate(self) -> None:
        """Update exploration rate with decay"""
        self.exploration_rate = max(
            self.min_exploration_rate,
            self.exploration_rate * self.exploration_decay
        )
    
    def _update_performance_metrics(self, reward: float, success: bool) -> None:
        """Update agent performance metrics"""
        self.performance_metrics['total_interactions'] += 1
        
        if success:
            self.performance_metrics['successful_interactions'] += 1
        
        # Update average reward (exponential moving average)
        alpha = 0.1
        self.performance_metrics['average_reward'] = (
            alpha * reward + (1 - alpha) * self.performance_metrics['average_reward']
        )
        
        # Update success rate
        success_rate = (
            self.performance_metrics['successful_interactions'] / 
            self.performance_metrics['total_interactions']
        )
        
        # Update learning efficiency (reward per interaction)
        self.performance_metrics['learning_efficiency'] = (
            self.performance_metrics['average_reward'] * success_rate
        )
    
    async def ainvoke(self, query: str, is_save_memory: bool = False, 
                     user_id: str = "unknown_user", learning_context: Dict = None, 
                     **kwargs) -> Any:
        """RL-enhanced asynchronous invoke with learning capabilities"""
        # Start episode if not already started
        if not self.current_episode_id:
            initial_state = self._extract_state_features(learning_context or {})
            self.start_episode(initial_state)
        
        # Extract current state
        current_state = self._extract_state_features(learning_context or {})
        self.current_state = current_state
        
        # Select action using RL policy
        action = self._select_action(current_state, query)
        self.last_action = action
        
        # Enhance learning context with RL action
        enhanced_context = (learning_context or {}).copy()
        enhanced_context['rl_action'] = action
        enhanced_context['rl_state'] = current_state
        
        # Call parent's ainvoke method
        response = await super().ainvoke(
            query, is_save_memory, user_id, enhanced_context, **kwargs
        )
        
        # Extract next state
        next_state = self._extract_state_features(enhanced_context)
        
        # Calculate reward
        reward = self._calculate_reward(response, enhanced_context, action)
        
        # Determine if episode is done (simple heuristic)
        done = self._is_episode_done(enhanced_context, response)
        
        # Store experience
        self._store_experience(current_state, action, reward, next_state, done)
        
        # Update metrics and exploration rate
        success = reward > 0
        self._update_performance_metrics(reward, success)
        self._update_exploration_rate()
        
        # Increment step count
        self.step_count += 1
        
        # End episode if done
        if done:
            self.current_episode_id = None
            if self.is_logging:
                logger.info(f"Episode completed for agent {self.agent_type}: "
                           f"steps={self.step_count}, final_reward={reward:.3f}")
        
        return response
    
    def _is_episode_done(self, learning_context: Dict[str, Any], response: Any) -> bool:
        """Determine if the current episode should end"""
        # Simple heuristics for episode termination
        
        # End after certain number of steps
        if self.step_count >= self.rl_config.get('max_episode_steps', 50):
            return True
        
        # End if user satisfaction is very high (successful completion)
        user_profile = learning_context.get("user_profile", {})
        if user_profile.get("satisfaction_score", 0) > 0.9:
            return True
        
        # End if learning objective is achieved
        progress_metrics = learning_context.get("progress_metrics", {})
        if progress_metrics.get("current_module_progress", 0) >= 1.0:
            return True
        
        return False
    
    def get_rl_capabilities(self) -> List[str]:
        """Get RL-specific capabilities"""
        return self.rl_capabilities
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        return self.performance_metrics.copy()
    
    def get_rl_state(self) -> Dict[str, Any]:
        """Get current RL state information"""
        return {
            "current_state": self.current_state,
            "last_action": self.last_action,
            "episode_id": self.current_episode_id,
            "step_count": self.step_count,
            "exploration_rate": self.exploration_rate,
            "performance_metrics": self.performance_metrics
        }
    
    def reset_rl_state(self) -> None:
        """Reset RL state for new learning session"""
        self.current_state = {}
        self.last_action = {}
        self.current_episode_id = None
        self.step_count = 0
        self.episode_start_time = None
        
        # Reset exploration rate
        self.exploration_rate = self.rl_config.get('exploration_rate', 0.1)
        
        if self.is_logging:
            logger.info(f"Reset RL state for agent {self.agent_type}")