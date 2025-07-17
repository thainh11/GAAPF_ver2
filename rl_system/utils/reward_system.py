"""Reward System for GAAPF RL Implementation

This module defines the reward structure for evaluating agent performance
in the dual-system RL approach.
"""

import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class RewardType(Enum):
    """Types of rewards in the system"""
    TASK_COMPLETION = "task_completion"
    USER_SATISFACTION = "user_satisfaction"
    LEARNING_EFFICIENCY = "learning_efficiency"
    COLLABORATION = "collaboration"
    KNOWLEDGE_TRANSFER = "knowledge_transfer"
    ADAPTATION = "adaptation"

@dataclass
class RewardEvent:
    """Represents a single reward event"""
    agent_id: str
    reward_type: RewardType
    value: float
    timestamp: float
    context: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None

class RewardCalculator:
    """Calculates rewards based on agent performance metrics"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.reward_weights = config.get('reward_weights', {
            RewardType.TASK_COMPLETION: 0.3,
            RewardType.USER_SATISFACTION: 0.25,
            RewardType.LEARNING_EFFICIENCY: 0.2,
            RewardType.COLLABORATION: 0.15,
            RewardType.KNOWLEDGE_TRANSFER: 0.05,
            RewardType.ADAPTATION: 0.05
        })
        self.reward_history: List[RewardEvent] = []
    
    def calculate_task_completion_reward(self, 
                                       task_success: bool,
                                       completion_time: float,
                                       expected_time: float,
                                       quality_score: float) -> float:
        """Calculate reward for task completion"""
        if not task_success:
            return -0.5  # Penalty for failure
        
        # Base reward for completion
        base_reward = 1.0
        
        # Time efficiency bonus/penalty
        time_ratio = completion_time / expected_time
        time_reward = max(0, 2.0 - time_ratio)  # Bonus for faster completion
        
        # Quality bonus
        quality_reward = quality_score * 0.5
        
        total_reward = base_reward + time_reward + quality_reward
        return np.clip(total_reward, 0, 3.0)
    
    def calculate_user_satisfaction_reward(self, 
                                         satisfaction_score: float,
                                         feedback_sentiment: float,
                                         interaction_quality: float) -> float:
        """Calculate reward based on user satisfaction"""
        # Normalize inputs to [0, 1]
        satisfaction_norm = np.clip(satisfaction_score, 0, 1)
        sentiment_norm = np.clip((feedback_sentiment + 1) / 2, 0, 1)  # Convert from [-1,1] to [0,1]
        interaction_norm = np.clip(interaction_quality, 0, 1)
        
        # Weighted combination
        reward = (satisfaction_norm * 0.5 + 
                 sentiment_norm * 0.3 + 
                 interaction_norm * 0.2)
        
        return reward * 2.0  # Scale to [0, 2]
    
    def calculate_learning_efficiency_reward(self,
                                           knowledge_gained: float,
                                           learning_time: float,
                                           retention_rate: float) -> float:
        """Calculate reward for learning efficiency"""
        if learning_time <= 0:
            return 0
        
        # Learning rate (knowledge per unit time)
        learning_rate = knowledge_gained / learning_time
        
        # Combine with retention
        efficiency = learning_rate * retention_rate
        
        return np.clip(efficiency * 2.0, 0, 2.0)
    
    def calculate_collaboration_reward(self,
                                     team_performance: float,
                                     communication_quality: float,
                                     coordination_score: float) -> float:
        """Calculate reward for collaboration effectiveness"""
        collaboration_score = (
            team_performance * 0.4 +
            communication_quality * 0.3 +
            coordination_score * 0.3
        )
        
        return np.clip(collaboration_score * 1.5, 0, 1.5)
    
    def calculate_knowledge_transfer_reward(self,
                                          knowledge_shared: float,
                                          transfer_success_rate: float) -> float:
        """Calculate reward for knowledge transfer"""
        transfer_reward = knowledge_shared * transfer_success_rate
        return np.clip(transfer_reward, 0, 1.0)
    
    def calculate_adaptation_reward(self,
                                  adaptation_speed: float,
                                  adaptation_accuracy: float) -> float:
        """Calculate reward for adaptation to new situations"""
        adaptation_score = (adaptation_speed * 0.6 + adaptation_accuracy * 0.4)
        return np.clip(adaptation_score, 0, 1.0)
    
    def calculate_total_reward(self, metrics: Dict[str, Any]) -> float:
        """Calculate total weighted reward from all components"""
        rewards = {}
        
        # Task completion reward
        if 'task_completion' in metrics:
            tc = metrics['task_completion']
            rewards[RewardType.TASK_COMPLETION] = self.calculate_task_completion_reward(
                tc.get('success', False),
                tc.get('completion_time', 0),
                tc.get('expected_time', 1),
                tc.get('quality_score', 0)
            )
        
        # User satisfaction reward
        if 'user_satisfaction' in metrics:
            us = metrics['user_satisfaction']
            rewards[RewardType.USER_SATISFACTION] = self.calculate_user_satisfaction_reward(
                us.get('satisfaction_score', 0),
                us.get('feedback_sentiment', 0),
                us.get('interaction_quality', 0)
            )
        
        # Learning efficiency reward
        if 'learning_efficiency' in metrics:
            le = metrics['learning_efficiency']
            rewards[RewardType.LEARNING_EFFICIENCY] = self.calculate_learning_efficiency_reward(
                le.get('knowledge_gained', 0),
                le.get('learning_time', 1),
                le.get('retention_rate', 0)
            )
        
        # Collaboration reward
        if 'collaboration' in metrics:
            col = metrics['collaboration']
            rewards[RewardType.COLLABORATION] = self.calculate_collaboration_reward(
                col.get('team_performance', 0),
                col.get('communication_quality', 0),
                col.get('coordination_score', 0)
            )
        
        # Knowledge transfer reward
        if 'knowledge_transfer' in metrics:
            kt = metrics['knowledge_transfer']
            rewards[RewardType.KNOWLEDGE_TRANSFER] = self.calculate_knowledge_transfer_reward(
                kt.get('knowledge_shared', 0),
                kt.get('transfer_success_rate', 0)
            )
        
        # Adaptation reward
        if 'adaptation' in metrics:
            adapt = metrics['adaptation']
            rewards[RewardType.ADAPTATION] = self.calculate_adaptation_reward(
                adapt.get('adaptation_speed', 0),
                adapt.get('adaptation_accuracy', 0)
            )
        
        # Calculate weighted total
        total_reward = sum(
            rewards.get(reward_type, 0) * weight
            for reward_type, weight in self.reward_weights.items()
        )
        
        return total_reward
    
    def record_reward(self, agent_id: str, reward_type: RewardType, 
                     value: float, context: Dict[str, Any],
                     timestamp: float) -> None:
        """Record a reward event"""
        event = RewardEvent(
            agent_id=agent_id,
            reward_type=reward_type,
            value=value,
            timestamp=timestamp,
            context=context
        )
        self.reward_history.append(event)
        
        logger.info(f"Recorded reward: {agent_id} - {reward_type.value}: {value}")
    
    def get_agent_reward_history(self, agent_id: str, 
                               reward_type: Optional[RewardType] = None) -> List[RewardEvent]:
        """Get reward history for a specific agent"""
        history = [event for event in self.reward_history if event.agent_id == agent_id]
        
        if reward_type:
            history = [event for event in history if event.reward_type == reward_type]
        
        return history
    
    def get_average_reward(self, agent_id: str, 
                          reward_type: Optional[RewardType] = None,
                          time_window: Optional[float] = None) -> float:
        """Calculate average reward for an agent"""
        history = self.get_agent_reward_history(agent_id, reward_type)
        
        if time_window:
            current_time = max(event.timestamp for event in self.reward_history) if self.reward_history else 0
            history = [event for event in history if current_time - event.timestamp <= time_window]
        
        if not history:
            return 0.0
        
        return sum(event.value for event in history) / len(history)
    
    def reset_history(self) -> None:
        """Reset reward history"""
        self.reward_history.clear()
        logger.info("Reward history reset")

class RewardShaper:
    """Shapes rewards to improve learning"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.shaping_enabled = config.get('reward_shaping', True)
        self.curiosity_weight = config.get('curiosity_weight', 0.1)
        self.exploration_bonus = config.get('exploration_bonus', 0.05)
    
    def shape_reward(self, base_reward: float, 
                    exploration_novelty: float,
                    curiosity_score: float) -> float:
        """Apply reward shaping techniques"""
        if not self.shaping_enabled:
            return base_reward
        
        shaped_reward = base_reward
        
        # Add exploration bonus
        exploration_bonus = exploration_novelty * self.exploration_bonus
        shaped_reward += exploration_bonus
        
        # Add curiosity-driven reward
        curiosity_bonus = curiosity_score * self.curiosity_weight
        shaped_reward += curiosity_bonus
        
        return shaped_reward