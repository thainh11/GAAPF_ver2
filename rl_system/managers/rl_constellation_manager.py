"""RL-Enhanced Constellation Manager for GAAPF Dual-System Implementation

This module provides an RL-enhanced version of the ConstellationManager that
optimizes agent team formation and coordination using reinforcement learning.
"""

import numpy as np
import time
import uuid
from typing import Dict, List, Any, Optional, Tuple, Set
from collections import defaultdict, deque
import logging
from dataclasses import dataclass
from enum import Enum

# Import base classes (with fallbacks for development)
try:
    from src.GAAPF.core.constellation.constellation_manager import ConstellationManager
    from src.GAAPF.core.agents import SpecializedAgent
except ImportError:
    # Fallback for development
    from abc import ABC, abstractmethod
    
    class ConstellationManager(ABC):
        def __init__(self, *args, **kwargs):
            pass
    
    class SpecializedAgent(ABC):
        pass

# Import RL components
from ..utils.reward_system import RewardCalculator, RewardType
from ..utils.experience_buffer import Experience, ExperienceBuffer
from ..agents.rl_specialized_agent import RLSpecializedAgent

logger = logging.getLogger(__name__)

class TeamFormationStrategy(Enum):
    """Strategies for team formation"""
    RANDOM = "random"
    GREEDY = "greedy"
    EPSILON_GREEDY = "epsilon_greedy"
    UCB = "upper_confidence_bound"
    THOMPSON_SAMPLING = "thompson_sampling"
    RL_OPTIMIZED = "rl_optimized"

@dataclass
class TeamPerformance:
    """Team performance metrics"""
    team_id: str
    agent_types: List[str]
    task_type: str
    success_rate: float
    average_reward: float
    collaboration_score: float
    efficiency_score: float
    user_satisfaction: float
    formation_count: int
    last_used: float

@dataclass
class TeamFormationAction:
    """Action for team formation"""
    selected_agents: List[str]
    formation_strategy: TeamFormationStrategy
    confidence_score: float
    expected_performance: float
    reasoning: str

@dataclass
class RLConstellationConfig:
    """Configuration for RL Constellation Manager"""
    exploration_rate: float = 0.2
    exploration_decay: float = 0.995
    min_exploration_rate: float = 0.05
    ucb_confidence: float = 2.0
    max_team_size: int = 5
    performance_history_size: int = 100
    reward_weights: Dict[str, float] = None
    enable_learning: bool = True
    enable_adaptation: bool = True
    formation_timeout: float = 30.0
    
    def __post_init__(self):
        if self.reward_weights is None:
            self.reward_weights = {
                'task_completion': 0.4,
                'collaboration': 0.3,
                'efficiency': 0.2,
                'user_satisfaction': 0.1
            }

class RLConstellationManager(ConstellationManager):
    """RL-enhanced constellation manager with optimized team formation"""
    
    def __init__(self,
                 agents: Dict[str, SpecializedAgent],
                 config: Dict[str, Any],
                 rl_config: Optional[Dict[str, Any]] = None,
                 reward_calculator: Optional[RewardCalculator] = None,
                 experience_buffer: Optional[ExperienceBuffer] = None,
                 is_logging: bool = False):
        """
        Initialize RL-enhanced constellation manager.
        
        Parameters:
        ----------
        agents : Dict[str, SpecializedAgent]
            Available agents
        config : Dict[str, Any]
            Manager configuration
        rl_config : Optional[Dict[str, Any]]
            RL-specific configuration
        reward_calculator : Optional[RewardCalculator]
            Reward calculation system
        experience_buffer : Optional[ExperienceBuffer]
            Experience storage system
        is_logging : bool
            Enable logging
        """
        # Initialize base manager
        self.agents = agents
        self.config = config
        self.is_logging = is_logging
        
        try:
            super().__init__(agents, config, is_logging)
        except Exception as e:
            # Fallback initialization already done above
            pass
        
        # RL-specific initialization
        self.rl_config = rl_config or {}
        self.reward_calculator = reward_calculator
        self.experience_buffer = experience_buffer
        
        # Team formation learning
        self.team_performance_history: Dict[str, TeamPerformance] = {}
        self.formation_strategies = list(TeamFormationStrategy)
        self.strategy_performance: Dict[TeamFormationStrategy, deque] = {
            strategy: deque(maxlen=100) for strategy in self.formation_strategies
        }
        
        # RL parameters
        self.exploration_rate = self.rl_config.get('exploration_rate', 0.2)
        self.exploration_decay = self.rl_config.get('exploration_decay', 0.995)
        self.min_exploration_rate = self.rl_config.get('min_exploration_rate', 0.05)
        self.ucb_confidence = self.rl_config.get('ucb_confidence', 2.0)
        
        # Team formation state
        self.current_formation_state = {}
        self.formation_episode_id = None
        self.formation_step_count = 0
        
        # Performance tracking
        self.formation_metrics = {
            'total_formations': 0,
            'successful_formations': 0,
            'average_team_performance': 0.0,
            'strategy_success_rates': {strategy: 0.0 for strategy in self.formation_strategies}
        }
        
        # Agent compatibility matrix (learned over time)
        self.agent_compatibility = defaultdict(lambda: defaultdict(float))
        self.agent_performance = defaultdict(lambda: {
            'individual_score': 0.5,
            'collaboration_score': 0.5,
            'task_success_rate': 0.5,
            'usage_count': 0
        })
        
        # Task-specific team preferences
        self.task_team_preferences = defaultdict(list)
        
        if self.is_logging:
            logger.info(f"Initialized RL-enhanced constellation manager with {len(agents)} agents")
    
    def form_constellation(self, 
                          task_description: str,
                          user_context: Dict[str, Any],
                          required_capabilities: Optional[List[str]] = None,
                          max_agents: int = 3) -> Tuple[List[str], Dict[str, Any]]:
        """Form optimal constellation using RL-enhanced selection"""
        
        # Start formation episode
        self.formation_episode_id = str(uuid.uuid4())
        self.formation_step_count = 0
        
        # Extract formation state
        formation_state = self._extract_formation_state(
            task_description, user_context, required_capabilities, max_agents
        )
        self.current_formation_state = formation_state
        
        # Select formation strategy
        strategy = self._select_formation_strategy(formation_state)
        
        # Form team using selected strategy
        selected_agents, formation_metadata = self._form_team_with_strategy(
            strategy, formation_state, task_description, user_context, 
            required_capabilities, max_agents
        )
        
        # Create formation action
        formation_action = TeamFormationAction(
            selected_agents=selected_agents,
            formation_strategy=strategy,
            confidence_score=formation_metadata.get('confidence', 0.5),
            expected_performance=formation_metadata.get('expected_performance', 0.5),
            reasoning=formation_metadata.get('reasoning', 'RL-optimized selection')
        )
        
        # Store formation for learning
        self._store_formation_decision(formation_state, formation_action)
        
        # Update metrics
        self.formation_metrics['total_formations'] += 1
        
        if self.is_logging:
            logger.info(f"Formed constellation using {strategy.value} strategy: {selected_agents}")
        
        return selected_agents, {
            'strategy': strategy.value,
            'confidence': formation_action.confidence_score,
            'expected_performance': formation_action.expected_performance,
            'reasoning': formation_action.reasoning,
            'formation_id': self.formation_episode_id
        }
    
    def _extract_formation_state(self, 
                               task_description: str,
                               user_context: Dict[str, Any],
                               required_capabilities: Optional[List[str]],
                               max_agents: int) -> Dict[str, Any]:
        """Extract state features for team formation"""
        user_profile = user_context.get('user_profile', {})
        
        # Task characteristics
        task_complexity = self._estimate_task_complexity(task_description)
        task_type = self._classify_task_type(task_description, required_capabilities)
        
        # User characteristics
        user_level = self._encode_user_level(user_profile.get('experience_level', 'beginner'))
        user_engagement = user_context.get('engagement_score', 0.5)
        
        # Context features
        interaction_count = user_context.get('interaction_count', 0)
        session_duration = user_context.get('session_duration', 0)
        
        # Available agents features
        available_agents = list(self.agents.keys())
        agent_capabilities = self._get_agent_capabilities_vector(available_agents)
        
        # Historical performance
        similar_task_performance = self._get_similar_task_performance(task_type)
        
        formation_state = {
            'task_complexity': task_complexity,
            'task_type': task_type,
            'user_level': user_level,
            'user_engagement': user_engagement,
            'interaction_count': interaction_count,
            'session_duration': session_duration,
            'max_agents': max_agents,
            'required_capabilities': required_capabilities or [],
            'available_agents': available_agents,
            'agent_capabilities': agent_capabilities,
            'similar_task_performance': similar_task_performance,
            'current_time': time.time()
        }
        
        return formation_state
    
    def _estimate_task_complexity(self, task_description: str) -> float:
        """Estimate task complexity from description"""
        # Simple heuristics for task complexity
        complexity_indicators = [
            'complex', 'advanced', 'detailed', 'comprehensive', 'in-depth',
            'multiple', 'various', 'different', 'compare', 'analyze'
        ]
        
        simplicity_indicators = [
            'simple', 'basic', 'quick', 'brief', 'overview', 'introduction'
        ]
        
        description_lower = task_description.lower()
        
        complexity_score = sum(1 for indicator in complexity_indicators 
                             if indicator in description_lower)
        simplicity_score = sum(1 for indicator in simplicity_indicators 
                             if indicator in description_lower)
        
        # Normalize based on description length
        length_factor = min(len(task_description) / 200, 1.0)
        
        complexity = (complexity_score - simplicity_score + length_factor) / 3
        return np.clip(complexity, 0.0, 1.0)
    
    def _classify_task_type(self, task_description: str, 
                          required_capabilities: Optional[List[str]]) -> str:
        """Classify task type for team formation"""
        description_lower = task_description.lower()
        capabilities = required_capabilities or []
        
        # Task type classification
        if any(word in description_lower for word in ['code', 'programming', 'implementation']):
            return 'coding'
        elif any(word in description_lower for word in ['explain', 'concept', 'theory']):
            return 'explanation'
        elif any(word in description_lower for word in ['practice', 'exercise', 'hands-on']):
            return 'practice'
        elif any(word in description_lower for word in ['debug', 'error', 'problem']):
            return 'troubleshooting'
        elif any(word in description_lower for word in ['project', 'build', 'create']):
            return 'project'
        elif any(cap in ['assessment', 'quiz'] for cap in capabilities):
            return 'assessment'
        else:
            return 'general'
    
    def _encode_user_level(self, level: str) -> float:
        """Encode user experience level"""
        level_mapping = {
            'beginner': 0.0,
            'intermediate': 0.5,
            'advanced': 1.0
        }
        return level_mapping.get(level.lower(), 0.0)
    
    def _get_agent_capabilities_vector(self, agent_names: List[str]) -> Dict[str, float]:
        """Get capabilities vector for available agents"""
        capabilities = {}
        
        for agent_name in agent_names:
            if agent_name in self.agents:
                agent = self.agents[agent_name]
                # Get agent capabilities (simplified)
                agent_caps = getattr(agent, 'CAPABILITIES', [])
                capabilities[agent_name] = len(agent_caps) / 10.0  # Normalize
            else:
                capabilities[agent_name] = 0.0
        
        return capabilities
    
    def _get_similar_task_performance(self, task_type: str) -> float:
        """Get performance for similar tasks"""
        similar_performances = [
            perf.average_reward for perf in self.team_performance_history.values()
            if perf.task_type == task_type
        ]
        
        if similar_performances:
            return np.mean(similar_performances)
        else:
            return 0.5  # Default performance
    
    def _select_formation_strategy(self, formation_state: Dict[str, Any]) -> TeamFormationStrategy:
        """Select team formation strategy using multi-armed bandit approach"""
        
        # Epsilon-greedy strategy selection
        if np.random.random() < self.exploration_rate:
            # Exploration: try different strategies
            strategy = np.random.choice(self.formation_strategies)
            if self.is_logging:
                logger.debug(f"Exploring with strategy: {strategy.value}")
        else:
            # Exploitation: use best performing strategy
            strategy = self._get_best_strategy(formation_state)
            if self.is_logging:
                logger.debug(f"Exploiting with strategy: {strategy.value}")
        
        return strategy
    
    def _get_best_strategy(self, formation_state: Dict[str, Any]) -> TeamFormationStrategy:
        """Get best performing strategy based on historical data"""
        strategy_scores = {}
        
        for strategy in self.formation_strategies:
            performances = self.strategy_performance[strategy]
            if performances:
                # Calculate UCB score
                mean_performance = np.mean(performances)
                n_trials = len(performances)
                total_trials = sum(len(perfs) for perfs in self.strategy_performance.values())
                
                if total_trials > 0 and n_trials > 0:
                    ucb_bonus = self.ucb_confidence * np.sqrt(np.log(total_trials) / n_trials)
                    strategy_scores[strategy] = mean_performance + ucb_bonus
                else:
                    strategy_scores[strategy] = 0.5  # Default score
            else:
                strategy_scores[strategy] = 1.0  # High score for untried strategies
        
        # Select strategy with highest UCB score
        best_strategy = max(strategy_scores.keys(), key=lambda s: strategy_scores[s])
        return best_strategy
    
    def _form_team_with_strategy(self, 
                               strategy: TeamFormationStrategy,
                               formation_state: Dict[str, Any],
                               task_description: str,
                               user_context: Dict[str, Any],
                               required_capabilities: Optional[List[str]],
                               max_agents: int) -> Tuple[List[str], Dict[str, Any]]:
        """Form team using specified strategy"""
        
        available_agents = formation_state['available_agents']
        task_type = formation_state['task_type']
        
        if strategy == TeamFormationStrategy.RANDOM:
            return self._random_team_formation(available_agents, max_agents)
        elif strategy == TeamFormationStrategy.GREEDY:
            return self._greedy_team_formation(formation_state, max_agents)
        elif strategy == TeamFormationStrategy.EPSILON_GREEDY:
            return self._epsilon_greedy_team_formation(formation_state, max_agents)
        elif strategy == TeamFormationStrategy.UCB:
            return self._ucb_team_formation(formation_state, max_agents)
        elif strategy == TeamFormationStrategy.THOMPSON_SAMPLING:
            return self._thompson_sampling_team_formation(formation_state, max_agents)
        elif strategy == TeamFormationStrategy.RL_OPTIMIZED:
            return self._rl_optimized_team_formation(formation_state, max_agents)
        else:
            # Fallback to greedy
            return self._greedy_team_formation(formation_state, max_agents)
    
    def _random_team_formation(self, available_agents: List[str], 
                             max_agents: int) -> Tuple[List[str], Dict[str, Any]]:
        """Random team formation"""
        selected_count = min(max_agents, len(available_agents))
        selected_agents = np.random.choice(available_agents, selected_count, replace=False).tolist()
        
        metadata = {
            'confidence': 0.3,
            'expected_performance': 0.4,
            'reasoning': 'Random selection for exploration'
        }
        
        return selected_agents, metadata
    
    def _greedy_team_formation(self, formation_state: Dict[str, Any], 
                             max_agents: int) -> Tuple[List[str], Dict[str, Any]]:
        """Greedy team formation based on individual agent performance"""
        available_agents = formation_state['available_agents']
        
        # Score agents based on individual performance
        agent_scores = {}
        for agent_name in available_agents:
            performance = self.agent_performance[agent_name]
            score = (
                performance['individual_score'] * 0.4 +
                performance['collaboration_score'] * 0.3 +
                performance['task_success_rate'] * 0.3
            )
            agent_scores[agent_name] = score
        
        # Select top agents
        sorted_agents = sorted(agent_scores.keys(), key=lambda a: agent_scores[a], reverse=True)
        selected_agents = sorted_agents[:max_agents]
        
        avg_score = np.mean([agent_scores[agent] for agent in selected_agents])
        
        metadata = {
            'confidence': min(avg_score + 0.2, 1.0),
            'expected_performance': avg_score,
            'reasoning': f'Selected top {len(selected_agents)} performing agents'
        }
        
        return selected_agents, metadata
    
    def _epsilon_greedy_team_formation(self, formation_state: Dict[str, Any], 
                                     max_agents: int) -> Tuple[List[str], Dict[str, Any]]:
        """Epsilon-greedy team formation"""
        if np.random.random() < 0.3:  # 30% exploration
            return self._random_team_formation(formation_state['available_agents'], max_agents)
        else:
            return self._greedy_team_formation(formation_state, max_agents)
    
    def _ucb_team_formation(self, formation_state: Dict[str, Any], 
                          max_agents: int) -> Tuple[List[str], Dict[str, Any]]:
        """Upper Confidence Bound team formation"""
        available_agents = formation_state['available_agents']
        total_usage = sum(self.agent_performance[agent]['usage_count'] 
                         for agent in available_agents)
        
        # Calculate UCB scores for agents
        ucb_scores = {}
        for agent_name in available_agents:
            performance = self.agent_performance[agent_name]
            mean_score = (
                performance['individual_score'] * 0.4 +
                performance['collaboration_score'] * 0.3 +
                performance['task_success_rate'] * 0.3
            )
            
            usage_count = max(performance['usage_count'], 1)
            if total_usage > 0:
                ucb_bonus = self.ucb_confidence * np.sqrt(np.log(total_usage) / usage_count)
                ucb_scores[agent_name] = mean_score + ucb_bonus
            else:
                ucb_scores[agent_name] = mean_score
        
        # Select agents with highest UCB scores
        sorted_agents = sorted(ucb_scores.keys(), key=lambda a: ucb_scores[a], reverse=True)
        selected_agents = sorted_agents[:max_agents]
        
        avg_ucb = np.mean([ucb_scores[agent] for agent in selected_agents])
        
        metadata = {
            'confidence': min(avg_ucb * 0.8, 1.0),
            'expected_performance': avg_ucb * 0.7,
            'reasoning': 'UCB-based selection balancing performance and exploration'
        }
        
        return selected_agents, metadata
    
    def _thompson_sampling_team_formation(self, formation_state: Dict[str, Any], 
                                        max_agents: int) -> Tuple[List[str], Dict[str, Any]]:
        """Thompson sampling team formation"""
        available_agents = formation_state['available_agents']
        
        # Sample from beta distributions for each agent
        sampled_scores = {}
        for agent_name in available_agents:
            performance = self.agent_performance[agent_name]
            usage_count = max(performance['usage_count'], 1)
            
            # Beta distribution parameters
            alpha = performance['task_success_rate'] * usage_count + 1
            beta = (1 - performance['task_success_rate']) * usage_count + 1
            
            sampled_scores[agent_name] = np.random.beta(alpha, beta)
        
        # Select agents with highest sampled scores
        sorted_agents = sorted(sampled_scores.keys(), 
                             key=lambda a: sampled_scores[a], reverse=True)
        selected_agents = sorted_agents[:max_agents]
        
        avg_sampled = np.mean([sampled_scores[agent] for agent in selected_agents])
        
        metadata = {
            'confidence': avg_sampled,
            'expected_performance': avg_sampled,
            'reasoning': 'Thompson sampling based on success rate distributions'
        }
        
        return selected_agents, metadata
    
    def _rl_optimized_team_formation(self, formation_state: Dict[str, Any], 
                                   max_agents: int) -> Tuple[List[str], Dict[str, Any]]:
        """RL-optimized team formation using learned policies"""
        # This would use a trained RL model for team formation
        # For now, use a sophisticated heuristic that combines multiple factors
        
        available_agents = formation_state['available_agents']
        task_type = formation_state['task_type']
        user_level = formation_state['user_level']
        task_complexity = formation_state['task_complexity']
        
        # Multi-factor scoring
        agent_scores = {}
        for agent_name in available_agents:
            performance = self.agent_performance[agent_name]
            
            # Base performance score
            base_score = (
                performance['individual_score'] * 0.3 +
                performance['collaboration_score'] * 0.3 +
                performance['task_success_rate'] * 0.4
            )
            
            # Task-specific adjustment
            task_bonus = self._get_task_specific_bonus(agent_name, task_type)
            
            # User level compatibility
            level_bonus = self._get_user_level_bonus(agent_name, user_level)
            
            # Complexity handling
            complexity_bonus = self._get_complexity_bonus(agent_name, task_complexity)
            
            # Diversity bonus (for team composition)
            diversity_bonus = self._get_diversity_bonus(agent_name, available_agents)
            
            total_score = (
                base_score * 0.5 +
                task_bonus * 0.2 +
                level_bonus * 0.15 +
                complexity_bonus * 0.1 +
                diversity_bonus * 0.05
            )
            
            agent_scores[agent_name] = total_score
        
        # Select optimal team composition
        selected_agents = self._select_optimal_team_composition(
            agent_scores, available_agents, max_agents
        )
        
        avg_score = np.mean([agent_scores[agent] for agent in selected_agents])
        
        metadata = {
            'confidence': min(avg_score + 0.1, 1.0),
            'expected_performance': avg_score,
            'reasoning': 'RL-optimized selection using multi-factor scoring'
        }
        
        return selected_agents, metadata
    
    def _get_task_specific_bonus(self, agent_name: str, task_type: str) -> float:
        """Get task-specific performance bonus for agent"""
        # Agent type to task type mapping
        task_agent_preferences = {
            'coding': ['code_assistant', 'troubleshooter'],
            'explanation': ['instructor', 'documentation_expert'],
            'practice': ['practice_facilitator', 'mentor'],
            'troubleshooting': ['troubleshooter', 'code_assistant'],
            'project': ['project_guide', 'mentor'],
            'assessment': ['assessment', 'progress_tracker'],
            'general': ['instructor', 'mentor']
        }
        
        preferred_agents = task_agent_preferences.get(task_type, [])
        if any(pref in agent_name for pref in preferred_agents):
            return 0.3
        else:
            return 0.0
    
    def _get_user_level_bonus(self, agent_name: str, user_level: float) -> float:
        """Get user level compatibility bonus"""
        # Agent preferences for different user levels
        if user_level < 0.3:  # Beginner
            beginner_agents = ['instructor', 'mentor', 'motivational_coach']
            if any(agent in agent_name for agent in beginner_agents):
                return 0.2
        elif user_level > 0.7:  # Advanced
            advanced_agents = ['code_assistant', 'research_assistant', 'project_guide']
            if any(agent in agent_name for agent in advanced_agents):
                return 0.2
        
        return 0.0
    
    def _get_complexity_bonus(self, agent_name: str, task_complexity: float) -> float:
        """Get complexity handling bonus"""
        if task_complexity > 0.7:  # High complexity
            complex_agents = ['knowledge_synthesizer', 'research_assistant']
            if any(agent in agent_name for agent in complex_agents):
                return 0.15
        
        return 0.0
    
    def _get_diversity_bonus(self, agent_name: str, available_agents: List[str]) -> float:
        """Get diversity bonus for team composition"""
        # Simple diversity measure based on agent type variety
        agent_types = set()
        for agent in available_agents:
            agent_type = agent.split('_')[0] if '_' in agent else agent
            agent_types.add(agent_type)
        
        diversity_score = len(agent_types) / len(available_agents)
        return diversity_score * 0.1
    
    def _select_optimal_team_composition(self, agent_scores: Dict[str, float], 
                                       available_agents: List[str], 
                                       max_agents: int) -> List[str]:
        """Select optimal team composition considering synergies"""
        if max_agents == 1:
            # Single agent selection
            best_agent = max(agent_scores.keys(), key=lambda a: agent_scores[a])
            return [best_agent]
        
        # Multi-agent selection with synergy consideration
        sorted_agents = sorted(agent_scores.keys(), key=lambda a: agent_scores[a], reverse=True)
        
        # Start with best agent
        selected_agents = [sorted_agents[0]]
        
        # Add agents that complement the team
        for agent in sorted_agents[1:]:
            if len(selected_agents) >= max_agents:
                break
            
            # Check compatibility with existing team
            compatibility_score = self._calculate_team_compatibility(
                selected_agents + [agent]
            )
            
            if compatibility_score > 0.6:  # Threshold for good compatibility
                selected_agents.append(agent)
        
        return selected_agents
    
    def _calculate_team_compatibility(self, team_agents: List[str]) -> float:
        """Calculate compatibility score for a team"""
        if len(team_agents) <= 1:
            return 1.0
        
        compatibility_scores = []
        for i, agent1 in enumerate(team_agents):
            for agent2 in team_agents[i+1:]:
                compatibility = self.agent_compatibility[agent1][agent2]
                if compatibility == 0:  # No data, assume neutral
                    compatibility = 0.5
                compatibility_scores.append(compatibility)
        
        return np.mean(compatibility_scores) if compatibility_scores else 0.5
    
    def _store_formation_decision(self, formation_state: Dict[str, Any], 
                                formation_action: TeamFormationAction) -> None:
        """Store formation decision for learning"""
        if not self.experience_buffer:
            return
        
        # Create experience for team formation
        experience = Experience(
            agent_id="constellation_manager",
            state=formation_state,
            action={
                'selected_agents': formation_action.selected_agents,
                'strategy': formation_action.formation_strategy.value,
                'confidence': formation_action.confidence_score
            },
            reward=0.0,  # Will be updated when performance is measured
            next_state={},  # Will be updated after task completion
            done=False,
            timestamp=time.time(),
            episode_id=self.formation_episode_id or "unknown",
            step_id=self.formation_step_count
        )
        
        self.experience_buffer.add_experience(experience)
        self.formation_step_count += 1
    
    def update_team_performance(self, team_agents: List[str], 
                              task_type: str,
                              performance_metrics: Dict[str, float]) -> None:
        """Update team performance based on task results"""
        
        # Create team ID
        team_id = "_".join(sorted(team_agents))
        
        # Extract performance metrics
        success_rate = performance_metrics.get('success_rate', 0.5)
        average_reward = performance_metrics.get('average_reward', 0.0)
        collaboration_score = performance_metrics.get('collaboration_score', 0.5)
        efficiency_score = performance_metrics.get('efficiency_score', 0.5)
        user_satisfaction = performance_metrics.get('user_satisfaction', 0.5)
        
        # Update team performance history
        if team_id in self.team_performance_history:
            team_perf = self.team_performance_history[team_id]
            # Update with exponential moving average
            alpha = 0.3
            team_perf.success_rate = alpha * success_rate + (1 - alpha) * team_perf.success_rate
            team_perf.average_reward = alpha * average_reward + (1 - alpha) * team_perf.average_reward
            team_perf.collaboration_score = alpha * collaboration_score + (1 - alpha) * team_perf.collaboration_score
            team_perf.efficiency_score = alpha * efficiency_score + (1 - alpha) * team_perf.efficiency_score
            team_perf.user_satisfaction = alpha * user_satisfaction + (1 - alpha) * team_perf.user_satisfaction
            team_perf.formation_count += 1
            team_perf.last_used = time.time()
        else:
            # Create new team performance record
            self.team_performance_history[team_id] = TeamPerformance(
                team_id=team_id,
                agent_types=team_agents,
                task_type=task_type,
                success_rate=success_rate,
                average_reward=average_reward,
                collaboration_score=collaboration_score,
                efficiency_score=efficiency_score,
                user_satisfaction=user_satisfaction,
                formation_count=1,
                last_used=time.time()
            )
        
        # Update individual agent performance
        for agent_name in team_agents:
            agent_perf = self.agent_performance[agent_name]
            agent_perf['usage_count'] += 1
            
            # Update individual scores
            alpha = 0.2
            agent_perf['individual_score'] = (
                alpha * (success_rate + efficiency_score) / 2 + 
                (1 - alpha) * agent_perf['individual_score']
            )
            agent_perf['collaboration_score'] = (
                alpha * collaboration_score + 
                (1 - alpha) * agent_perf['collaboration_score']
            )
            agent_perf['task_success_rate'] = (
                alpha * success_rate + 
                (1 - alpha) * agent_perf['task_success_rate']
            )
        
        # Update agent compatibility matrix
        for i, agent1 in enumerate(team_agents):
            for agent2 in team_agents[i+1:]:
                current_compatibility = self.agent_compatibility[agent1][agent2]
                new_compatibility = (collaboration_score + success_rate) / 2
                
                # Update with exponential moving average
                alpha = 0.3
                self.agent_compatibility[agent1][agent2] = (
                    alpha * new_compatibility + (1 - alpha) * current_compatibility
                )
                self.agent_compatibility[agent2][agent1] = self.agent_compatibility[agent1][agent2]
        
        # Update formation metrics
        if success_rate > 0.6:  # Consider successful if > 60%
            self.formation_metrics['successful_formations'] += 1
        
        # Update strategy performance if we know which strategy was used
        # This would require tracking the strategy used for this team
        
        # Update exploration rate
        self._update_exploration_rate()
        
        if self.is_logging:
            logger.info(f"Updated performance for team {team_id}: "
                       f"success={success_rate:.3f}, reward={average_reward:.3f}")
    
    def _update_exploration_rate(self) -> None:
        """Update exploration rate with decay"""
        self.exploration_rate = max(
            self.min_exploration_rate,
            self.exploration_rate * self.exploration_decay
        )
    
    def get_formation_metrics(self) -> Dict[str, Any]:
        """Get constellation formation metrics"""
        total_formations = self.formation_metrics['total_formations']
        successful_formations = self.formation_metrics['successful_formations']
        
        metrics = self.formation_metrics.copy()
        
        if total_formations > 0:
            metrics['success_rate'] = successful_formations / total_formations
        else:
            metrics['success_rate'] = 0.0
        
        # Add strategy performance
        for strategy in self.formation_strategies:
            performances = self.strategy_performance[strategy]
            if performances:
                metrics['strategy_success_rates'][strategy] = np.mean(performances)
        
        return metrics
    
    def get_agent_performance_summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary of individual agent performance"""
        return dict(self.agent_performance)
    
    def get_team_performance_summary(self) -> Dict[str, TeamPerformance]:
        """Get summary of team performance"""
        return self.team_performance_history.copy()
    
    def reset_learning_state(self) -> None:
        """Reset learning state for new session"""
        self.formation_episode_id = None
        self.formation_step_count = 0
        self.current_formation_state = {}
        
        # Reset exploration rate
        self.exploration_rate = self.rl_config.get('exploration_rate', 0.2)
        
        if self.is_logging:
            logger.info("Reset RL constellation manager learning state")