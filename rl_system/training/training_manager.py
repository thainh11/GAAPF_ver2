"""Training Manager for GAAPF RL System

This module manages the training process for the reinforcement learning
components in the GAAPF dual-system approach.
"""

import numpy as np
import torch
import time
import json
import os
from typing import Dict, List, Any, Optional, Tuple, Union
from collections import defaultdict, deque
import logging
from dataclasses import dataclass, asdict
from enum import Enum
import threading
from concurrent.futures import ThreadPoolExecutor

# Import RL components
from ..utils.reward_system import RewardCalculator, RewardType
from ..utils.experience_buffer import ExperienceBuffer, Experience, Episode
from ..algorithms.maddpg import MADDPG, MADDPGConfig
from ..algorithms.dqn import DQNAgent, DQNConfig
from ..algorithms.policy_gradient import PolicyGradientAgent, PolicyGradientConfig
from ..agents.rl_specialized_agent import RLSpecializedAgent
from ..managers.rl_constellation_manager import RLConstellationManager

logger = logging.getLogger(__name__)

class TrainingPhase(Enum):
    """Training phases"""
    INITIALIZATION = "initialization"
    WARMUP = "warmup"
    ACTIVE_LEARNING = "active_learning"
    FINE_TUNING = "fine_tuning"
    EVALUATION = "evaluation"
    COMPLETED = "completed"

@dataclass
class TrainingConfig:
    """Configuration for training manager"""
    # General training settings
    max_episodes: int = 10000
    max_steps_per_episode: int = 1000
    evaluation_frequency: int = 100
    save_frequency: int = 500
    log_frequency: int = 10
    
    # Training phases
    warmup_episodes: int = 100
    active_learning_episodes: int = 8000
    fine_tuning_episodes: int = 1000
    
    # Multi-agent settings
    use_maddpg: bool = True
    use_individual_agents: bool = True
    
    # Curriculum learning
    use_curriculum: bool = True
    curriculum_stages: int = 5
    
    # Experience replay
    shared_experience: bool = True
    experience_sharing_rate: float = 0.1
    
    # Performance thresholds
    success_threshold: float = 0.8
    convergence_threshold: float = 0.01
    convergence_window: int = 100
    
    # Resource management
    max_parallel_agents: int = 4
    memory_limit_gb: float = 8.0
    
    # Paths
    model_save_dir: str = "models"
    log_save_dir: str = "logs"
    checkpoint_dir: str = "checkpoints"
    
    # Device
    device: str = "cpu"

@dataclass
class TrainingMetrics:
    """Training metrics tracking"""
    episode: int = 0
    total_steps: int = 0
    phase: TrainingPhase = TrainingPhase.INITIALIZATION
    
    # Performance metrics
    episode_rewards: List[float] = None
    episode_lengths: List[int] = None
    success_rates: List[float] = None
    
    # Learning metrics
    policy_losses: Dict[str, List[float]] = None
    value_losses: Dict[str, List[float]] = None
    q_values: Dict[str, List[float]] = None
    
    # Multi-agent metrics
    collaboration_scores: List[float] = None
    team_performance: Dict[str, float] = None
    
    # System metrics
    training_time: float = 0.0
    memory_usage: float = 0.0
    convergence_score: float = 0.0
    
    def __post_init__(self):
        if self.episode_rewards is None:
            self.episode_rewards = []
        if self.episode_lengths is None:
            self.episode_lengths = []
        if self.success_rates is None:
            self.success_rates = []
        if self.policy_losses is None:
            self.policy_losses = defaultdict(list)
        if self.value_losses is None:
            self.value_losses = defaultdict(list)
        if self.q_values is None:
            self.q_values = defaultdict(list)
        if self.collaboration_scores is None:
            self.collaboration_scores = []
        if self.team_performance is None:
            self.team_performance = {}

class TrainingManager:
    """Main training manager for GAAPF RL system"""
    
    def __init__(self,
                 config: TrainingConfig,
                 reward_calculator: RewardCalculator,
                 experience_buffer: ExperienceBuffer,
                 constellation_manager: RLConstellationManager,
                 agents: Dict[str, RLSpecializedAgent],
                 maddpg_config: Optional[MADDPGConfig] = None,
                 dqn_config: Optional[DQNConfig] = None,
                 pg_config: Optional[PolicyGradientConfig] = None):
        """
        Initialize training manager.
        
        Parameters:
        ----------
        config : TrainingConfig
            Training configuration
        reward_calculator : RewardCalculator
            Reward calculation system
        experience_buffer : ExperienceBuffer
            Experience storage system
        constellation_manager : RLConstellationManager
            RL-enhanced constellation manager
        agents : Dict[str, RLSpecializedAgent]
            RL-enhanced specialized agents
        maddpg_config : Optional[MADDPGConfig]
            MADDPG configuration
        dqn_config : Optional[DQNConfig]
            DQN configuration
        pg_config : Optional[PolicyGradientConfig]
            Policy gradient configuration
        """
        self.config = config
        self.reward_calculator = reward_calculator
        self.experience_buffer = experience_buffer
        self.constellation_manager = constellation_manager
        self.agents = agents
        
        # Algorithm configurations
        self.maddpg_config = maddpg_config or MADDPGConfig()
        self.dqn_config = dqn_config or DQNConfig()
        self.pg_config = pg_config or PolicyGradientConfig()
        
        # Initialize RL algorithms
        self._initialize_algorithms()
        
        # Training state
        self.metrics = TrainingMetrics()
        self.current_phase = TrainingPhase.INITIALIZATION
        self.training_start_time = None
        self.is_training = False
        self.should_stop = False
        
        # Episode tracking
        self.current_episode = 0
        self.current_step = 0
        self.episode_rewards = deque(maxlen=1000)
        self.episode_lengths = deque(maxlen=1000)
        
        # Performance tracking
        self.best_performance = 0.0
        self.convergence_tracker = deque(maxlen=config.convergence_window)
        
        # Threading for parallel training
        self.executor = ThreadPoolExecutor(max_workers=config.max_parallel_agents)
        self.training_lock = threading.Lock()
        
        # Create directories
        self._create_directories()
        
        logger.info(f"Initialized training manager with {len(agents)} agents")
    
    def _initialize_algorithms(self):
        """Initialize RL algorithms"""
        # Initialize MADDPG if enabled
        if self.config.use_maddpg:
            agent_configs = {}
            for agent_id, agent in self.agents.items():
                agent_configs[agent_id] = {
                    'state_dim': agent.state_dim,
                    'action_dim': agent.action_dim
                }
            
            self.maddpg = MADDPG(agent_configs, self.maddpg_config)
            logger.info("Initialized MADDPG algorithm")
        else:
            self.maddpg = None
        
        # Initialize individual agent algorithms
        if self.config.use_individual_agents:
            self.individual_algorithms = {}
            
            for agent_id, agent in self.agents.items():
                # Choose algorithm based on agent type or configuration
                if hasattr(agent, 'preferred_algorithm'):
                    algorithm = agent.preferred_algorithm
                else:
                    algorithm = 'dqn'  # Default
                
                if algorithm == 'dqn':
                    self.individual_algorithms[agent_id] = DQNAgent(
                        agent_id, agent.state_dim, agent.action_dim, self.dqn_config
                    )
                elif algorithm in ['ppo', 'actor_critic', 'reinforce']:
                    self.individual_algorithms[agent_id] = PolicyGradientAgent(
                        agent_id, agent.state_dim, agent.action_dim, 
                        self.pg_config, algorithm
                    )
                
                logger.info(f"Initialized {algorithm.upper()} for agent {agent_id}")
        else:
            self.individual_algorithms = {}
    
    def _create_directories(self):
        """Create necessary directories"""
        directories = [
            self.config.model_save_dir,
            self.config.log_save_dir,
            self.config.checkpoint_dir
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def start_training(self) -> None:
        """Start the training process"""
        if self.is_training:
            logger.warning("Training is already in progress")
            return
        
        self.is_training = True
        self.should_stop = False
        self.training_start_time = time.time()
        
        logger.info("Starting RL training process")
        
        try:
            self._training_loop()
        except Exception as e:
            logger.error(f"Training failed with error: {e}")
            raise
        finally:
            self.is_training = False
            logger.info("Training process completed")
    
    def stop_training(self) -> None:
        """Stop the training process"""
        self.should_stop = True
        logger.info("Training stop requested")
    
    def _training_loop(self) -> None:
        """Main training loop"""
        for episode in range(self.config.max_episodes):
            if self.should_stop:
                break
            
            self.current_episode = episode
            
            # Update training phase
            self._update_training_phase()
            
            # Run episode
            episode_metrics = self._run_episode()
            
            # Update metrics
            self._update_metrics(episode_metrics)
            
            # Evaluation
            if episode % self.config.evaluation_frequency == 0:
                self._evaluate_performance()
            
            # Save models
            if episode % self.config.save_frequency == 0:
                self._save_models()
            
            # Logging
            if episode % self.config.log_frequency == 0:
                self._log_progress()
            
            # Check convergence
            if self._check_convergence():
                logger.info(f"Training converged at episode {episode}")
                break
        
        # Final evaluation and save
        self._evaluate_performance()
        self._save_models()
        self.current_phase = TrainingPhase.COMPLETED
    
    def _update_training_phase(self) -> None:
        """Update current training phase"""
        episode = self.current_episode
        
        if episode < self.config.warmup_episodes:
            self.current_phase = TrainingPhase.WARMUP
        elif episode < self.config.warmup_episodes + self.config.active_learning_episodes:
            self.current_phase = TrainingPhase.ACTIVE_LEARNING
        elif episode < (self.config.warmup_episodes + 
                       self.config.active_learning_episodes + 
                       self.config.fine_tuning_episodes):
            self.current_phase = TrainingPhase.FINE_TUNING
        else:
            self.current_phase = TrainingPhase.EVALUATION
        
        self.metrics.phase = self.current_phase
    
    def _run_episode(self) -> Dict[str, Any]:
        """Run a single training episode"""
        episode_start_time = time.time()
        episode_reward = 0.0
        episode_length = 0
        episode_experiences = []
        
        # Initialize episode
        self._reset_episode()
        
        # Simulate learning interaction
        for step in range(self.config.max_steps_per_episode):
            self.current_step = step
            
            # Generate synthetic learning scenario
            scenario = self._generate_learning_scenario()
            
            # Form constellation
            selected_agents, formation_metadata = self.constellation_manager.form_constellation(
                task_description=scenario['task_description'],
                user_context=scenario['user_context'],
                required_capabilities=scenario.get('required_capabilities'),
                max_agents=scenario.get('max_agents', 3)
            )
            
            # Execute learning interaction
            interaction_result = self._execute_learning_interaction(
                selected_agents, scenario
            )
            
            # Calculate rewards
            rewards = self._calculate_step_rewards(interaction_result, scenario)
            
            # Store experiences
            self._store_step_experiences(
                selected_agents, scenario, interaction_result, rewards
            )
            
            # Update agents
            if self.current_phase != TrainingPhase.WARMUP:
                self._update_agents()
            
            # Update episode metrics
            episode_reward += sum(rewards.values())
            episode_length += 1
            
            # Check episode termination
            if interaction_result.get('done', False):
                break
        
        # Episode completion
        episode_time = time.time() - episode_start_time
        
        # Update constellation manager performance
        self._update_constellation_performance(selected_agents, episode_reward)
        
        return {
            'episode_reward': episode_reward,
            'episode_length': episode_length,
            'episode_time': episode_time,
            'selected_agents': selected_agents,
            'formation_metadata': formation_metadata
        }
    
    def _reset_episode(self) -> None:
        """Reset for new episode"""
        # Reset agents
        for agent in self.agents.values():
            agent.reset_episode()
        
        # Reset algorithms
        if self.maddpg:
            self.maddpg.reset_episode()
        
        for algorithm in self.individual_algorithms.values():
            algorithm.reset_episode()
        
        # Reset constellation manager
        self.constellation_manager.reset_learning_state()
    
    def _generate_learning_scenario(self) -> Dict[str, Any]:
        """Generate synthetic learning scenario"""
        # Define scenario types based on training phase
        if self.current_phase == TrainingPhase.WARMUP:
            complexity = np.random.uniform(0.1, 0.4)
            scenario_types = ['basic_concept', 'simple_practice']
        elif self.current_phase == TrainingPhase.ACTIVE_LEARNING:
            complexity = np.random.uniform(0.3, 0.8)
            scenario_types = ['coding_task', 'problem_solving', 'explanation', 'project']
        elif self.current_phase == TrainingPhase.FINE_TUNING:
            complexity = np.random.uniform(0.6, 1.0)
            scenario_types = ['advanced_project', 'complex_debugging', 'research_task']
        else:
            complexity = np.random.uniform(0.4, 0.9)
            scenario_types = ['mixed_task', 'comprehensive_project']
        
        scenario_type = np.random.choice(scenario_types)
        
        # Generate user profile
        user_levels = ['beginner', 'intermediate', 'advanced']
        user_level = np.random.choice(user_levels)
        
        # Create scenario
        scenario = {
            'task_description': self._generate_task_description(scenario_type, complexity),
            'scenario_type': scenario_type,
            'complexity': complexity,
            'user_context': {
                'user_profile': {
                    'experience_level': user_level,
                    'learning_style': np.random.choice(['visual', 'auditory', 'kinesthetic']),
                    'pace_preference': np.random.choice(['slow', 'medium', 'fast'])
                },
                'engagement_score': np.random.uniform(0.3, 1.0),
                'interaction_count': np.random.randint(1, 20),
                'session_duration': np.random.uniform(300, 3600)  # 5 min to 1 hour
            },
            'required_capabilities': self._get_required_capabilities(scenario_type),
            'max_agents': np.random.randint(2, 4),
            'expected_duration': np.random.uniform(60, 600)  # 1-10 minutes
        }
        
        return scenario
    
    def _generate_task_description(self, scenario_type: str, complexity: float) -> str:
        """Generate task description based on scenario type and complexity"""
        task_templates = {
            'basic_concept': [
                "Explain the concept of {concept}",
                "What is {concept} and how does it work?",
                "Provide an introduction to {concept}"
            ],
            'simple_practice': [
                "Create a simple {language} program that {task}",
                "Write code to {task} using {language}",
                "Practice {concept} with a basic example"
            ],
            'coding_task': [
                "Implement {algorithm} in {language}",
                "Create a {project_type} that {functionality}",
                "Debug this {language} code that {problem}"
            ],
            'problem_solving': [
                "Solve this {domain} problem: {problem}",
                "How would you approach {challenge}?",
                "Design a solution for {scenario}"
            ],
            'explanation': [
                "Explain how {concept} works in detail",
                "Compare {concept1} and {concept2}",
                "Describe the advantages and disadvantages of {approach}"
            ],
            'project': [
                "Build a {project_type} application that {features}",
                "Create a comprehensive {domain} project",
                "Develop a {complexity_level} {project_type} with {requirements}"
            ]
        }
        
        # Select template and fill placeholders
        templates = task_templates.get(scenario_type, task_templates['basic_concept'])
        template = np.random.choice(templates)
        
        # Fill placeholders based on complexity
        placeholders = self._get_task_placeholders(complexity)
        
        try:
            description = template.format(**placeholders)
        except KeyError:
            # Fallback if placeholders don't match
            description = f"Complete a {scenario_type} task with complexity {complexity:.2f}"
        
        return description
    
    def _get_task_placeholders(self, complexity: float) -> Dict[str, str]:
        """Get task placeholders based on complexity"""
        if complexity < 0.3:
            concepts = ['variables', 'loops', 'functions', 'conditionals']
            languages = ['Python', 'JavaScript']
            tasks = ['print hello world', 'calculate sum', 'find maximum']
        elif complexity < 0.7:
            concepts = ['classes', 'inheritance', 'algorithms', 'data structures']
            languages = ['Python', 'JavaScript', 'Java', 'C++']
            tasks = ['sort array', 'implement search', 'create API']
        else:
            concepts = ['design patterns', 'optimization', 'machine learning', 'distributed systems']
            languages = ['Python', 'JavaScript', 'Java', 'C++', 'Go', 'Rust']
            tasks = ['optimize performance', 'implement ML model', 'design architecture']
        
        return {
            'concept': np.random.choice(concepts),
            'concept1': np.random.choice(concepts),
            'concept2': np.random.choice(concepts),
            'language': np.random.choice(languages),
            'task': np.random.choice(tasks),
            'algorithm': np.random.choice(['sorting', 'searching', 'graph traversal']),
            'project_type': np.random.choice(['web app', 'mobile app', 'CLI tool', 'library']),
            'functionality': np.random.choice(['manages data', 'processes files', 'serves content']),
            'problem': np.random.choice(['performance issue', 'logic error', 'syntax error']),
            'domain': np.random.choice(['programming', 'algorithms', 'software design']),
            'challenge': np.random.choice(['optimization', 'scalability', 'maintainability']),
            'scenario': np.random.choice(['large dataset', 'real-time system', 'distributed application']),
            'approach': np.random.choice(['object-oriented', 'functional', 'procedural']),
            'features': np.random.choice(['user authentication', 'data visualization', 'real-time updates']),
            'complexity_level': 'simple' if complexity < 0.5 else 'complex',
            'requirements': np.random.choice(['testing', 'documentation', 'deployment'])
        }
    
    def _get_required_capabilities(self, scenario_type: str) -> List[str]:
        """Get required capabilities for scenario type"""
        capability_mapping = {
            'basic_concept': ['explanation', 'teaching'],
            'simple_practice': ['coding', 'guidance'],
            'coding_task': ['coding', 'debugging', 'review'],
            'problem_solving': ['analysis', 'problem_solving', 'guidance'],
            'explanation': ['explanation', 'teaching', 'examples'],
            'project': ['project_management', 'coding', 'architecture'],
            'advanced_project': ['architecture', 'optimization', 'best_practices'],
            'complex_debugging': ['debugging', 'analysis', 'troubleshooting'],
            'research_task': ['research', 'analysis', 'documentation']
        }
        
        return capability_mapping.get(scenario_type, ['general'])
    
    def _execute_learning_interaction(self, selected_agents: List[str], 
                                    scenario: Dict[str, Any]) -> Dict[str, Any]:
        """Execute learning interaction with selected agents"""
        # Simulate agent collaboration
        interaction_quality = np.random.uniform(0.4, 1.0)
        user_satisfaction = np.random.uniform(0.3, 1.0)
        task_completion = np.random.uniform(0.5, 1.0)
        
        # Adjust based on agent capabilities and scenario complexity
        complexity_factor = 1.0 - scenario['complexity'] * 0.3
        agent_factor = len(selected_agents) / 4.0  # Normalize by max agents
        
        interaction_quality *= (complexity_factor + agent_factor) / 2
        user_satisfaction *= (complexity_factor + agent_factor) / 2
        task_completion *= (complexity_factor + agent_factor) / 2
        
        # Clip values
        interaction_quality = np.clip(interaction_quality, 0.0, 1.0)
        user_satisfaction = np.clip(user_satisfaction, 0.0, 1.0)
        task_completion = np.clip(task_completion, 0.0, 1.0)
        
        return {
            'interaction_quality': interaction_quality,
            'user_satisfaction': user_satisfaction,
            'task_completion': task_completion,
            'collaboration_score': np.mean([interaction_quality, task_completion]),
            'efficiency_score': task_completion / max(len(selected_agents), 1),
            'done': task_completion > 0.8 and user_satisfaction > 0.7
        }
    
    def _calculate_step_rewards(self, interaction_result: Dict[str, Any], 
                              scenario: Dict[str, Any]) -> Dict[str, float]:
        """Calculate rewards for the step"""
        base_reward = interaction_result['task_completion'] * 10.0
        
        # Individual agent rewards
        agent_rewards = {}
        for agent_id in self.agents.keys():
            reward = base_reward
            
            # Add bonuses
            reward += interaction_result['user_satisfaction'] * 5.0
            reward += interaction_result['collaboration_score'] * 3.0
            
            # Complexity bonus
            reward += scenario['complexity'] * 2.0
            
            # Small random variation
            reward += np.random.uniform(-1.0, 1.0)
            
            agent_rewards[agent_id] = reward
        
        return agent_rewards
    
    def _store_step_experiences(self, selected_agents: List[str], 
                              scenario: Dict[str, Any],
                              interaction_result: Dict[str, Any],
                              rewards: Dict[str, float]) -> None:
        """Store experiences from the step"""
        # Create experiences for each agent
        for agent_id in selected_agents:
            if agent_id in self.agents:
                agent = self.agents[agent_id]
                
                # Get agent's state representation
                state = agent.extract_state_features(
                    scenario['task_description'],
                    scenario['user_context'],
                    scenario.get('required_capabilities', [])
                )
                
                # Get agent's action (simplified)
                action = agent.select_action(state, explore=True)
                
                # Create experience
                experience = Experience(
                    agent_id=agent_id,
                    state=state,
                    action=action,
                    reward=rewards.get(agent_id, 0.0),
                    next_state=state,  # Simplified for now
                    done=interaction_result.get('done', False),
                    timestamp=time.time(),
                    episode_id=f"episode_{self.current_episode}",
                    step_id=self.current_step
                )
                
                # Store in experience buffer
                self.experience_buffer.add_experience(experience)
    
    def _update_agents(self) -> None:
        """Update RL agents"""
        update_results = {}
        
        # Update MADDPG if enabled
        if self.maddpg and len(self.experience_buffer) > self.maddpg_config.batch_size:
            # Convert experiences to MADDPG format and update
            maddpg_results = self._update_maddpg()
            update_results.update(maddpg_results)
        
        # Update individual agents
        for agent_id, algorithm in self.individual_algorithms.items():
            if hasattr(algorithm, 'update'):
                agent_results = algorithm.update()
                if agent_results:
                    for key, value in agent_results.items():
                        update_results[f"{agent_id}_{key}"] = value
        
        # Update metrics
        self._update_algorithm_metrics(update_results)
    
    def _update_maddpg(self) -> Dict[str, float]:
        """Update MADDPG algorithm"""
        # This would require converting experiences to MADDPG format
        # For now, return empty results
        return {}
    
    def _update_algorithm_metrics(self, update_results: Dict[str, float]) -> None:
        """Update algorithm-specific metrics"""
        for key, value in update_results.items():
            if 'loss' in key:
                agent_id = key.split('_')[0]
                if 'policy' in key:
                    self.metrics.policy_losses[agent_id].append(value)
                elif 'value' in key or 'critic' in key:
                    self.metrics.value_losses[agent_id].append(value)
            elif 'q_value' in key:
                agent_id = key.split('_')[0]
                self.metrics.q_values[agent_id].append(value)
    
    def _update_constellation_performance(self, selected_agents: List[str], 
                                        episode_reward: float) -> None:
        """Update constellation manager performance"""
        # Calculate performance metrics
        performance_metrics = {
            'success_rate': 1.0 if episode_reward > 50.0 else 0.0,
            'average_reward': episode_reward,
            'collaboration_score': np.random.uniform(0.5, 1.0),  # Simplified
            'efficiency_score': np.random.uniform(0.5, 1.0),     # Simplified
            'user_satisfaction': np.random.uniform(0.5, 1.0)     # Simplified
        }
        
        # Update constellation manager
        self.constellation_manager.update_team_performance(
            selected_agents, 'general', performance_metrics
        )
    
    def _update_metrics(self, episode_metrics: Dict[str, Any]) -> None:
        """Update training metrics"""
        self.metrics.episode += 1
        self.metrics.total_steps += episode_metrics['episode_length']
        
        # Episode metrics
        episode_reward = episode_metrics['episode_reward']
        episode_length = episode_metrics['episode_length']
        
        self.metrics.episode_rewards.append(episode_reward)
        self.metrics.episode_lengths.append(episode_length)
        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(episode_length)
        
        # Success rate
        success = 1.0 if episode_reward > 50.0 else 0.0
        self.metrics.success_rates.append(success)
        
        # Collaboration score (simplified)
        collaboration_score = np.random.uniform(0.5, 1.0)
        self.metrics.collaboration_scores.append(collaboration_score)
        
        # Update convergence tracker
        if len(self.episode_rewards) >= 10:
            recent_performance = np.mean(list(self.episode_rewards)[-10:])
            self.convergence_tracker.append(recent_performance)
        
        # Update training time
        if self.training_start_time:
            self.metrics.training_time = time.time() - self.training_start_time
    
    def _evaluate_performance(self) -> None:
        """Evaluate current performance"""
        if len(self.episode_rewards) < 10:
            return
        
        # Calculate recent performance
        recent_rewards = list(self.episode_rewards)[-100:]
        recent_success = list(self.metrics.success_rates)[-100:]
        
        avg_reward = np.mean(recent_rewards)
        success_rate = np.mean(recent_success)
        
        # Update best performance
        current_performance = (avg_reward + success_rate * 100) / 2
        if current_performance > self.best_performance:
            self.best_performance = current_performance
            self._save_best_models()
        
        logger.info(f"Evaluation - Avg Reward: {avg_reward:.2f}, "
                   f"Success Rate: {success_rate:.2f}, "
                   f"Best Performance: {self.best_performance:.2f}")
    
    def _check_convergence(self) -> bool:
        """Check if training has converged"""
        if len(self.convergence_tracker) < self.config.convergence_window:
            return False
        
        # Check if performance has stabilized
        recent_performance = list(self.convergence_tracker)
        performance_std = np.std(recent_performance)
        
        converged = performance_std < self.config.convergence_threshold
        
        if converged:
            self.metrics.convergence_score = 1.0 - performance_std
        
        return converged
    
    def _save_models(self) -> None:
        """Save all models"""
        timestamp = int(time.time())
        
        # Save MADDPG
        if self.maddpg:
            maddpg_dir = os.path.join(self.config.model_save_dir, f"maddpg_{timestamp}")
            self.maddpg.save_models(maddpg_dir)
        
        # Save individual algorithms
        for agent_id, algorithm in self.individual_algorithms.items():
            if hasattr(algorithm, 'save_model'):
                model_path = os.path.join(
                    self.config.model_save_dir, 
                    f"{agent_id}_{timestamp}.pth"
                )
                algorithm.save_model(model_path)
        
        # Save training state
        self._save_training_state(timestamp)
        
        logger.info(f"Saved models at episode {self.current_episode}")
    
    def _save_best_models(self) -> None:
        """Save best performing models"""
        # Save MADDPG
        if self.maddpg:
            maddpg_dir = os.path.join(self.config.model_save_dir, "maddpg_best")
            self.maddpg.save_models(maddpg_dir)
        
        # Save individual algorithms
        for agent_id, algorithm in self.individual_algorithms.items():
            if hasattr(algorithm, 'save_model'):
                model_path = os.path.join(
                    self.config.model_save_dir, 
                    f"{agent_id}_best.pth"
                )
                algorithm.save_model(model_path)
        
        logger.info("Saved best models")
    
    def _save_training_state(self, timestamp: int) -> None:
        """Save training state"""
        state = {
            'config': asdict(self.config),
            'metrics': asdict(self.metrics),
            'current_episode': self.current_episode,
            'current_phase': self.current_phase.value,
            'best_performance': self.best_performance,
            'timestamp': timestamp
        }
        
        state_path = os.path.join(
            self.config.checkpoint_dir, 
            f"training_state_{timestamp}.json"
        )
        
        with open(state_path, 'w') as f:
            json.dump(state, f, indent=2, default=str)
    
    def _log_progress(self) -> None:
        """Log training progress"""
        if len(self.episode_rewards) < 1:
            return
        
        recent_rewards = list(self.episode_rewards)[-10:]
        recent_success = list(self.metrics.success_rates)[-10:] if self.metrics.success_rates else [0]
        
        avg_reward = np.mean(recent_rewards)
        success_rate = np.mean(recent_success)
        
        logger.info(
            f"Episode {self.current_episode} | "
            f"Phase: {self.current_phase.value} | "
            f"Avg Reward: {avg_reward:.2f} | "
            f"Success Rate: {success_rate:.2f} | "
            f"Buffer Size: {len(self.experience_buffer)}"
        )
    
    def get_training_status(self) -> Dict[str, Any]:
        """Get current training status"""
        status = {
            'is_training': self.is_training,
            'current_episode': self.current_episode,
            'current_phase': self.current_phase.value,
            'progress_percentage': (self.current_episode / self.config.max_episodes) * 100,
            'best_performance': self.best_performance,
            'training_time': self.metrics.training_time,
            'buffer_size': len(self.experience_buffer)
        }
        
        if len(self.episode_rewards) > 0:
            status['recent_avg_reward'] = np.mean(list(self.episode_rewards)[-10:])
        
        if len(self.metrics.success_rates) > 0:
            status['recent_success_rate'] = np.mean(self.metrics.success_rates[-10:])
        
        return status
    
    def get_detailed_metrics(self) -> Dict[str, Any]:
        """Get detailed training metrics"""
        detailed_metrics = asdict(self.metrics)
        
        # Add algorithm-specific metrics
        if self.maddpg:
            detailed_metrics['maddpg_metrics'] = self.maddpg.get_training_metrics()
        
        detailed_metrics['individual_agent_metrics'] = {}
        for agent_id, algorithm in self.individual_algorithms.items():
            if hasattr(algorithm, 'get_training_metrics'):
                detailed_metrics['individual_agent_metrics'][agent_id] = \
                    algorithm.get_training_metrics()
        
        # Add constellation manager metrics
        detailed_metrics['constellation_metrics'] = \
            self.constellation_manager.get_formation_metrics()
        
        return detailed_metrics
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load training checkpoint"""
        with open(checkpoint_path, 'r') as f:
            state = json.load(f)
        
        self.current_episode = state['current_episode']
        self.current_phase = TrainingPhase(state['current_phase'])
        self.best_performance = state['best_performance']
        
        logger.info(f"Loaded checkpoint from episode {self.current_episode}")
    
    def cleanup(self) -> None:
        """Cleanup resources"""
        if self.executor:
            self.executor.shutdown(wait=True)
        
        logger.info("Training manager cleanup completed")