"""Experience Buffer for GAAPF RL Implementation

This module manages the storage and retrieval of agent experiences
for training reinforcement learning models.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import deque
import pickle
import json
import logging
import random
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class Experience:
    """Represents a single experience tuple"""
    agent_id: str
    state: Dict[str, Any]
    action: Dict[str, Any]
    reward: float
    next_state: Dict[str, Any]
    done: bool
    timestamp: float
    episode_id: str
    step_id: int
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class Episode:
    """Represents a complete episode"""
    episode_id: str
    agent_id: str
    experiences: List[Experience]
    total_reward: float
    episode_length: int
    start_time: float
    end_time: float
    success: bool
    metadata: Optional[Dict[str, Any]] = None

class ExperienceBuffer:
    """Manages storage and sampling of agent experiences"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.max_size = config.get('max_buffer_size', 100000)
        self.min_size_for_sampling = config.get('min_size_for_sampling', 1000)
        self.save_path = Path(config.get('save_path', 'data/rl_experiences'))
        self.save_path.mkdir(parents=True, exist_ok=True)
        
        # Main experience buffer
        self.experiences: deque = deque(maxlen=self.max_size)
        
        # Episode tracking
        self.episodes: Dict[str, Episode] = {}
        self.current_episodes: Dict[str, List[Experience]] = {}
        
        # Prioritized sampling weights
        self.priorities: deque = deque(maxlen=self.max_size)
        self.alpha = config.get('priority_alpha', 0.6)  # Prioritization strength
        self.beta = config.get('priority_beta', 0.4)    # Importance sampling correction
        
        # Statistics
        self.stats = {
            'total_experiences': 0,
            'total_episodes': 0,
            'average_episode_length': 0,
            'average_reward': 0
        }
        
        logger.info(f"ExperienceBuffer initialized with max_size={self.max_size}")
    
    def add_experience(self, experience: Experience) -> None:
        """Add a single experience to the buffer"""
        self.experiences.append(experience)
        
        # Calculate priority (TD error proxy - using absolute reward for now)
        priority = abs(experience.reward) + 1e-6  # Small epsilon to avoid zero priority
        self.priorities.append(priority)
        
        # Track current episode
        if experience.agent_id not in self.current_episodes:
            self.current_episodes[experience.agent_id] = []
        self.current_episodes[experience.agent_id].append(experience)
        
        # If episode is done, finalize it
        if experience.done:
            self._finalize_episode(experience.agent_id, experience.episode_id)
        
        self.stats['total_experiences'] += 1
        self._update_stats()
        
        logger.debug(f"Added experience for agent {experience.agent_id}, step {experience.step_id}")
    
    def _finalize_episode(self, agent_id: str, episode_id: str) -> None:
        """Finalize a completed episode"""
        if agent_id not in self.current_episodes:
            return
        
        episode_experiences = self.current_episodes[agent_id]
        if not episode_experiences:
            return
        
        total_reward = sum(exp.reward for exp in episode_experiences)
        episode_length = len(episode_experiences)
        start_time = episode_experiences[0].timestamp
        end_time = episode_experiences[-1].timestamp
        success = episode_experiences[-1].reward > 0  # Simple success criterion
        
        episode = Episode(
            episode_id=episode_id,
            agent_id=agent_id,
            experiences=episode_experiences.copy(),
            total_reward=total_reward,
            episode_length=episode_length,
            start_time=start_time,
            end_time=end_time,
            success=success
        )
        
        self.episodes[episode_id] = episode
        self.current_episodes[agent_id] = []  # Reset for next episode
        
        self.stats['total_episodes'] += 1
        logger.info(f"Finalized episode {episode_id} for agent {agent_id}: "
                   f"length={episode_length}, reward={total_reward:.3f}, success={success}")
    
    def sample_batch(self, batch_size: int, 
                    prioritized: bool = True,
                    agent_id: Optional[str] = None) -> Tuple[List[Experience], np.ndarray, List[int]]:
        """Sample a batch of experiences"""
        if len(self.experiences) < self.min_size_for_sampling:
            raise ValueError(f"Not enough experiences for sampling. "
                           f"Have {len(self.experiences)}, need {self.min_size_for_sampling}")
        
        # Filter by agent if specified
        if agent_id:
            valid_indices = [i for i, exp in enumerate(self.experiences) 
                           if exp.agent_id == agent_id]
            if len(valid_indices) < batch_size:
                raise ValueError(f"Not enough experiences for agent {agent_id}")
        else:
            valid_indices = list(range(len(self.experiences)))
        
        if prioritized and len(self.priorities) == len(self.experiences):
            # Prioritized sampling
            priorities = np.array([self.priorities[i] for i in valid_indices])
            probabilities = priorities ** self.alpha
            probabilities /= probabilities.sum()
            
            # Sample indices
            sampled_indices = np.random.choice(
                valid_indices, size=batch_size, p=probabilities, replace=False
            )
            
            # Calculate importance sampling weights
            weights = (len(valid_indices) * probabilities[sampled_indices]) ** (-self.beta)
            weights /= weights.max()  # Normalize
        else:
            # Uniform sampling
            sampled_indices = random.sample(valid_indices, batch_size)
            weights = np.ones(batch_size)
        
        # Get experiences
        sampled_experiences = [self.experiences[i] for i in sampled_indices]
        
        return sampled_experiences, weights, sampled_indices
    
    def sample_episode_batch(self, batch_size: int, 
                           agent_id: Optional[str] = None) -> List[Episode]:
        """Sample a batch of complete episodes"""
        available_episodes = list(self.episodes.values())
        
        if agent_id:
            available_episodes = [ep for ep in available_episodes if ep.agent_id == agent_id]
        
        if len(available_episodes) < batch_size:
            raise ValueError(f"Not enough episodes for sampling. "
                           f"Have {len(available_episodes)}, need {batch_size}")
        
        return random.sample(available_episodes, batch_size)
    
    def update_priorities(self, indices: List[int], td_errors: np.ndarray) -> None:
        """Update priorities based on TD errors"""
        for idx, td_error in zip(indices, td_errors):
            if idx < len(self.priorities):
                self.priorities[idx] = abs(td_error) + 1e-6
    
    def get_agent_experiences(self, agent_id: str, 
                            limit: Optional[int] = None) -> List[Experience]:
        """Get all experiences for a specific agent"""
        agent_experiences = [exp for exp in self.experiences if exp.agent_id == agent_id]
        
        if limit:
            agent_experiences = agent_experiences[-limit:]  # Get most recent
        
        return agent_experiences
    
    def get_agent_episodes(self, agent_id: str) -> List[Episode]:
        """Get all episodes for a specific agent"""
        return [ep for ep in self.episodes.values() if ep.agent_id == agent_id]
    
    def get_successful_episodes(self, agent_id: Optional[str] = None) -> List[Episode]:
        """Get all successful episodes"""
        episodes = list(self.episodes.values())
        
        if agent_id:
            episodes = [ep for ep in episodes if ep.agent_id == agent_id]
        
        return [ep for ep in episodes if ep.success]
    
    def _update_stats(self) -> None:
        """Update buffer statistics"""
        if self.experiences:
            self.stats['average_reward'] = np.mean([exp.reward for exp in self.experiences])
        
        if self.episodes:
            episode_lengths = [ep.episode_length for ep in self.episodes.values()]
            self.stats['average_episode_length'] = np.mean(episode_lengths)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get buffer statistics"""
        return {
            **self.stats,
            'buffer_size': len(self.experiences),
            'episode_count': len(self.episodes),
            'buffer_utilization': len(self.experiences) / self.max_size
        }
    
    def save_to_disk(self, filename: Optional[str] = None) -> None:
        """Save buffer to disk"""
        if filename is None:
            filename = f"experience_buffer_{len(self.experiences)}.pkl"
        
        filepath = self.save_path / filename
        
        # Convert to serializable format
        data = {
            'experiences': [asdict(exp) for exp in self.experiences],
            'episodes': {k: asdict(v) for k, v in self.episodes.items()},
            'stats': self.stats,
            'config': self.config
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        logger.info(f"Saved experience buffer to {filepath}")
    
    def load_from_disk(self, filename: str) -> None:
        """Load buffer from disk"""
        filepath = self.save_path / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"Experience buffer file not found: {filepath}")
        
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        # Reconstruct experiences
        self.experiences.clear()
        for exp_dict in data['experiences']:
            exp = Experience(**exp_dict)
            self.experiences.append(exp)
        
        # Reconstruct episodes
        self.episodes.clear()
        for ep_id, ep_dict in data['episodes'].items():
            # Reconstruct experiences in episode
            ep_dict['experiences'] = [Experience(**exp_dict) for exp_dict in ep_dict['experiences']]
            episode = Episode(**ep_dict)
            self.episodes[ep_id] = episode
        
        self.stats = data['stats']
        
        # Rebuild priorities (simple approach)
        self.priorities.clear()
        for exp in self.experiences:
            priority = abs(exp.reward) + 1e-6
            self.priorities.append(priority)
        
        logger.info(f"Loaded experience buffer from {filepath}: "
                   f"{len(self.experiences)} experiences, {len(self.episodes)} episodes")
    
    def clear(self) -> None:
        """Clear all experiences and episodes"""
        self.experiences.clear()
        self.episodes.clear()
        self.current_episodes.clear()
        self.priorities.clear()
        
        self.stats = {
            'total_experiences': 0,
            'total_episodes': 0,
            'average_episode_length': 0,
            'average_reward': 0
        }
        
        logger.info("Experience buffer cleared")
    
    def export_for_analysis(self, filename: Optional[str] = None) -> None:
        """Export buffer data for analysis"""
        if filename is None:
            filename = f"experience_analysis_{len(self.experiences)}.json"
        
        filepath = self.save_path / filename
        
        # Prepare data for JSON export
        analysis_data = {
            'stats': self.get_stats(),
            'agent_performance': {},
            'episode_summary': []
        }
        
        # Agent performance summary
        agent_ids = set(exp.agent_id for exp in self.experiences)
        for agent_id in agent_ids:
            agent_experiences = self.get_agent_experiences(agent_id)
            agent_episodes = self.get_agent_episodes(agent_id)
            
            analysis_data['agent_performance'][agent_id] = {
                'total_experiences': len(agent_experiences),
                'total_episodes': len(agent_episodes),
                'average_reward': np.mean([exp.reward for exp in agent_experiences]) if agent_experiences else 0,
                'success_rate': sum(1 for ep in agent_episodes if ep.success) / len(agent_episodes) if agent_episodes else 0
            }
        
        # Episode summary
        for episode in self.episodes.values():
            analysis_data['episode_summary'].append({
                'episode_id': episode.episode_id,
                'agent_id': episode.agent_id,
                'total_reward': episode.total_reward,
                'episode_length': episode.episode_length,
                'success': episode.success,
                'duration': episode.end_time - episode.start_time
            })
        
        with open(filepath, 'w') as f:
            json.dump(analysis_data, f, indent=2)
        
        logger.info(f"Exported analysis data to {filepath}")