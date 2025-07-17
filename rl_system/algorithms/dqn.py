"""Deep Q-Network (DQN) Implementation for GAAPF

This module implements DQN with various improvements including:
- Double DQN
- Dueling DQN
- Prioritized Experience Replay
- Noisy Networks
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Dict, List, Tuple, Any, Optional
from collections import deque, namedtuple
import random
import logging
from dataclasses import dataclass
import math

logger = logging.getLogger(__name__)

# Experience tuple for DQN
DQNExperience = namedtuple('DQNExperience', [
    'state', 'action', 'reward', 'next_state', 'done', 'priority'
])

@dataclass
class DQNConfig:
    """Configuration for DQN algorithm"""
    state_dim: int = 64
    action_dim: int = 10
    hidden_dim: int = 256
    lr: float = 1e-4
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995
    target_update_frequency: int = 1000
    buffer_size: int = 100000
    batch_size: int = 32
    warmup_steps: int = 1000
    double_dqn: bool = True
    dueling_dqn: bool = True
    prioritized_replay: bool = True
    noisy_networks: bool = False
    alpha: float = 0.6  # Prioritized replay alpha
    beta_start: float = 0.4  # Prioritized replay beta
    beta_end: float = 1.0
    device: str = 'cpu'

class NoisyLinear(nn.Module):
    """Noisy linear layer for exploration"""
    
    def __init__(self, in_features: int, out_features: int, std_init: float = 0.5):
        super(NoisyLinear, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        
        # Learnable parameters
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        
        # Noise buffers
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))
        
        self.reset_parameters()
        self.reset_noise()
    
    def reset_parameters(self):
        """Reset network parameters"""
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))
    
    def reset_noise(self):
        """Reset noise"""
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)
    
    def _scale_noise(self, size: int) -> torch.Tensor:
        """Scale noise"""
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        
        return F.linear(input, weight, bias)

class DuelingDQN(nn.Module):
    """Dueling DQN network"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256, 
                 noisy: bool = False):
        super(DuelingDQN, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.noisy = noisy
        
        # Shared feature extraction
        self.feature_layer = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Value stream
        if noisy:
            self.value_stream = nn.Sequential(
                NoisyLinear(hidden_dim, hidden_dim),
                nn.ReLU(),
                NoisyLinear(hidden_dim, 1)
            )
        else:
            self.value_stream = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )
        
        # Advantage stream
        if noisy:
            self.advantage_stream = nn.Sequential(
                NoisyLinear(hidden_dim, hidden_dim),
                nn.ReLU(),
                NoisyLinear(hidden_dim, action_dim)
            )
        else:
            self.advantage_stream = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dim)
            )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0.01)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        features = self.feature_layer(state)
        
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        
        # Combine value and advantage
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        
        return q_values
    
    def reset_noise(self):
        """Reset noise in noisy layers"""
        if self.noisy:
            for module in self.modules():
                if isinstance(module, NoisyLinear):
                    module.reset_noise()

class PrioritizedReplayBuffer:
    """Prioritized experience replay buffer"""
    
    def __init__(self, capacity: int, alpha: float = 0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        
    def push(self, experience: DQNExperience):
        """Add experience to buffer"""
        max_priority = self.priorities.max() if self.buffer else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.pos] = experience
        
        self.priorities[self.pos] = max_priority
        self.pos = (self.pos + 1) % self.capacity
    
    def sample(self, batch_size: int, beta: float = 0.4) -> Tuple[List[DQNExperience], np.ndarray, np.ndarray]:
        """Sample batch with priorities"""
        if len(self.buffer) == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:self.pos]
        
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        experiences = [self.buffer[idx] for idx in indices]
        
        # Importance sampling weights
        total = len(self.buffer)
        weights = (total * probabilities[indices]) ** (-beta)
        weights /= weights.max()
        
        return experiences, indices, weights
    
    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """Update priorities"""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
    
    def __len__(self) -> int:
        return len(self.buffer)

class SimpleReplayBuffer:
    """Simple replay buffer"""
    
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, experience: DQNExperience):
        """Add experience to buffer"""
        self.buffer.append(experience)
    
    def sample(self, batch_size: int) -> List[DQNExperience]:
        """Sample batch from buffer"""
        return random.sample(self.buffer, batch_size)
    
    def __len__(self) -> int:
        return len(self.buffer)

class DQNAgent:
    """Deep Q-Network Agent"""
    
    def __init__(self, 
                 agent_id: str,
                 state_dim: int,
                 action_dim: int,
                 config: DQNConfig):
        """
        Initialize DQN agent.
        
        Parameters:
        ----------
        agent_id : str
            Unique identifier for the agent
        state_dim : int
            Dimension of state space
        action_dim : int
            Dimension of action space
        config : DQNConfig
            Configuration parameters
        """
        self.agent_id = agent_id
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config
        
        self.device = torch.device(config.device)
        
        # Networks
        if config.dueling_dqn:
            self.q_network = DuelingDQN(
                state_dim, action_dim, config.hidden_dim, config.noisy_networks
            ).to(self.device)
            self.target_network = DuelingDQN(
                state_dim, action_dim, config.hidden_dim, config.noisy_networks
            ).to(self.device)
        else:
            # Standard DQN
            self.q_network = nn.Sequential(
                nn.Linear(state_dim, config.hidden_dim),
                nn.ReLU(),
                nn.Linear(config.hidden_dim, config.hidden_dim),
                nn.ReLU(),
                nn.Linear(config.hidden_dim, action_dim)
            ).to(self.device)
            
            self.target_network = nn.Sequential(
                nn.Linear(state_dim, config.hidden_dim),
                nn.ReLU(),
                nn.Linear(config.hidden_dim, config.hidden_dim),
                nn.ReLU(),
                nn.Linear(config.hidden_dim, action_dim)
            ).to(self.device)
        
        # Copy weights to target network
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=config.lr)
        
        # Replay buffer
        if config.prioritized_replay:
            self.replay_buffer = PrioritizedReplayBuffer(config.buffer_size, config.alpha)
            self.beta = config.beta_start
            self.beta_increment = (config.beta_end - config.beta_start) / 100000
        else:
            self.replay_buffer = SimpleReplayBuffer(config.buffer_size)
        
        # Exploration
        self.epsilon = config.epsilon_start
        
        # Training state
        self.training_step = 0
        self.target_update_counter = 0
        
        # Metrics
        self.training_metrics = {
            'losses': [],
            'q_values': [],
            'epsilon_values': [],
            'episode_rewards': [],
            'episode_lengths': []
        }
        
        logger.info(f"Initialized DQN agent {agent_id} with state_dim={state_dim}, action_dim={action_dim}")
    
    def act(self, state: np.ndarray, training: bool = True) -> int:
        """Select action using epsilon-greedy policy"""
        if training and (np.random.random() < self.epsilon or 
                        self.training_step < self.config.warmup_steps):
            # Random action
            return np.random.randint(0, self.action_dim)
        
        # Greedy action
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        self.q_network.eval()
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
            action = q_values.argmax().item()
        
        if training:
            self.q_network.train()
        
        return action
    
    def store_experience(self, state: np.ndarray, action: int, reward: float, 
                        next_state: np.ndarray, done: bool):
        """Store experience in replay buffer"""
        experience = DQNExperience(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
            priority=1.0  # Will be updated for prioritized replay
        )
        
        self.replay_buffer.push(experience)
    
    def update(self) -> Dict[str, float]:
        """Update Q-network"""
        if len(self.replay_buffer) < self.config.batch_size or \
           self.training_step < self.config.warmup_steps:
            return {}
        
        # Sample batch
        if self.config.prioritized_replay:
            experiences, indices, weights = self.replay_buffer.sample(
                self.config.batch_size, self.beta
            )
            weights = torch.FloatTensor(weights).to(self.device)
            self.beta = min(self.config.beta_end, self.beta + self.beta_increment)
        else:
            experiences = self.replay_buffer.sample(self.config.batch_size)
            weights = torch.ones(self.config.batch_size).to(self.device)
            indices = None
        
        # Convert to tensors
        states = torch.FloatTensor([e.state for e in experiences]).to(self.device)
        actions = torch.LongTensor([e.action for e in experiences]).to(self.device)
        rewards = torch.FloatTensor([e.reward for e in experiences]).to(self.device)
        next_states = torch.FloatTensor([e.next_state for e in experiences]).to(self.device)
        dones = torch.BoolTensor([e.done for e in experiences]).to(self.device)
        
        # Current Q-values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Next Q-values
        with torch.no_grad():
            if self.config.double_dqn:
                # Double DQN: use main network to select actions, target network to evaluate
                next_actions = self.q_network(next_states).argmax(1, keepdim=True)
                next_q_values = self.target_network(next_states).gather(1, next_actions)
            else:
                # Standard DQN
                next_q_values = self.target_network(next_states).max(1)[0].unsqueeze(1)
            
            target_q_values = rewards.unsqueeze(1) + \
                            (self.config.gamma * next_q_values * (~dones).unsqueeze(1).float())
        
        # Compute loss
        td_errors = target_q_values - current_q_values
        
        if self.config.prioritized_replay:
            # Weighted loss for prioritized replay
            loss = (weights.unsqueeze(1) * td_errors.pow(2)).mean()
            
            # Update priorities
            priorities = td_errors.abs().detach().cpu().numpy().flatten()
            priorities = np.clip(priorities, 1e-6, None)  # Avoid zero priorities
            self.replay_buffer.update_priorities(indices, priorities)
        else:
            loss = F.mse_loss(current_q_values, target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        # Reset noise in noisy networks
        if self.config.noisy_networks:
            self.q_network.reset_noise()
            self.target_network.reset_noise()
        
        # Update target network
        self.target_update_counter += 1
        if self.target_update_counter >= self.config.target_update_frequency:
            self.target_network.load_state_dict(self.q_network.state_dict())
            self.target_update_counter = 0
        
        # Update exploration
        if not self.config.noisy_networks:  # Only decay epsilon if not using noisy networks
            self.epsilon = max(self.config.epsilon_end, 
                             self.epsilon * self.config.epsilon_decay)
        
        self.training_step += 1
        
        # Update metrics
        self.training_metrics['losses'].append(loss.item())
        self.training_metrics['q_values'].append(current_q_values.mean().item())
        self.training_metrics['epsilon_values'].append(self.epsilon)
        
        return {
            'loss': loss.item(),
            'q_value': current_q_values.mean().item(),
            'epsilon': self.epsilon,
            'training_step': self.training_step
        }
    
    def save_model(self, filepath: str):
        """Save agent model"""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_step': self.training_step,
            'epsilon': self.epsilon,
            'config': self.config
        }, filepath)
    
    def load_model(self, filepath: str):
        """Load agent model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        self.training_step = checkpoint.get('training_step', 0)
        self.epsilon = checkpoint.get('epsilon', self.config.epsilon_start)
    
    def get_training_metrics(self) -> Dict[str, Any]:
        """Get training metrics"""
        metrics = self.training_metrics.copy()
        
        # Add summary statistics
        if self.training_metrics['losses']:
            metrics['avg_loss'] = np.mean(self.training_metrics['losses'][-100:])
        if self.training_metrics['q_values']:
            metrics['avg_q_value'] = np.mean(self.training_metrics['q_values'][-100:])
        
        metrics['training_step'] = self.training_step
        metrics['epsilon'] = self.epsilon
        metrics['buffer_size'] = len(self.replay_buffer)
        
        return metrics
    
    def reset_episode(self):
        """Reset for new episode"""
        if self.config.noisy_networks:
            self.q_network.reset_noise()
            self.target_network.reset_noise()
    
    def set_training_mode(self, training: bool = True):
        """Set training mode"""
        if training:
            self.q_network.train()
        else:
            self.q_network.eval()