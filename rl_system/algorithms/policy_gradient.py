"""Policy Gradient Algorithms for GAAPF

This module implements various policy gradient methods including:
- REINFORCE
- Actor-Critic
- Proximal Policy Optimization (PPO)
- Trust Region Policy Optimization (TRPO)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
from typing import Dict, List, Tuple, Any, Optional, Union
from collections import deque, namedtuple
import logging
from dataclasses import dataclass
import math

logger = logging.getLogger(__name__)

# Experience tuple for Policy Gradient
PGExperience = namedtuple('PGExperience', [
    'state', 'action', 'reward', 'log_prob', 'value', 'done'
])

@dataclass
class PolicyGradientConfig:
    """Configuration for Policy Gradient algorithms"""
    state_dim: int = 64
    action_dim: int = 10
    hidden_dim: int = 256
    lr_actor: float = 3e-4
    lr_critic: float = 1e-3
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    entropy_coeff: float = 0.01
    value_loss_coeff: float = 0.5
    max_grad_norm: float = 0.5
    ppo_epochs: int = 4
    batch_size: int = 64
    buffer_size: int = 2048
    continuous_actions: bool = False
    action_std: float = 0.5
    device: str = 'cpu'

class PolicyNetwork(nn.Module):
    """Policy network for discrete or continuous actions"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256, 
                 continuous: bool = False):
        super(PolicyNetwork, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.continuous = continuous
        
        # Shared layers
        self.shared_layers = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        if continuous:
            # Continuous actions: output mean and log_std
            self.mean_layer = nn.Linear(hidden_dim, action_dim)
            self.log_std_layer = nn.Linear(hidden_dim, action_dim)
        else:
            # Discrete actions: output action probabilities
            self.action_layer = nn.Linear(hidden_dim, action_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0.0)
        
        # Initialize final layer with smaller weights
        if self.continuous:
            nn.init.orthogonal_(self.mean_layer.weight, gain=0.01)
            nn.init.orthogonal_(self.log_std_layer.weight, gain=0.01)
        else:
            nn.init.orthogonal_(self.action_layer.weight, gain=0.01)
    
    def forward(self, state: torch.Tensor) -> Union[Categorical, Normal]:
        """Forward pass returning action distribution"""
        features = self.shared_layers(state)
        
        if self.continuous:
            mean = self.mean_layer(features)
            log_std = self.log_std_layer(features)
            std = torch.exp(torch.clamp(log_std, -20, 2))
            return Normal(mean, std)
        else:
            logits = self.action_layer(features)
            return Categorical(logits=logits)
    
    def get_action_and_log_prob(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get action and log probability"""
        dist = self.forward(state)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        if self.continuous:
            log_prob = log_prob.sum(dim=-1)  # Sum over action dimensions
        
        return action, log_prob
    
    def get_log_prob_and_entropy(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get log probability and entropy for given state-action pairs"""
        dist = self.forward(state)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        
        if self.continuous:
            log_prob = log_prob.sum(dim=-1)
            entropy = entropy.sum(dim=-1)
        
        return log_prob, entropy

class ValueNetwork(nn.Module):
    """Value network for critic"""
    
    def __init__(self, state_dim: int, hidden_dim: int = 256):
        super(ValueNetwork, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0.0)
        
        # Initialize final layer with smaller weights
        nn.init.orthogonal_(self.network[-1].weight, gain=1.0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        return self.network(state)

class RolloutBuffer:
    """Buffer for storing rollout data"""
    
    def __init__(self, buffer_size: int, state_dim: int, action_dim: int, 
                 continuous_actions: bool = False):
        self.buffer_size = buffer_size
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.continuous_actions = continuous_actions
        
        self.states = np.zeros((buffer_size, state_dim), dtype=np.float32)
        
        if continuous_actions:
            self.actions = np.zeros((buffer_size, action_dim), dtype=np.float32)
        else:
            self.actions = np.zeros(buffer_size, dtype=np.int64)
        
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.log_probs = np.zeros(buffer_size, dtype=np.float32)
        self.values = np.zeros(buffer_size, dtype=np.float32)
        self.dones = np.zeros(buffer_size, dtype=np.bool_)
        self.advantages = np.zeros(buffer_size, dtype=np.float32)
        self.returns = np.zeros(buffer_size, dtype=np.float32)
        
        self.ptr = 0
        self.size = 0
    
    def store(self, state: np.ndarray, action: Union[int, np.ndarray], reward: float,
              log_prob: float, value: float, done: bool):
        """Store experience"""
        assert self.ptr < self.buffer_size
        
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.log_probs[self.ptr] = log_prob
        self.values[self.ptr] = value
        self.dones[self.ptr] = done
        
        self.ptr += 1
        self.size = max(self.size, self.ptr)
    
    def compute_advantages_and_returns(self, last_value: float, gamma: float, 
                                     gae_lambda: float):
        """Compute advantages and returns using GAE"""
        advantages = np.zeros_like(self.rewards)
        last_gae_lam = 0
        
        for step in reversed(range(self.ptr)):
            if step == self.ptr - 1:
                next_non_terminal = 1.0 - self.dones[step]
                next_value = last_value
            else:
                next_non_terminal = 1.0 - self.dones[step]
                next_value = self.values[step + 1]
            
            delta = self.rewards[step] + gamma * next_value * next_non_terminal - self.values[step]
            advantages[step] = last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam
        
        self.advantages[:self.ptr] = advantages
        self.returns[:self.ptr] = advantages + self.values[:self.ptr]
    
    def get_batch(self, batch_size: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """Get batch of experiences"""
        if batch_size is None:
            indices = np.arange(self.ptr)
        else:
            indices = np.random.choice(self.ptr, batch_size, replace=False)
        
        return {
            'states': torch.FloatTensor(self.states[indices]),
            'actions': torch.LongTensor(self.actions[indices]) if not self.continuous_actions 
                      else torch.FloatTensor(self.actions[indices]),
            'log_probs': torch.FloatTensor(self.log_probs[indices]),
            'values': torch.FloatTensor(self.values[indices]),
            'advantages': torch.FloatTensor(self.advantages[indices]),
            'returns': torch.FloatTensor(self.returns[indices])
        }
    
    def clear(self):
        """Clear buffer"""
        self.ptr = 0
        self.size = 0

class PolicyGradientAgent:
    """Policy Gradient Agent with multiple algorithm support"""
    
    def __init__(self, 
                 agent_id: str,
                 state_dim: int,
                 action_dim: int,
                 config: PolicyGradientConfig,
                 algorithm: str = 'ppo'):
        """
        Initialize Policy Gradient agent.
        
        Parameters:
        ----------
        agent_id : str
            Unique identifier for the agent
        state_dim : int
            Dimension of state space
        action_dim : int
            Dimension of action space
        config : PolicyGradientConfig
            Configuration parameters
        algorithm : str
            Algorithm to use ('reinforce', 'actor_critic', 'ppo')
        """
        self.agent_id = agent_id
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config
        self.algorithm = algorithm
        
        self.device = torch.device(config.device)
        
        # Networks
        self.policy_network = PolicyNetwork(
            state_dim, action_dim, config.hidden_dim, config.continuous_actions
        ).to(self.device)
        
        if algorithm in ['actor_critic', 'ppo']:
            self.value_network = ValueNetwork(
                state_dim, config.hidden_dim
            ).to(self.device)
        
        # Optimizers
        self.policy_optimizer = optim.Adam(
            self.policy_network.parameters(), lr=config.lr_actor
        )
        
        if hasattr(self, 'value_network'):
            self.value_optimizer = optim.Adam(
                self.value_network.parameters(), lr=config.lr_critic
            )
        
        # Rollout buffer
        self.rollout_buffer = RolloutBuffer(
            config.buffer_size, state_dim, action_dim, config.continuous_actions
        )
        
        # Training state
        self.training_step = 0
        self.episode_count = 0
        
        # Metrics
        self.training_metrics = {
            'policy_losses': [],
            'value_losses': [],
            'entropies': [],
            'episode_rewards': [],
            'episode_lengths': []
        }
        
        logger.info(f"Initialized {algorithm.upper()} agent {agent_id} with "
                   f"state_dim={state_dim}, action_dim={action_dim}")
    
    def act(self, state: np.ndarray, training: bool = True) -> Tuple[Union[int, np.ndarray], float, float]:
        """Select action and get value estimate"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action, log_prob = self.policy_network.get_action_and_log_prob(state_tensor)
            
            if hasattr(self, 'value_network'):
                value = self.value_network(state_tensor)
            else:
                value = torch.tensor(0.0)
        
        if self.config.continuous_actions:
            action_np = action.cpu().numpy()[0]
        else:
            action_np = action.item()
        
        return action_np, log_prob.item(), value.item()
    
    def store_experience(self, state: np.ndarray, action: Union[int, np.ndarray], 
                        reward: float, log_prob: float, value: float, done: bool):
        """Store experience in rollout buffer"""
        self.rollout_buffer.store(state, action, reward, log_prob, value, done)
    
    def update(self, last_value: float = 0.0) -> Dict[str, float]:
        """Update policy and value networks"""
        if self.rollout_buffer.ptr == 0:
            return {}
        
        # Compute advantages and returns
        self.rollout_buffer.compute_advantages_and_returns(
            last_value, self.config.gamma, self.config.gae_lambda
        )
        
        if self.algorithm == 'reinforce':
            return self._update_reinforce()
        elif self.algorithm == 'actor_critic':
            return self._update_actor_critic()
        elif self.algorithm == 'ppo':
            return self._update_ppo()
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")
    
    def _update_reinforce(self) -> Dict[str, float]:
        """Update using REINFORCE algorithm"""
        batch = self.rollout_buffer.get_batch()
        
        states = batch['states'].to(self.device)
        actions = batch['actions'].to(self.device)
        returns = batch['returns'].to(self.device)
        
        # Normalize returns
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Compute policy loss
        log_probs, entropy = self.policy_network.get_log_prob_and_entropy(states, actions)
        policy_loss = -(log_probs * returns).mean()
        
        # Add entropy bonus
        entropy_loss = -self.config.entropy_coeff * entropy.mean()
        total_loss = policy_loss + entropy_loss
        
        # Update policy
        self.policy_optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), 
                                     self.config.max_grad_norm)
        self.policy_optimizer.step()
        
        # Clear buffer
        self.rollout_buffer.clear()
        
        # Update metrics
        self.training_metrics['policy_losses'].append(policy_loss.item())
        self.training_metrics['entropies'].append(entropy.mean().item())
        
        return {
            'policy_loss': policy_loss.item(),
            'entropy': entropy.mean().item()
        }
    
    def _update_actor_critic(self) -> Dict[str, float]:
        """Update using Actor-Critic algorithm"""
        batch = self.rollout_buffer.get_batch()
        
        states = batch['states'].to(self.device)
        actions = batch['actions'].to(self.device)
        advantages = batch['advantages'].to(self.device)
        returns = batch['returns'].to(self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Compute policy loss
        log_probs, entropy = self.policy_network.get_log_prob_and_entropy(states, actions)
        policy_loss = -(log_probs * advantages).mean()
        
        # Compute value loss
        values = self.value_network(states).squeeze()
        value_loss = F.mse_loss(values, returns)
        
        # Total loss
        entropy_loss = -self.config.entropy_coeff * entropy.mean()
        total_policy_loss = policy_loss + entropy_loss
        
        # Update policy
        self.policy_optimizer.zero_grad()
        total_policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), 
                                     self.config.max_grad_norm)
        self.policy_optimizer.step()
        
        # Update value network
        self.value_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.value_network.parameters(), 
                                     self.config.max_grad_norm)
        self.value_optimizer.step()
        
        # Clear buffer
        self.rollout_buffer.clear()
        
        # Update metrics
        self.training_metrics['policy_losses'].append(policy_loss.item())
        self.training_metrics['value_losses'].append(value_loss.item())
        self.training_metrics['entropies'].append(entropy.mean().item())
        
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.mean().item()
        }
    
    def _update_ppo(self) -> Dict[str, float]:
        """Update using PPO algorithm"""
        batch = self.rollout_buffer.get_batch()
        
        states = batch['states'].to(self.device)
        actions = batch['actions'].to(self.device)
        old_log_probs = batch['log_probs'].to(self.device)
        advantages = batch['advantages'].to(self.device)
        returns = batch['returns'].to(self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        policy_losses = []
        value_losses = []
        entropies = []
        
        # PPO epochs
        for _ in range(self.config.ppo_epochs):
            # Sample mini-batch
            batch_indices = torch.randperm(len(states))[:self.config.batch_size]
            
            batch_states = states[batch_indices]
            batch_actions = actions[batch_indices]
            batch_old_log_probs = old_log_probs[batch_indices]
            batch_advantages = advantages[batch_indices]
            batch_returns = returns[batch_indices]
            
            # Compute current policy log probs and entropy
            log_probs, entropy = self.policy_network.get_log_prob_and_entropy(
                batch_states, batch_actions
            )
            
            # Compute ratio
            ratio = torch.exp(log_probs - batch_old_log_probs)
            
            # Compute surrogate losses
            surr1 = ratio * batch_advantages
            surr2 = torch.clamp(ratio, 1.0 - self.config.clip_epsilon, 
                              1.0 + self.config.clip_epsilon) * batch_advantages
            
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Compute value loss
            values = self.value_network(batch_states).squeeze()
            value_loss = F.mse_loss(values, batch_returns)
            
            # Total loss
            entropy_loss = -self.config.entropy_coeff * entropy.mean()
            total_loss = (policy_loss + 
                         self.config.value_loss_coeff * value_loss + 
                         entropy_loss)
            
            # Update networks
            self.policy_optimizer.zero_grad()
            self.value_optimizer.zero_grad()
            total_loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), 
                                         self.config.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(self.value_network.parameters(), 
                                         self.config.max_grad_norm)
            
            self.policy_optimizer.step()
            self.value_optimizer.step()
            
            policy_losses.append(policy_loss.item())
            value_losses.append(value_loss.item())
            entropies.append(entropy.mean().item())
        
        # Clear buffer
        self.rollout_buffer.clear()
        
        # Update metrics
        avg_policy_loss = np.mean(policy_losses)
        avg_value_loss = np.mean(value_losses)
        avg_entropy = np.mean(entropies)
        
        self.training_metrics['policy_losses'].append(avg_policy_loss)
        self.training_metrics['value_losses'].append(avg_value_loss)
        self.training_metrics['entropies'].append(avg_entropy)
        
        return {
            'policy_loss': avg_policy_loss,
            'value_loss': avg_value_loss,
            'entropy': avg_entropy
        }
    
    def save_model(self, filepath: str):
        """Save agent model"""
        save_dict = {
            'policy_network_state_dict': self.policy_network.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'training_step': self.training_step,
            'episode_count': self.episode_count,
            'config': self.config,
            'algorithm': self.algorithm
        }
        
        if hasattr(self, 'value_network'):
            save_dict['value_network_state_dict'] = self.value_network.state_dict()
            save_dict['value_optimizer_state_dict'] = self.value_optimizer.state_dict()
        
        torch.save(save_dict, filepath)
    
    def load_model(self, filepath: str):
        """Load agent model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.policy_network.load_state_dict(checkpoint['policy_network_state_dict'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
        
        if hasattr(self, 'value_network') and 'value_network_state_dict' in checkpoint:
            self.value_network.load_state_dict(checkpoint['value_network_state_dict'])
            self.value_optimizer.load_state_dict(checkpoint['value_optimizer_state_dict'])
        
        self.training_step = checkpoint.get('training_step', 0)
        self.episode_count = checkpoint.get('episode_count', 0)
    
    def get_training_metrics(self) -> Dict[str, Any]:
        """Get training metrics"""
        metrics = self.training_metrics.copy()
        
        # Add summary statistics
        if self.training_metrics['policy_losses']:
            metrics['avg_policy_loss'] = np.mean(self.training_metrics['policy_losses'][-100:])
        if self.training_metrics['value_losses']:
            metrics['avg_value_loss'] = np.mean(self.training_metrics['value_losses'][-100:])
        if self.training_metrics['entropies']:
            metrics['avg_entropy'] = np.mean(self.training_metrics['entropies'][-100:])
        
        metrics['training_step'] = self.training_step
        metrics['episode_count'] = self.episode_count
        metrics['buffer_size'] = self.rollout_buffer.size
        
        return metrics
    
    def reset_episode(self):
        """Reset for new episode"""
        self.episode_count += 1
    
    def set_training_mode(self, training: bool = True):
        """Set training mode"""
        if training:
            self.policy_network.train()
            if hasattr(self, 'value_network'):
                self.value_network.train()
        else:
            self.policy_network.eval()
            if hasattr(self, 'value_network'):
                self.value_network.eval()