"""Multi-Agent Deep Deterministic Policy Gradient (MADDPG) Implementation

This module implements MADDPG for multi-agent coordination in the GAAPF system.
MADDPG is particularly suitable for environments with multiple agents that need
to learn coordinated behaviors.
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
import copy

logger = logging.getLogger(__name__)

# Experience tuple for MADDPG
MADDPGExperience = namedtuple('MADDPGExperience', [
    'states', 'actions', 'rewards', 'next_states', 'dones'
])

@dataclass
class MADDPGConfig:
    """Configuration for MADDPG algorithm"""
    state_dim: int = 64
    action_dim: int = 32
    hidden_dim: int = 256
    lr_actor: float = 1e-4
    lr_critic: float = 1e-3
    gamma: float = 0.95
    tau: float = 0.01
    buffer_size: int = 100000
    batch_size: int = 64
    noise_std: float = 0.1
    noise_decay: float = 0.995
    min_noise_std: float = 0.01
    update_frequency: int = 1
    warmup_steps: int = 1000
    device: str = 'cpu'

class OUNoise:
    """Ornstein-Uhlenbeck noise for action exploration"""
    
    def __init__(self, size: int, mu: float = 0.0, theta: float = 0.15, sigma: float = 0.2):
        self.size = size
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.size) * self.mu
        
    def reset(self):
        """Reset noise state"""
        self.state = np.ones(self.size) * self.mu
        
    def sample(self) -> np.ndarray:
        """Sample noise"""
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(self.size)
        self.state += dx
        return self.state

class Actor(nn.Module):
    """Actor network for MADDPG"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super(Actor, self).__init__()
        
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, action_dim)
        
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        
        self.dropout = nn.Dropout(0.1)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize network weights"""
        for layer in [self.fc1, self.fc2, self.fc3]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.constant_(layer.bias, 0.01)
        
        # Final layer with smaller weights
        nn.init.uniform_(self.fc4.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.fc4.bias, -3e-3, 3e-3)
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        x = state
        
        # Handle batch normalization for different batch sizes
        if x.size(0) > 1:
            x = F.relu(self.bn1(self.fc1(x)))
            x = self.dropout(x)
            x = F.relu(self.bn2(self.fc2(x)))
            x = self.dropout(x)
            x = F.relu(self.bn3(self.fc3(x)))
        else:
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            x = F.relu(self.fc2(x))
            x = self.dropout(x)
            x = F.relu(self.fc3(x))
        
        x = torch.tanh(self.fc4(x))
        return x

class Critic(nn.Module):
    """Critic network for MADDPG"""
    
    def __init__(self, total_state_dim: int, total_action_dim: int, hidden_dim: int = 256):
        super(Critic, self).__init__()
        
        # State processing
        self.state_fc1 = nn.Linear(total_state_dim, hidden_dim)
        self.state_fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        # Action processing
        self.action_fc1 = nn.Linear(total_action_dim, hidden_dim)
        
        # Combined processing
        self.combined_fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.combined_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.output_fc = nn.Linear(hidden_dim, 1)
        
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        self.bn4 = nn.BatchNorm1d(hidden_dim)
        
        self.dropout = nn.Dropout(0.1)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize network weights"""
        for layer in [self.state_fc1, self.state_fc2, self.action_fc1, 
                     self.combined_fc1, self.combined_fc2]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.constant_(layer.bias, 0.01)
        
        # Final layer with smaller weights
        nn.init.uniform_(self.output_fc.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.output_fc.bias, -3e-3, 3e-3)
        
    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        # Process states
        if states.size(0) > 1:
            state_out = F.relu(self.bn1(self.state_fc1(states)))
            state_out = self.dropout(state_out)
            state_out = F.relu(self.bn2(self.state_fc2(state_out)))
        else:
            state_out = F.relu(self.state_fc1(states))
            state_out = self.dropout(state_out)
            state_out = F.relu(self.state_fc2(state_out))
        
        # Process actions
        if actions.size(0) > 1:
            action_out = F.relu(self.bn3(self.action_fc1(actions)))
        else:
            action_out = F.relu(self.action_fc1(actions))
        
        # Combine state and action representations
        combined = torch.cat([state_out, action_out], dim=1)
        
        if combined.size(0) > 1:
            x = F.relu(self.bn4(self.combined_fc1(combined)))
            x = self.dropout(x)
            x = F.relu(self.combined_fc2(x))
        else:
            x = F.relu(self.combined_fc1(combined))
            x = self.dropout(x)
            x = F.relu(self.combined_fc2(x))
        
        q_value = self.output_fc(x)
        return q_value

class ReplayBuffer:
    """Replay buffer for MADDPG"""
    
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity
        
    def push(self, experience: MADDPGExperience):
        """Add experience to buffer"""
        self.buffer.append(experience)
        
    def sample(self, batch_size: int) -> MADDPGExperience:
        """Sample batch from buffer"""
        batch = random.sample(self.buffer, batch_size)
        
        # Transpose batch
        states = torch.FloatTensor([e.states for e in batch])
        actions = torch.FloatTensor([e.actions for e in batch])
        rewards = torch.FloatTensor([e.rewards for e in batch])
        next_states = torch.FloatTensor([e.next_states for e in batch])
        dones = torch.BoolTensor([e.dones for e in batch])
        
        return MADDPGExperience(states, actions, rewards, next_states, dones)
    
    def __len__(self) -> int:
        return len(self.buffer)

class MADDPGAgent:
    """Individual agent in MADDPG"""
    
    def __init__(self, 
                 agent_id: str,
                 state_dim: int,
                 action_dim: int,
                 total_state_dim: int,
                 total_action_dim: int,
                 config: MADDPGConfig):
        """
        Initialize MADDPG agent.
        
        Parameters:
        ----------
        agent_id : str
            Unique identifier for the agent
        state_dim : int
            Dimension of individual agent's state
        action_dim : int
            Dimension of individual agent's action
        total_state_dim : int
            Dimension of global state (all agents)
        total_action_dim : int
            Dimension of global action (all agents)
        config : MADDPGConfig
            Configuration parameters
        """
        self.agent_id = agent_id
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.total_state_dim = total_state_dim
        self.total_action_dim = total_action_dim
        self.config = config
        
        self.device = torch.device(config.device)
        
        # Networks
        self.actor = Actor(state_dim, action_dim, config.hidden_dim).to(self.device)
        self.actor_target = Actor(state_dim, action_dim, config.hidden_dim).to(self.device)
        self.critic = Critic(total_state_dim, total_action_dim, config.hidden_dim).to(self.device)
        self.critic_target = Critic(total_state_dim, total_action_dim, config.hidden_dim).to(self.device)
        
        # Copy weights to target networks
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config.lr_critic)
        
        # Noise for exploration
        self.noise = OUNoise(action_dim)
        self.noise_std = config.noise_std
        
        # Training state
        self.training_step = 0
        
    def act(self, state: np.ndarray, add_noise: bool = True) -> np.ndarray:
        """Select action given state"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state_tensor).cpu().numpy()[0]
        self.actor.train()
        
        if add_noise:
            noise = self.noise.sample() * self.noise_std
            action = np.clip(action + noise, -1.0, 1.0)
        
        return action
    
    def update_critic(self, 
                     states: torch.Tensor,
                     actions: torch.Tensor,
                     rewards: torch.Tensor,
                     next_states: torch.Tensor,
                     next_actions: torch.Tensor,
                     dones: torch.Tensor) -> float:
        """Update critic network"""
        # Compute target Q-values
        with torch.no_grad():
            target_q = self.critic_target(next_states, next_actions)
            target_q = rewards + (self.config.gamma * target_q * (~dones).float())
        
        # Compute current Q-values
        current_q = self.critic(states, actions)
        
        # Critic loss
        critic_loss = F.mse_loss(current_q, target_q)
        
        # Update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()
        
        return critic_loss.item()
    
    def update_actor(self, states: torch.Tensor, agent_actions: List[torch.Tensor]) -> float:
        """Update actor network"""
        # Compute actor loss
        actions = torch.cat(agent_actions, dim=1)
        actor_loss = -self.critic(states, actions).mean()
        
        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()
        
        return actor_loss.item()
    
    def soft_update(self):
        """Soft update of target networks"""
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.config.tau * param.data + (1.0 - self.config.tau) * target_param.data)
        
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.config.tau * param.data + (1.0 - self.config.tau) * target_param.data)
    
    def decay_noise(self):
        """Decay exploration noise"""
        self.noise_std = max(self.config.min_noise_std, 
                           self.noise_std * self.config.noise_decay)
    
    def reset_noise(self):
        """Reset noise state"""
        self.noise.reset()
    
    def save_model(self, filepath: str):
        """Save agent model"""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'training_step': self.training_step,
            'noise_std': self.noise_std
        }, filepath)
    
    def load_model(self, filepath: str):
        """Load agent model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        
        self.training_step = checkpoint.get('training_step', 0)
        self.noise_std = checkpoint.get('noise_std', self.config.noise_std)
        
        # Update target networks
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

class MADDPG:
    """Multi-Agent Deep Deterministic Policy Gradient"""
    
    def __init__(self, 
                 agent_configs: Dict[str, Dict[str, int]],
                 config: MADDPGConfig):
        """
        Initialize MADDPG.
        
        Parameters:
        ----------
        agent_configs : Dict[str, Dict[str, int]]
            Configuration for each agent: {agent_id: {'state_dim': int, 'action_dim': int}}
        config : MADDPGConfig
            Global configuration
        """
        self.config = config
        self.agent_configs = agent_configs
        
        # Calculate total dimensions
        self.total_state_dim = sum(cfg['state_dim'] for cfg in agent_configs.values())
        self.total_action_dim = sum(cfg['action_dim'] for cfg in agent_configs.values())
        
        # Create agents
        self.agents = {}
        for agent_id, agent_config in agent_configs.items():
            self.agents[agent_id] = MADDPGAgent(
                agent_id=agent_id,
                state_dim=agent_config['state_dim'],
                action_dim=agent_config['action_dim'],
                total_state_dim=self.total_state_dim,
                total_action_dim=self.total_action_dim,
                config=config
            )
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(config.buffer_size)
        
        # Training state
        self.training_step = 0
        self.episode_count = 0
        
        # Metrics
        self.training_metrics = {
            'actor_losses': {agent_id: [] for agent_id in self.agents.keys()},
            'critic_losses': {agent_id: [] for agent_id in self.agents.keys()},
            'episode_rewards': [],
            'episode_lengths': []
        }
        
        logger.info(f"Initialized MADDPG with {len(self.agents)} agents")
        logger.info(f"Total state dim: {self.total_state_dim}, Total action dim: {self.total_action_dim}")
    
    def act(self, states: Dict[str, np.ndarray], add_noise: bool = True) -> Dict[str, np.ndarray]:
        """Get actions for all agents"""
        actions = {}
        for agent_id, state in states.items():
            if agent_id in self.agents:
                actions[agent_id] = self.agents[agent_id].act(state, add_noise)
            else:
                logger.warning(f"Unknown agent ID: {agent_id}")
                actions[agent_id] = np.zeros(self.agent_configs.get(agent_id, {}).get('action_dim', 1))
        
        return actions
    
    def store_experience(self, 
                        states: Dict[str, np.ndarray],
                        actions: Dict[str, np.ndarray],
                        rewards: Dict[str, float],
                        next_states: Dict[str, np.ndarray],
                        dones: Dict[str, bool]):
        """Store experience in replay buffer"""
        # Convert to global format
        global_states = np.concatenate([states[agent_id] for agent_id in self.agents.keys()])
        global_actions = np.concatenate([actions[agent_id] for agent_id in self.agents.keys()])
        global_rewards = np.array([rewards[agent_id] for agent_id in self.agents.keys()])
        global_next_states = np.concatenate([next_states[agent_id] for agent_id in self.agents.keys()])
        global_dones = np.array([dones[agent_id] for agent_id in self.agents.keys()])
        
        experience = MADDPGExperience(
            states=global_states,
            actions=global_actions,
            rewards=global_rewards,
            next_states=global_next_states,
            dones=global_dones
        )
        
        self.replay_buffer.push(experience)
    
    def update(self) -> Dict[str, float]:
        """Update all agents"""
        if len(self.replay_buffer) < self.config.batch_size or \
           self.training_step < self.config.warmup_steps:
            return {}
        
        # Sample batch
        batch = self.replay_buffer.sample(self.config.batch_size)
        
        states = batch.states.to(torch.device(self.config.device))
        actions = batch.actions.to(torch.device(self.config.device))
        rewards = batch.rewards.to(torch.device(self.config.device))
        next_states = batch.next_states.to(torch.device(self.config.device))
        dones = batch.dones.to(torch.device(self.config.device))
        
        # Get next actions from target actors
        next_actions = []
        action_start = 0
        for agent_id in self.agents.keys():
            agent = self.agents[agent_id]
            state_start = sum(self.agent_configs[aid]['state_dim'] 
                            for aid in list(self.agents.keys())[:list(self.agents.keys()).index(agent_id)])
            state_end = state_start + self.agent_configs[agent_id]['state_dim']
            
            agent_next_states = next_states[:, state_start:state_end]
            agent_next_actions = agent.actor_target(agent_next_states)
            next_actions.append(agent_next_actions)
        
        next_actions_tensor = torch.cat(next_actions, dim=1)
        
        # Update critics
        critic_losses = {}
        for agent_id in self.agents.keys():
            agent = self.agents[agent_id]
            agent_rewards = rewards[:, list(self.agents.keys()).index(agent_id)].unsqueeze(1)
            agent_dones = dones[:, list(self.agents.keys()).index(agent_id)].unsqueeze(1)
            
            critic_loss = agent.update_critic(
                states, actions, agent_rewards, next_states, next_actions_tensor, agent_dones
            )
            critic_losses[agent_id] = critic_loss
        
        # Update actors
        actor_losses = {}
        for agent_id in self.agents.keys():
            agent = self.agents[agent_id]
            
            # Get current actions from all agents
            current_actions = []
            for aid in self.agents.keys():
                if aid == agent_id:
                    # Use current agent's actor
                    state_start = sum(self.agent_configs[aid2]['state_dim'] 
                                    for aid2 in list(self.agents.keys())[:list(self.agents.keys()).index(aid)])
                    state_end = state_start + self.agent_configs[aid]['state_dim']
                    agent_states = states[:, state_start:state_end]
                    agent_actions = agent.actor(agent_states)
                    current_actions.append(agent_actions)
                else:
                    # Use other agents' current actions (detached)
                    action_start = sum(self.agent_configs[aid2]['action_dim'] 
                                     for aid2 in list(self.agents.keys())[:list(self.agents.keys()).index(aid)])
                    action_end = action_start + self.agent_configs[aid]['action_dim']
                    other_actions = actions[:, action_start:action_end].detach()
                    current_actions.append(other_actions)
            
            actor_loss = agent.update_actor(states, current_actions)
            actor_losses[agent_id] = actor_loss
        
        # Soft update target networks
        for agent in self.agents.values():
            agent.soft_update()
        
        # Decay noise
        if self.training_step % 100 == 0:
            for agent in self.agents.values():
                agent.decay_noise()
        
        self.training_step += 1
        
        # Update metrics
        for agent_id, loss in actor_losses.items():
            self.training_metrics['actor_losses'][agent_id].append(loss)
        for agent_id, loss in critic_losses.items():
            self.training_metrics['critic_losses'][agent_id].append(loss)
        
        return {**{f'actor_loss_{aid}': loss for aid, loss in actor_losses.items()},
                **{f'critic_loss_{aid}': loss for aid, loss in critic_losses.items()}}
    
    def reset_episode(self):
        """Reset for new episode"""
        for agent in self.agents.values():
            agent.reset_noise()
        
        self.episode_count += 1
    
    def save_models(self, directory: str):
        """Save all agent models"""
        import os
        os.makedirs(directory, exist_ok=True)
        
        for agent_id, agent in self.agents.items():
            filepath = os.path.join(directory, f'maddpg_{agent_id}.pth')
            agent.save_model(filepath)
        
        # Save MADDPG state
        maddpg_state = {
            'training_step': self.training_step,
            'episode_count': self.episode_count,
            'agent_configs': self.agent_configs,
            'config': self.config
        }
        
        import pickle
        with open(os.path.join(directory, 'maddpg_state.pkl'), 'wb') as f:
            pickle.dump(maddpg_state, f)
        
        logger.info(f"Saved MADDPG models to {directory}")
    
    def load_models(self, directory: str):
        """Load all agent models"""
        import os
        import pickle
        
        # Load MADDPG state
        with open(os.path.join(directory, 'maddpg_state.pkl'), 'rb') as f:
            maddpg_state = pickle.load(f)
        
        self.training_step = maddpg_state['training_step']
        self.episode_count = maddpg_state['episode_count']
        
        # Load agent models
        for agent_id, agent in self.agents.items():
            filepath = os.path.join(directory, f'maddpg_{agent_id}.pth')
            if os.path.exists(filepath):
                agent.load_model(filepath)
            else:
                logger.warning(f"Model file not found for agent {agent_id}: {filepath}")
        
        logger.info(f"Loaded MADDPG models from {directory}")
    
    def get_training_metrics(self) -> Dict[str, Any]:
        """Get training metrics"""
        metrics = self.training_metrics.copy()
        
        # Add summary statistics
        for agent_id in self.agents.keys():
            if self.training_metrics['actor_losses'][agent_id]:
                metrics[f'avg_actor_loss_{agent_id}'] = np.mean(
                    self.training_metrics['actor_losses'][agent_id][-100:]
                )
            if self.training_metrics['critic_losses'][agent_id]:
                metrics[f'avg_critic_loss_{agent_id}'] = np.mean(
                    self.training_metrics['critic_losses'][agent_id][-100:]
                )
        
        metrics['training_step'] = self.training_step
        metrics['episode_count'] = self.episode_count
        metrics['buffer_size'] = len(self.replay_buffer)
        
        return metrics
    
    def set_training_mode(self, training: bool = True):
        """Set training mode for all agents"""
        for agent in self.agents.values():
            if training:
                agent.actor.train()
                agent.critic.train()
            else:
                agent.actor.eval()
                agent.critic.eval()