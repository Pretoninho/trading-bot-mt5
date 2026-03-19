"""
PPO+GAE Agent with PyTorch - Production Ready

Implements:
- Generalized Advantage Estimation (GAE) with λ parameter
- Proximal Policy Optimization (PPO) with importance sampling clipping
- Trajectory buffer for batch processing
- Tensorboard logging for monitoring
- GPU-accelerated batched operations
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, List, Optional
from collections import deque
from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime

from .networks import PPONetworks


class ReplayBuffer:
    """
    Circular buffer for storing trajectories.
    
    Stores transitions and provides efficient batching for PPO updates.
    """
    
    def __init__(self, max_size: int = 2048):
        """
        Initialize replay buffer.
        
        Args:
            max_size: Maximum number of transitions to store
        """
        self.max_size = max_size
        self.buffer = {
            'observations': np.zeros((max_size, 391)),
            'actions': np.zeros(max_size, dtype=np.int64),
            'rewards': np.zeros(max_size),
            'values': np.zeros(max_size),
            'log_probs': np.zeros(max_size),
            'dones': np.zeros(max_size, dtype=bool),
            'advantages': np.zeros(max_size),
            'returns': np.zeros(max_size),
        }
        self.idx = 0
        self.size = 0
    
    def add(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        value: float,
        log_prob: float,
        done: bool,
    ) -> None:
        """Add single transition to buffer."""
        self.buffer['observations'][self.idx] = observation
        self.buffer['actions'][self.idx] = action
        self.buffer['rewards'][self.idx] = reward
        self.buffer['values'][self.idx] = value
        self.buffer['log_probs'][self.idx] = log_prob
        self.buffer['dones'][self.idx] = done
        
        self.idx = (self.idx + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
    
    def compute_advantages(
        self,
        gamma: float = 1.0,
        lambda_: float = 0.95,
    ) -> None:
        """
        Compute GAE advantages in-place.
        
        Args:
            gamma: Discount factor
            lambda_: GAE smoothing parameter
        """
        buffer = self.buffer
        advantages = np.zeros(self.size)
        
        # Compute TD errors (deltas)
        deltas = np.zeros(self.size)
        for t in range(self.size):
            if buffer['dones'][t]:
                deltas[t] = buffer['rewards'][t] - buffer['values'][t]
            else:
                next_value = buffer['values'][t + 1] if t + 1 < self.size else 0
                deltas[t] = buffer['rewards'][t] + gamma * next_value - buffer['values'][t]
        
        # Backward GAE accumulation
        gae_coeff = gamma * lambda_
        advantage = 0.0
        for t in reversed(range(self.size)):
            if buffer['dones'][t]:
                advantage = deltas[t]
            else:
                advantage = deltas[t] + gae_coeff * advantage
            advantages[t] = advantage
        
        # Normalize advantages
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
        
        buffer['advantages'][:self.size] = advantages
        buffer['returns'][:self.size] = advantages + buffer['values'][:self.size]
    
    def get_batch(
        self,
        batch_size: int,
        device: str = 'cpu',
    ) -> Dict[str, torch.Tensor]:
        """
        Get random batch from buffer.
        
        Returns tensors on specified device.
        """
        indices = np.random.choice(self.size, size=min(batch_size, self.size), replace=False)
        
        batch = {}
        for key in ['observations', 'actions', 'returns', 'advantages', 'log_probs']:
            data = self.buffer[key][indices]
            batch[key] = torch.FloatTensor(data).to(device)
        
        # Actions should be long tensor
        batch['actions'] = batch['actions'].long()
        
        return batch
    
    def reset(self) -> None:
        """Clear buffer."""
        self.idx = 0
        self.size = 0


class PPOAgentPyTorch:
    """
    Production-ready PPO+GAE Agent using PyTorch.
    
    Features:
    - Batched forward passes (GPU-accelerated)
    - Automatic differentiation for gradients
    - GAE advantage estimation
    - PPO-clip objective with clipping statistics
    - Tensorboard monitoring
    - Checkpoint saving/loading
    """
    
    def __init__(
        self,
        observation_dim: int = 391,
        action_dim: int = 6,
        hidden_dim: int = 128,
        learning_rate: float = 3e-4,
        gamma: float = 1.0,
        lambda_: float = 0.95,
        clip_ratio: float = 0.2,
        entropy_coef: float = 0.01,
        critic_coef: float = 1.0,
        max_grad_norm: float = 0.5,
        epochs: int = 5,
        batch_size: int = 256,
        device: Optional[str] = None,
        log_dir: Optional[str] = None,
    ):
        """
        Initialize PPO+GAE agent.
        
        Args:
            observation_dim: State dimension
            action_dim: Action dimension
            hidden_dim: Neural network hidden size
            learning_rate: Optimizer learning rate
            gamma: Discount factor
            lambda_: GAE smoothing (0=TD, 1=MC)
            clip_ratio: PPO clipping epsilon
            entropy_coef: Entropy bonus coefficient
            critic_coef: Critic loss weight
            max_grad_norm: Gradient clipping threshold
            epochs: Number of epochs per update
            batch_size: Batch size for training
            device: 'cuda' or 'cpu'
            log_dir: Directory for Tensorboard logs
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Networks
        self.networks = PPONetworks(
            observation_dim=observation_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            learning_rate=learning_rate,
            device=self.device,
        )
        
        # Hyperparameters
        self.gamma = gamma
        self.lambda_ = lambda_
        self.clip_ratio = clip_ratio
        self.entropy_coef = entropy_coef
        self.critic_coef = critic_coef
        self.max_grad_norm = max_grad_norm
        self.epochs = epochs
        self.batch_size = batch_size
        
        # Trajectory buffer
        self.buffer = ReplayBuffer(max_size=2048)
        
        # Logging
        self.log_dir = log_dir or f"./logs/ppo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.log_dir, exist_ok=True)
        self.writer = SummaryWriter(self.log_dir)
        self.global_step = 0
        self.episode_count = 0
        
        # Statistics
        self.episode_rewards = []
        self.episode_return = 0
        self.episode_length = 0
    
    def select_action(
        self,
        observation: np.ndarray,
        deterministic: bool = False,
    ) -> Tuple[int, float, float]:
        """
        Select action using current policy.
        
        Args:
            observation: State vector [obs_dim]
            deterministic: If True, use argmax (evaluation mode)
            
        Returns:
            action: Selected action
            log_prob: Log probability log π(a|s)
            value: Value estimate V(s)
        """
        obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            policy, value = self.networks.get_policy_and_value(obs_tensor)
        
        if deterministic:
            action = policy.logits.argmax(dim=1).item()
            log_prob = torch.log_softmax(policy.logits, dim=1)[0, action].item()
        else:
            action = policy.sample()[0].item()
            log_prob = policy.log_prob(torch.tensor([action]).to(self.device)).item()
        
        value = value.squeeze().item()
        
        return action, log_prob, value
    
    def store_transition(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        value: float,
        log_prob: float,
        done: bool,
    ) -> None:
        """Store trajectory transition."""
        self.buffer.add(observation, action, reward, value, log_prob, done)
        
        self.episode_return += reward
        self.episode_length += 1
        self.global_step += 1
        
        if done:
            self.episode_rewards.append(self.episode_return)
            self.episode_count += 1
            
            # Log episode statistics
            self.writer.add_scalar('episode/return', self.episode_return, self.episode_count)
            self.writer.add_scalar('episode/length', self.episode_length, self.episode_count)
            
            self.episode_return = 0
            self.episode_length = 0
    
    def update(self) -> Dict[str, float]:
        """
        Update policy and value networks using PPO+GAE.
        
        Implements:
        1. GAE advantage computation
        2. Advantage normalization
        3. PPO-clip optimization
        4. Multiple epochs over buffer
        5. Gradient clipping
        """
        if self.buffer.size == 0:
            return {'status': 'no_data'}
        
        # Compute GAE advantages
        self.buffer.compute_advantages(gamma=self.gamma, lambda_=self.lambda_)
        
        # Statistics for logging
        metrics = {
            'actor_loss': 0.0,
            'critic_loss': 0.0,
            'entropy': 0.0,
            'policy_loss': 0.0,
            'clipped_ratio': 0.0,
            'value_loss': 0.0,
        }
        
        num_updates = 0
        num_clipped = 0
        
        # Multiple epochs
        for epoch in range(self.epochs):
            # Shuffle and batch
            indices = np.random.permutation(self.buffer.size)
            
            for start_idx in range(0, self.buffer.size, self.batch_size):
                end_idx = min(start_idx + self.batch_size, self.buffer.size)
                batch_indices = indices[start_idx:end_idx]
                
                # Extract batch
                obs = torch.FloatTensor(self.buffer.buffer['observations'][batch_indices]).to(self.device)
                actions = torch.LongTensor(self.buffer.buffer['actions'][batch_indices]).to(self.device)
                returns = torch.FloatTensor(self.buffer.buffer['returns'][batch_indices]).to(self.device)
                advantages = torch.FloatTensor(self.buffer.buffer['advantages'][batch_indices]).to(self.device)
                old_log_probs = torch.FloatTensor(self.buffer.buffer['log_probs'][batch_indices]).to(self.device)
                
                # Forward pass
                action_logits, values = self.networks.forward_pass(obs)
                values = values.squeeze()
                
                # Policy distribution
                policy = torch.distributions.Categorical(logits=action_logits)
                new_log_probs = policy.log_prob(actions)
                entropy = policy.entropy().mean()
                
                # Importance sampling ratio
                ratio = torch.exp(new_log_probs - old_log_probs)
                
                # PPO-clip objective
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Count clipped
                num_clipped += (torch.abs(ratio - 1) > self.clip_ratio).float().sum().item()
                
                # Critic loss (MSE)
                critic_loss = F.mse_loss(values, returns)
                
                # Total loss
                total_loss = policy_loss + self.critic_coef * critic_loss - self.entropy_coef * entropy
                
                # Backward pass
                self.networks.optimizer.zero_grad()
                total_loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.networks.network.parameters(),
                    self.max_grad_norm
                )
                
                # Update
                self.networks.optimizer.step()
                
                # Accumulate metrics
                metrics['actor_loss'] += policy_loss.item()
                metrics['critic_loss'] += critic_loss.item()
                metrics['entropy'] += entropy.item()
                metrics['value_loss'] += critic_loss.item()
                
                num_updates += 1
        
        # Average metrics
        for key in ['actor_loss', 'critic_loss', 'entropy', 'value_loss']:
            metrics[key] /= num_updates
        
        metrics['clipped_ratio'] = num_clipped / (num_updates * self.batch_size)
        
        # Write to Tensorboard
        update_num = self.episode_count // 10  # Update every 10 episodes
        self.writer.add_scalar('training/actor_loss', metrics['actor_loss'], update_num)
        self.writer.add_scalar('training/critic_loss', metrics['critic_loss'], update_num)
        self.writer.add_scalar('training/entropy', metrics['entropy'], update_num)
        self.writer.add_scalar('training/clipped_ratio', metrics['clipped_ratio'], update_num)
        
        # Clear buffer
        self.buffer.reset()
        
        return metrics
    
    def save_checkpoint(self, filepath: str) -> None:
        """Save agent checkpoint."""
        self.networks.save_checkpoint(filepath)
    
    def load_checkpoint(self, filepath: str) -> None:
        """Load agent checkpoint."""
        self.networks.load_checkpoint(filepath)
    
    def get_device_info(self) -> str:
        """Get device information."""
        info = self.networks.get_device_info()
        return str(info)
