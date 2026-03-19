"""
PyTorch Neural Network Architecture for PPO+GAE Trading Agent

Production-ready networks with:
- Shared feature backbone
- Actor (policy) and Critic (value) heads
- Batch processing
- GPU support
- Proper weight initialization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class ActorCriticNetwork(nn.Module):
    """
    Shared Actor-Critic Architecture for PPO+GAE.
    
    Architecture:
        Input [batch, 391] → Shared Backbone → [actor, critic]
        
                       ┌─ Actor Head → π(a|s) [batch, 6]
        Backbone [128] ┤
                       └─ Critic Head → V(s) [batch, 1]
    
    Features:
    - Shared backbone (128D hidden layer)
    - Separate heads for actor and critic
    - Batch processing support
    - GPU-compatible
    """
    
    def __init__(
        self,
        observation_dim: int = 391,
        action_dim: int = 6,
        hidden_dim: int = 128,
        activation: nn.Module = nn.ReLU,
    ):
        """
        Initialize Actor-Critic network.
        
        Args:
            observation_dim: State space dimension (391 for EURUSD)
            action_dim: Number of actions (6 for trading)
            hidden_dim: Hidden layer dimension
            activation: Activation function (default: ReLU)
        """
        super().__init__()
        
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        # ========== SHARED BACKBONE ==========
        # Maps observation → hidden representation
        self.shared = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            activation(),
            nn.Linear(hidden_dim, hidden_dim),
            activation(),
        )
        
        # ========== ACTOR HEAD ==========
        # Outputs logits for action distribution π(a|s)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            activation(),
            nn.Linear(hidden_dim, action_dim),
        )
        
        # ========== CRITIC HEAD ==========
        # Outputs value estimate V(s)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            activation(),
            nn.Linear(hidden_dim, 1),
        )
        
        # Weight initialization (important for stability)
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.0)
                nn.init.constant_(module.bias, 0)
    
    def forward(
        self,
        observations: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass: compute policy and value.
        
        Args:
            observations: Batch of states [batch_size, obs_dim]
            
        Returns:
            action_logits: [batch_size, action_dim] - raw logits
            values: [batch_size, 1] - value estimates
        """
        # Shared backbone
        hidden = self.shared(observations)
        
        # Actor head: policy logits
        action_logits = self.actor(hidden)
        
        # Critic head: value estimate
        values = self.critic(hidden)
        
        return action_logits, values
    
    def get_policy_and_value(
        self,
        observations: torch.Tensor,
    ) -> Tuple[torch.distributions.Categorical, torch.Tensor]:
        """
        Get policy distribution and value estimate.
        
        Used during inference (sampling actions).
        
        Args:
            observations: Batch of states [batch_size, obs_dim]
            
        Returns:
            policy: torch.distributions.Categorical
            values: [batch_size, 1]
        """
        action_logits, values = self.forward(observations)
        
        # Create categorical distribution from logits
        policy = torch.distributions.Categorical(logits=action_logits)
        
        return policy, values


class PPONetworks:
    """
    Wrapper managing both shared network and optimization.
    
    Handles:
    - Network initialization
    - Device management (CPU/GPU)
    - Parameter optimization
    - Learning rate scheduling
    """
    
    def __init__(
        self,
        observation_dim: int = 391,
        action_dim: int = 6,
        hidden_dim: int = 128,
        learning_rate: float = 3e-4,
        device: Optional[str] = None,
    ):
        """
        Initialize PPO networks.
        
        Args:
            observation_dim: State dimension
            action_dim: Action dimension
            hidden_dim: Hidden layer size
            learning_rate: Optimizer learning rate
            device: 'cuda' or 'cpu' (auto-detect if None)
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create network on device
        self.network = ActorCriticNetwork(
            observation_dim=observation_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
        ).to(self.device)
        
        # Optimizer: uses all network parameters
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=learning_rate,
        )
        
        self.learning_rate = learning_rate
    
    def get_policy_and_value(
        self,
        observations: torch.Tensor,
    ) -> Tuple[torch.distributions.Categorical, torch.Tensor]:
        """Get policy and value for batch of observations."""
        if not isinstance(observations, torch.Tensor):
            observations = torch.FloatTensor(observations).to(self.device)
        
        with torch.no_grad():
            policy, values = self.network.get_policy_and_value(observations)
        
        return policy, values
    
    def forward_pass(
        self,
        observations: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning logits and values."""
        return self.network(observations)
    
    def save_checkpoint(self, filepath: str) -> None:
        """Save network weights to file."""
        torch.save({
            'network_state': self.network.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
        }, filepath)
        print(f"✓ Checkpoint saved: {filepath}")
    
    def load_checkpoint(self, filepath: str) -> None:
        """Load network weights from file."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.network.load_state_dict(checkpoint['network_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        print(f"✓ Checkpoint loaded: {filepath}")
    
    def get_device_info(self) -> dict:
        """Return device information for logging."""
        if self.device == 'cuda':
            return {
                'device': 'CUDA',
                'gpu_name': torch.cuda.get_device_name(0),
                'gpu_memory_allocated': f"{torch.cuda.memory_allocated() / 1e9:.2f} GB",
            }
        else:
            return {'device': 'CPU'}
