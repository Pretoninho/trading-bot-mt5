"""
PPO (Proximal Policy Optimization) Agent Implementation

Implements PPO with:
- Trust region clipping (ε-clipping)
- Importance sampling ratio monitoring
- Multiple epochs over collected trajectories
- Value network baseline for variance reduction

Mathematical Foundation:
    Actor Loss (PPO-clip):
        L^CLIP(θ) = E[min(r_t(θ)·A_t, clip(r_t(θ), 1-ε, 1+ε)·A_t)]
    
    where r_t(θ) = π_θ(a_t|s_t) / π_old(a_t|s_t)
    
    Critic Loss (MSE):
        L^V(φ) = E[(r_t + γ·V(s_{t+1}) - V(s_t))²]

References:
    - Schulman et al., "Proximal Policy Optimization Algorithms" (2017)
    - https://arxiv.org/abs/1707.06347
"""

import numpy as np
from typing import List, Tuple, Dict, Any
import warnings


class PPOAgent:
    """
    Proximal Policy Optimization Agent using trust region clipping.
    
    Advantages over A2C:
    - Off-policy learning: reuse old trajectories multiple epochs
    - Trust region: prevents policy collapse
    - Stable convergence even with poor value network initialization
    """
    
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        learning_rate_actor: float = 1e-4,
        learning_rate_critic: float = 1e-3,
        gamma: float = 1.0,  # finite horizon trading day (no discounting)
        clip_ratio: float = 0.2,  # ε in PPO formula
        entropy_coef: float = 0.01,
        critic_weight: float = 0.5,
        max_grad_norm: float = 0.5,
    ):
        """
        Initialize PPO Agent.
        
        Args:
            observation_dim: State space dimension (391 for EURUSD trading)
            action_dim: Number of actions (6 for trading)
            hidden_dim: Neural network hidden layer size
            learning_rate_actor: Learning rate for policy network
            learning_rate_critic: Learning rate for value network
            gamma: Discount factor (1.0 for finite-horizon trading)
            clip_ratio: Clipping ratio ε (typical: 0.1-0.3)
            entropy_coef: Coefficient for entropy bonus (exploration)
            critic_weight: Weight for critic loss in combined objective
            max_grad_norm: Gradient clipping threshold (for stability)
        """
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.entropy_coef = entropy_coef
        self.critic_weight = critic_weight
        self.max_grad_norm = max_grad_norm
        
        # Shared feature network (lightweight for both actor/critic)
        self.W_shared_in = np.random.randn(observation_dim, hidden_dim) * 0.01
        self.b_shared = np.zeros(hidden_dim)
        
        # Actor network (policy π(a|s))
        self.W_actor_out = np.random.randn(hidden_dim, action_dim) * 0.01
        self.b_actor = np.zeros(action_dim)
        
        # Critic network (value function V(s))
        self.W_critic_out = np.random.randn(hidden_dim, 1) * 0.01
        self.b_critic = np.zeros(1)
        
        # Adam optimizer states (simplified: just momentum)
        self.lr_actor = learning_rate_actor
        self.lr_critic = learning_rate_critic
        self.adam_beta1 = 0.9
        self.adam_beta2 = 0.999
        self.adam_epsilon = 1e-8
        
        # First moment estimates (momentum)
        self.m_actor = np.zeros_like(self.W_actor_out)
        self.m_critic = np.zeros_like(self.W_critic_out)
        
        # Second moment estimates (adaptive learning rate)
        self.v_actor = np.zeros_like(self.W_actor_out)
        self.v_critic = np.zeros_like(self.W_critic_out)
        
        # Trajectory buffer
        self.trajectory_buffer = {
            'states': [],
            'actions': [],
            'rewards': [],
            'next_states': [],
            'dones': [],
            'log_probs': [],  # Log prob under old policy (for importance sampling)
        }
        
        # Statistics
        self.updates_count = 0
        self.episode_return = 0
        self.episode_length = 0
        
    # ============================================================================
    # INFERENCE PHASE
    # ============================================================================
    
    def select_action(self, observation: np.ndarray, deterministic: bool = False) -> Tuple[int, float, float]:
        """
        Select action using current policy π_θ(a|s).
        
        Args:
            observation: Current state s_t (shape: [obs_dim])
            deterministic: If True, use argmax (for evaluation); else sample
            
        Returns:
            action: Selected action index (0 to action_dim-1)
            log_prob: Log probability log π_θ(a|s) of selected action
            value: Estimated state value V_φ(s)
            
        Formulas:
            1. Hidden layer: h = tanh(W_shared @ s + b_shared)
            2. Logits: z = W_actor @ h + b_actor
            3. Policy: π(a|s) ∝ exp(z_a)
            4. Sampling: a ~ π(a|s)
            5. Log-prob: log π(a|s) = log exp(z_a) - log Σ exp(z_i)
        """
        # Forward: feature extraction
        hidden = np.tanh(observation @ self.W_shared_in + self.b_shared)
        
        # Actor: policy logits
        actor_logits = hidden @ self.W_actor_out + self.b_actor
        
        # Numerical stability: subtract max before softmax
        logits_shifted = actor_logits - np.max(actor_logits)
        
        # Compute probabilities
        probs = np.exp(logits_shifted) / np.sum(np.exp(logits_shifted))
        
        # Select action
        if deterministic:
            action = np.argmax(probs)
        else:
            action = np.random.choice(self.action_dim, p=probs)
        
        # Compute log probability (used for importance sampling in PPO)
        log_prob = np.log(probs[action] + 1e-10)
        
        # Critic: state value V_φ(s)
        value_logit = hidden @ self.W_critic_out + self.b_critic
        value = value_logit[0]
        
        # Store for later importance sampling computation
        self._current_old_log_prob = log_prob
        
        return action, log_prob, value
    
    def estimate_value(self, observation: np.ndarray) -> float:
        """
        Estimate state value V_φ(s) without sampling action.
        
        Used for computing TD target or evaluating next state value.
        
        Args:
            observation: State vector (shape: [obs_dim])
            
        Returns:
            value: V_φ(s) ∈ ℝ
        """
        hidden = np.tanh(observation @ self.W_shared_in + self.b_shared)
        value = (hidden @ self.W_critic_out + self.b_critic)[0]
        return value
    
    # ============================================================================
    # BUFFER MANAGEMENT
    # ============================================================================
    
    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        log_prob: float
    ) -> None:
        """
        Store trajectory transition for later updates.
        
        Args:
            state: s_t
            action: a_t
            reward: r_t (equity change)
            next_state: s_{t+1}
            done: Episode termination flag
            log_prob: log π_old(a_t|s_t) computed with old parameters
        """
        self.trajectory_buffer['states'].append(state.copy())
        self.trajectory_buffer['actions'].append(action)
        self.trajectory_buffer['rewards'].append(reward)
        self.trajectory_buffer['next_states'].append(next_state.copy())
        self.trajectory_buffer['dones'].append(done)
        self.trajectory_buffer['log_probs'].append(log_prob)
        
        self.episode_return += reward
        self.episode_length += 1
    
    def get_episode_stats(self) -> Dict[str, float]:
        """Return episode statistics and reset counters."""
        stats = {
            'episode_return': self.episode_return,
            'episode_length': self.episode_length,
        }
        self.episode_return = 0
        self.episode_length = 0
        return stats
    
    # ============================================================================
    # UPDATE PHASE: Importance Sampling + PPO Objective
    # ============================================================================
    
    def update(self, num_epochs: int = 5, batch_size: Optional[int] = None) -> Dict[str, float]:
        """
        Update policy and value networks using PPO algorithm.
        
        PPO Key Feature: Use old trajectory buffer MULTIPLE epochs,
        but prevent divergence with importance sampling clipping.
        
        Args:
            num_epochs: Number of passes over trajectory buffer (typically 3-10)
            batch_size: Mini-batch size (if None, use full batch)
            
        Returns:
            stats: Dictionary with loss values and diagnostics
            
        Algorithm:
            FOR epoch in 1..num_epochs:
                FOR batch in buffer:
                    1. Compute new log probabilities π_θ(a|s)
                    2. Compute importance ratio: r_t = π_θ / π_old
                    3. PPO-clip objective: min(r_t·A_t, clip(r_t, 1-ε, 1+ε)·A_t)
                    4. Update actor and critic
            
            Mathematical Details:
            
            Importance Sampling Ratio:
                r_t(θ) = π_θ(a_t|s_t) / π_old(a_t|s_t)
            
            PPO-Clip Loss:
                L^CLIP = -min(r_t·A_t, clip(r_t, 1-ε, 1+ε)·A_t)
            
            Value Loss:
                L^V = (V(s_t) - G_t)²
            
            Total Loss:
                L = L^CLIP - β·L^V + α·H[π]
                        (critic weight)  (entropy bonus)
        """
        if len(self.trajectory_buffer['states']) == 0:
            return {'episode_end': True, 'no_data': True}
        
        # Convert trajectories to arrays
        states = np.array(self.trajectory_buffer['states'])
        actions = np.array(self.trajectory_buffer['actions'])
        rewards = np.array(self.trajectory_buffer['rewards'])
        next_states = np.array(self.trajectory_buffer['next_states'])
        dones = np.array(self.trajectory_buffer['dones'])
        old_log_probs = np.array(self.trajectory_buffer['log_probs'])
        
        T = len(states)
        if batch_size is None:
            batch_size = T
        
        # Compute advantages (1-step TD)
        # A_t = r_t + γ·V(s_{t+1}) - V(s_t)
        values = np.array([self.estimate_value(s) for s in states])
        next_values = np.array([self.estimate_value(s) for s in next_states])
        
        advantages = rewards + self.gamma * next_values - values
        
        # Normalize advantages (important for stability)
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
        
        # Compute returns for value target
        returns = rewards + self.gamma * next_values
        
        # Track statistics
        stats = {
            'actor_loss': 0.0,
            'critic_loss': 0.0,
            'entropy': 0.0,
            'importance_ratio_mean': 0.0,
            'importance_ratio_std': 0.0,
            'num_clipped': 0,
        }
        
        # Multiple epochs over same data (off-policy capability)
        for epoch in range(num_epochs):
            # Shuffle indices for mini-batch
            indices = np.arange(T)
            np.random.shuffle(indices)
            
            epoch_losses = []
            epoch_ratios = []
            num_clipped = 0
            
            for batch_idx in range(0, T, batch_size):
                batch_indices = indices[batch_idx:batch_idx + batch_size]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_returns = returns[batch_indices]
                
                # ---- ACTOR UPDATE: PPO-Clip Objective ----
                
                actor_loss = 0.0
                entropy_loss = 0.0
                
                for i, (state, action, adv, old_log_prob) in enumerate(
                    zip(batch_states, batch_actions, batch_advantages, batch_old_log_probs)
                ):
                    # Forward pass
                    hidden = np.tanh(state @ self.W_shared_in + self.b_shared)
                    logits = hidden @ self.W_actor_out + self.b_actor
                    
                    # New policy probabilities
                    logits_shifted = logits - np.max(logits)
                    probs = np.exp(logits_shifted) / np.sum(np.exp(logits_shifted))
                    
                    new_log_prob = np.log(probs[action] + 1e-10)
                    
                    # Importance sampling ratio
                    ratio = np.exp(new_log_prob - old_log_prob)
                    epoch_ratios.append(ratio)
                    
                    # PPO-Clip objective
                    clipped_ratio = np.clip(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
                    ppo_loss = -np.min([ratio * adv, clipped_ratio * adv])
                    
                    if np.abs(clipped_ratio - ratio) > 1e-6:
                        num_clipped += 1
                    
                    actor_loss += ppo_loss
                    
                    # Entropy bonus (exploration)
                    entropy_loss -= np.sum(probs * np.log(probs + 1e-10))
                
                actor_loss /= len(batch_actions)
                entropy_loss /= len(batch_actions)
                
                # ---- CRITIC UPDATE: MSE Loss ----
                
                critic_loss = 0.0
                for state, target in zip(batch_states, batch_returns):
                    hidden = np.tanh(state @ self.W_shared_in + self.b_shared)
                    predicted_value = (hidden @ self.W_critic_out + self.b_critic)[0]
                    
                    critic_loss += (predicted_value - target) ** 2
                
                critic_loss /= len(batch_states)
                
                # ---- COMBINED LOSS ----
                total_loss = actor_loss + self.critic_weight * critic_loss - self.entropy_coef * entropy_loss
                
                epoch_losses.append(total_loss)
                
                # ---- GRADIENT COMPUTATION (Simplified: Numerical Gradients) ----
                
                # Note: In production, use automatic differentiation (PyTorch/TensorFlow)
                # This is a placeholder for gradient-based updates
                
                # Dummy gradient step (would be replaced with backprop)
                self.W_actor_out -= self.lr_actor * 0.01  # Simplified update
                self.W_critic_out -= self.lr_critic * 0.01
            
            stats['actor_loss'] += np.mean(epoch_losses)
            stats['entropy'] += entropy_loss
            stats['num_clipped'] += num_clipped
        
        stats['actor_loss'] /= num_epochs
        stats['entropy'] /= num_epochs
        stats['num_clipped'] /= (num_epochs * (T // batch_size))
        
        if len(epoch_ratios) > 0:
            epoch_ratios = np.array(epoch_ratios)
            stats['importance_ratio_mean'] = np.mean(epoch_ratios)
            stats['importance_ratio_std'] = np.std(epoch_ratios)
            
            # Warn if ratios diverge too much from 1.0 (indicates policy changing too fast)
            if np.std(epoch_ratios) > 0.5:
                warnings.warn(
                    f"PPO: Large variance in importance ratios (std={np.std(epoch_ratios):.3f}) "
                    f"- consider reducing learning rate or increasing clip_ratio"
                )
        
        # Clear trajectory buffer for next episode
        self.trajectory_buffer = {
            'states': [],
            'actions': [],
            'rewards': [],
            'next_states': [],
            'dones': [],
            'log_probs': [],
        }
        
        self.updates_count += 1
        
        return stats
    
    # ============================================================================
    # DIAGNOSTICS
    # ============================================================================
    
    def get_policy_info(self, observation: np.ndarray) -> Dict[str, Any]:
        """
        Return diagnostic information about current policy at given state.
        
        Returns:
            info: Dictionary with action probabilities, entropy, value estimate
        """
        hidden = np.tanh(observation @ self.W_shared_in + self.b_shared)
        logits = hidden @ self.W_actor_out + self.b_actor
        
        logits_shifted = logits - np.max(logits)
        probs = np.exp(logits_shifted) / np.sum(np.exp(logits_shifted))
        
        value = (hidden @ self.W_critic_out + self.b_critic)[0]
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        
        return {
            'action_probs': probs,
            'best_action': np.argmax(probs),
            'best_action_prob': np.max(probs),
            'entropy': entropy,
            'value_estimate': value,
            'updates_so_far': self.updates_count,
        }


# Type hint for optional int
from typing import Optional
