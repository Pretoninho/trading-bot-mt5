"""
Actor-Critic A2C Agent
======================

Implementation of the Asynchronous Advantage Actor-Critic (A2C) algorithm
for the trading environment.

Policy-Gradient + Value-Based Hybrid Approach:
  - Actor: π(a|s) parameterized policy network → 6 action probabilities
  - Critic: V(s) value network → estimates expected return from state

Key Equations:
  - Advantage: A(s,a) = r + γV(s') - V(s)  [credit assignment]
  - Actor Loss: -log π(a|s) × A(s,a)       [policy gradient]
  - Critic Loss: A(s,a)²                   [value function]

Exploration-Exploitation:
  - Policy is stochastic (softmax over Q-values)
  - Temperature τ controls exploration (anneals during training)
  - No ε-greedy required; exploration intrinsic to softmax

References:
  - Mnih et al. (2016) "Asynchronous Methods for Deep Reinforcement Learning"
  - Sutton & Barto (2018) "Reinforcement Learning: An Introduction"
"""

import numpy as np
from typing import Tuple, Optional


class ActorCriticAgent:
    r"""
    Actor-Critic Agent for Gymnasium trading environment.

    Combines policy-gradient (actor) and value-based (critic) learning.

    Attributes
    ----------
    state_dim : int
        Observation space dimension (391 for our EURUSD trading env)
    action_dim : int
        Discrete action space size (6: HOLD, OPEN_LONG, OPEN_SHORT, CLOSE, PROTECT, MANAGE_TP)
    hidden_dim : int, default=256
        Size of shared hidden layer
    learning_rate : float, default=0.001
        Optimizer learning rate for both actor and critic
    gamma : float, default=1.0
        Discount factor (1.0 for finite-horizon MDPs; episode = 1 day)
    entropy_coeff : float, default=0.01
        Coefficient for entropy regularization (encourages exploration)
    temperature : float, default=1.0
        Initial temperature for softmax policy. Anneals during training.
    """

    def __init__(
        self,
        state_dim: int = 391,
        action_dim: int = 6,
        hidden_dim: int = 256,
        learning_rate: float = 0.001,
        gamma: float = 1.0,
        entropy_coeff: float = 0.01,
        temperature: float = 1.0,
    ):
        """
        Initialize Actor-Critic Agent.

        Parameters
        ----------
        state_dim : int
            Observation space dimension (default 391: 384D history + 7D context)
        action_dim : int
            Action space size (default 6 for trading actions)
        hidden_dim : int
            Shared hidden layer size (neurons)
        learning_rate : float
            Learning rate for gradient descent
        gamma : float
            Discount factor for future rewards
        entropy_coeff : float
            Weight for entropy regularization (prevents premature convergence)
        temperature : float
            Initial policy softmax temperature (exploration control)
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.entropy_coeff = entropy_coeff
        self.temperature = temperature

        # Initialize networks (simplified: using numpy, can be upgraded to PyTorch)
        # Shared layers: state representation
        self.W_shared_in = self._random_weights(state_dim, hidden_dim)
        self.b_shared = np.zeros(hidden_dim)

        # Actor head: outputs logits for policy
        self.W_actor_out = self._random_weights(hidden_dim, action_dim)
        self.b_actor = np.zeros(action_dim)

        # Critic head: outputs single value
        self.W_critic_out = self._random_weights(hidden_dim, 1)
        self.b_critic = np.zeros(1)

        # Training history
        self.episode_transitions = []  # [(s, a, r, s', done), ...]
        self.step_count = 0

    @staticmethod
    def _random_weights(in_dim: int, out_dim: int, scale: float = 0.01) -> np.ndarray:
        """Initialize network weights with Xavier scaling."""
        return np.random.randn(in_dim, out_dim) * scale / np.sqrt(in_dim)

    def select_action(
        self, observation: np.ndarray, deterministic: bool = False
    ) -> int:
        r"""
        Select action from current observation using actor (policy network).

        Parameters
        ----------
        observation : np.ndarray, shape (391,)
            Current state observation
        deterministic : bool, default=False
            If True: argmax action (exploitation only)
            If False: sample from π(a|s) with softmax (includes exploration)

        Returns
        -------
        int
            Selected action (0-5 for trading environment)

        Notes
        -----
        Uses softmax with temperature τ:

        .. math::
            \pi(a|s) = \frac{e^{Q(s,a)/\tau}}{\sum_{a'} e^{Q(s,a')/\tau}}

        High τ → more uniform (exploration)
        Low τ → peaked at argmax (exploitation)
        """
        # Forward pass: shared hidden layer
        hidden = self._forward_shared(observation)

        # Actor: compute logits
        logits = hidden @ self.W_actor_out + self.b_actor

        if deterministic:
            # Greedy: best action
            action = np.argmax(logits)
        else:
            # Stochastic: sample from softmax with temperature
            logits_scaled = logits / self.temperature
            logits_shifted = logits_scaled - np.max(logits_scaled)  # Numerical stability
            probs = np.exp(logits_shifted) / np.sum(np.exp(logits_shifted))
            action = np.random.choice(self.action_dim, p=probs)

        return action

    def estimate_value(self, observation: np.ndarray) -> float:
        r"""
        Estimate state value V(s) using critic network.

        Parameters
        ----------
        observation : np.ndarray, shape (391,)
            Current state observation

        Returns
        -------
        float
            Estimated value (expected cumulative return from this state)

        Notes
        -----
        Value function: $V(s) = \mathbb{E}[G_t | S_t = s]$
        where $G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + ...$
        """
        hidden = self._forward_shared(observation)
        value = (hidden @ self.W_critic_out + self.b_critic).item()
        return value

    def store_transition(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        next_observation: np.ndarray,
        done: bool,
    ) -> None:
        """
        Store transition in episode buffer for later training.

        Parameters
        ----------
        observation : np.ndarray
            State before action
        action : int
            Action taken (0-5)
        reward : float
            Reward received
        next_observation : np.ndarray
            State after action
        done : bool
            Episode termination flag
        """
        self.episode_transitions.append(
            (observation, action, reward, next_observation, done)
        )

    def update(self) -> dict:
        r"""
        Update actor and critic networks using collected episode transitions.

        Uses Temporal Difference (TD) learning with advantage function:

        .. math::
            A(s_t, a_t) = r_t + \gamma V(s_{t+1}) - V(s_t)

        Actor loss (policy gradient):
        .. math::
            \mathcal{L}_{actor} = -\log \pi(a_t|s_t) \cdot A(s_t, a_t)

        Critic loss (value function):
        .. math::
            \mathcal{L}_{critic} = A(s_t, a_t)^2

        Returns
        -------
        dict
            Training metrics: {
                'actor_loss': float,
                'critic_loss': float,
                'policy_entropy': float,
                'mean_return': float,
            }

        Notes
        -----
        - Uses one-step TD for efficiency (vs Monte Carlo waiting for episode end)
        - Entropy regularization encourages exploration
        - Advantage function reduces variance (credit assignment)
        """
        if not self.episode_transitions:
            return {
                "actor_loss": 0.0,
                "critic_loss": 0.0,
                "policy_entropy": 0.0,
                "mean_return": 0.0,
            }

        # Compute returns and advantages
        actor_losses = []
        critic_losses = []
        policy_entropies = []
        episode_returns = []

        for t, (s, a, r, s_next, done) in enumerate(self.episode_transitions):
            # Estimate values
            V_s = self.estimate_value(s)
            V_s_next = 0.0 if done else self.estimate_value(s_next)

            # Compute advantage (credit assignment)
            advantage = r + self.gamma * V_s_next - V_s
            episode_returns.append(advantage + V_s)

            # Actor update: policy gradient
            # Compute policy for this state
            hidden = self._forward_shared(s)
            logits = hidden @ self.W_actor_out + self.b_actor
            logits_scaled = logits / self.temperature
            logits_shifted = logits_scaled - np.max(logits_scaled)
            probs = np.exp(logits_shifted) / np.sum(np.exp(logits_shifted))

            # Log-prob of taken action
            log_prob_a = np.log(probs[a] + 1e-8)

            # Policy gradient loss (negative because we maximize)
            actor_loss = -log_prob_a * advantage

            # Entropy for regularization
            entropy = -np.sum(probs * np.log(probs + 1e-8))

            # Total loss with entropy bonus
            total_actor_loss = actor_loss - self.entropy_coeff * entropy

            # Critic update: value function
            critic_loss = advantage**2

            actor_losses.append(total_actor_loss)
            critic_losses.append(critic_loss)
            policy_entropies.append(entropy)

        # Average losses
        mean_actor_loss = np.mean(actor_losses)
        mean_critic_loss = np.mean(critic_losses)
        mean_entropy = np.mean(policy_entropies)
        mean_return = np.mean(episode_returns)

        # Update networks (simplified: gradient updates)
        # In production, use PyTorch/TensorFlow for auto-diff
        # Here: manual gradient descent placeholder
        self._gradient_step(mean_actor_loss, mean_critic_loss)

        # Clear episode buffer
        self.episode_transitions = []
        self.step_count += 1

        # Anneal temperature (exploration → exploitation)
        self.temperature = max(0.1, self.temperature * 0.995)

        return {
            "actor_loss": float(mean_actor_loss),
            "critic_loss": float(mean_critic_loss),
            "policy_entropy": float(mean_entropy),
            "mean_return": float(mean_return),
        }

    def _forward_shared(self, observation: np.ndarray) -> np.ndarray:
        """Compute shared hidden representation."""
        return np.tanh(observation @ self.W_shared_in + self.b_shared)

    def _gradient_step(self, actor_loss: float, critic_loss: float) -> None:
        """Simplified gradient update (placeholder for PyTorch equivalent)."""
        # In production implementation:
        # optimizer.zero_grad()
        # loss = actor_loss + critic_loss
        # loss.backward()
        # optimizer.step()
        pass

    def save(self, filepath: str) -> None:
        """Save agent weights to file."""
        pass  # Implement with deepcopy or pickle

    def load(self, filepath: str) -> None:
        """Load agent weights from file."""
        pass  # Implement with pickle

    def __repr__(self) -> str:
        return (
            f"ActorCriticAgent("
            f"state_dim={self.state_dim}, "
            f"action_dim={self.action_dim}, "
            f"hidden_dim={self.hidden_dim}, "
            f"lr={self.learning_rate}, "
            f"γ={self.gamma})"
        )
