"""
trading_env.agents
==================
Reinforcement learning agents for the trading environment.
"""

from .actor_critic import ActorCriticAgent
from .ppo import PPOAgent
from .reinforce import SimpleReinforce, ReinforceWithBaseline
from .gae import GAEAdvantageEstimator, PPOAgentWithGAE

# PyTorch Production-Ready
from .networks import ActorCriticNetwork, PPONetworks
from .ppo_pytorch import PPOAgentPyTorch, ReplayBuffer

__all__ = [
    # Numpy-based (research/baseline)
    "ActorCriticAgent",
    "PPOAgent",
    "SimpleReinforce",
    "ReinforceWithBaseline",
    "GAEAdvantageEstimator",
    "PPOAgentWithGAE",
    # PyTorch Production
    "ActorCriticNetwork",
    "PPONetworks",
    "PPOAgentPyTorch",
    "ReplayBuffer",
]
