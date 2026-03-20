"""
Example Training Loop for PPO+GAE PyTorch Agent

Demonstrates:
- Initialization with PyTorch backend
- Episode simulation with engineered features (456D state)
- Multi-component reward system integration
- Collecting trajectories
- Batch optimization
- Tensorboard monitoring
- Checkpoint saving

Usage:
    python examples/train_ppo_pytorch.py
"""

import numpy as np
import torch
from typing import Tuple, Optional
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trading_env.agents import PPOAgentPyTorch
from trading_env.utils.reward_functions import (
    RewardCalculator,
    reward_config_enhanced,
    RewardConfig,
)


def simulate_trading_env_step(
    state: np.ndarray,
    action: int,
    reward_calc: Optional[RewardCalculator] = None,
    prev_equity: float = 10000.0,
) -> Tuple[np.ndarray, float, bool, dict]:
    """
    Simulate one step of trading environment with engineered features.
    
    In production, this would interface with real MT5 or Gymnasium env.
    Uses multi-component reward system for richer training signals.
    
    Parameters
    ----------
    state : np.ndarray
        Current observation (456D: 391 OHLC + 65 engineered features)
    action : int
        Discrete action (0-5)
    reward_calc : RewardCalculator, optional
        Reward calculator with configured components
    prev_equity : float
        Previous equity level for reward calculation
    
    Returns
    -------
    tuple
        (next_state, reward, done, info)
    """
    # Generate next state with engineered features (456D)
    next_state = state + np.random.randn(456) * 0.01
    next_state = np.clip(next_state, -10, 10)  # Feature bounds
    
    # Simulate equity change
    equity_change = np.random.randn() * 100 + 50 * (action - 2.5)
    curr_equity = max(1000, prev_equity + equity_change)  # Prevent negative equity
    
    # Use multi-component reward if calculator provided
    if reward_calc is not None:
        # Track equity
        reward_calc.record_equity(curr_equity)
        
        # Simulate position state
        position_open = action in [1, 2]  # LONG or SHORT
        is_invalid = False  # Simplified
        
        # Calculate multi-component reward
        reward, components = reward_calc.calculate_reward(
            prev_equity=prev_equity,
            curr_equity=curr_equity,
            position_open=position_open,
            is_invalid_action=is_invalid,
        )
        
        if position_open:
            reward_calc.increment_bars_held()
    else:
        # Fallback to simple equity-based reward
        reward = (curr_equity - prev_equity) / max(prev_equity, 1.0)
        components = {"equity": reward}
    
    # Determine if episode terminates
    equity_return = (curr_equity - 10000) / 10000
    done = abs(equity_return) > 0.02  # ±2% termination
    
    info = {
        "equity": curr_equity,
        "equity_change": equity_change,
        "reward_components": components,
        "position_open": position_open if reward_calc else False,
    }
    
    return next_state, reward, done, info


def train_episode(
    agent: PPOAgentPyTorch,
    reward_config: Optional[RewardConfig] = None,
    max_steps: int = 100,
) -> dict:
    """
    Simulate one training episode with multi-component reward system.
    
    Args:
        agent: PPOAgentPyTorch instance
        reward_config: RewardConfig for multi-component rewards (optional)
        max_steps: Maximum steps per episode
        
    Returns:
        episode_stats: Dictionary with episode metrics
    """
    # Initialize reward calculator if config provided
    reward_calc = RewardCalculator(reward_config) if reward_config else None
    
    # Initialize state (456D: 391 base + 65 engineered features)
    state = np.random.randn(456) * 0.1
    state = np.clip(state, -10, 10)  # Feature bounds
    
    episode_reward = 0.0
    episode_length = 0
    component_sums = {}
    prev_equity = 10000.0
    
    for step in range(max_steps):
        # Select action
        action, log_prob, value = agent.select_action(state)
        
        # Simulate environment step with reward system
        next_state, reward, done, info = simulate_trading_env_step(
            state,
            action,
            reward_calc=reward_calc,
            prev_equity=prev_equity,
        )
        
        # Store transition
        agent.store_transition(
            observation=state,
            action=action,
            reward=reward,
            value=value,
            log_prob=log_prob,
            done=done,
        )
        
        episode_reward += reward
        episode_length += 1
        
        # Track reward components
        if "reward_components" in info:
            for comp_name, comp_val in info["reward_components"].items():
                component_sums[comp_name] = component_sums.get(comp_name, 0) + comp_val
        
        if done:
            break
        
        state = next_state
        prev_equity = info.get("equity", prev_equity)
    
    # Compute average component rewards
    avg_components = {
        k: v / max(episode_length, 1) for k, v in component_sums.items()
    }
    
    return {
        'reward': episode_reward,
        'length': episode_length,
        'avg_components': avg_components,
        'reward_calc': reward_calc,
    }


def main():
    """Main training loop with integrated reward system."""
    
    print("=" * 80)
    print("PPO+GAE PyTorch Agent - Training with Multi-Component Reward System")
    print("=" * 80)
    
    # ============================================================================
    # Initialize Reward System
    # ============================================================================
    
    print("\n📊 Reward System Configuration:")
    reward_config = reward_config_enhanced()
    print(f"  Config: enhanced")
    print(f"  Equity Scale:         {reward_config.equity_scale}")
    print(f"  Sharpe Ratio Bonus:   {reward_config.use_sharpe} (scale={reward_config.sharpe_scale})")
    print(f"  Streak Bonus:         {reward_config.use_streak} (scale={reward_config.streak_scale})")
    print(f"  Duration Reward:      {reward_config.use_duration} (scale={reward_config.duration_scale})")
    print(f"  Drawdown Penalty:     {reward_config.use_drawdown} (scale={reward_config.drawdown_scale})")
    print(f"  Opportunity Cost:     {reward_config.use_opportunity_cost} (scale={reward_config.opportunity_scale})")
    
    # ============================================================================
    # Initialize Agent
    # ============================================================================
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n🖥️  Device: {device}")
    
    agent = PPOAgentPyTorch(
        observation_dim=456,  # 391 base + 65 engineered features
        action_dim=6,
        hidden_dim=128,
        learning_rate=3e-4,
        gamma=1.0,
        lambda_=0.95,
        clip_ratio=0.2,
        entropy_coef=0.01,
        epochs=5,
        batch_size=128,
        device=device,
        log_dir="./logs/ppo_training_reward_optimized",
    )
    
    print(f"✓ Agent initialized")
    print(f"  Observation Dim: 456 (391 OHLC + 65 engineered features)")
    print(f"  Action Dim: 6")
    print(f"  Device Info: {agent.get_device_info()}")
    
    # ============================================================================
    # Training Loop
    # ============================================================================
    
    print("\n" + "=" * 80)
    print("Starting Training...")
    print("=" * 80)
    
    num_episodes = 10
    update_interval = 2  # Update every N episodes
    
    for episode in range(1, num_episodes + 1):
        # Run episode with reward system
        stats = train_episode(agent, reward_config=reward_config, max_steps=100)
        
        print(f"\n[Episode {episode:3d}]")
        print(f"  Total Reward: {stats['reward']:>8.4f}")
        print(f"  Length:       {stats['length']:>8d}")
        
        # Display average component rewards
        if stats.get('avg_components'):
            print(f"  Components:")
            for comp_name, comp_val in stats['avg_components'].items():
                print(f"    {comp_name:20s}: {comp_val:>8.5f}")
        
        # Update every N episodes
        if episode % update_interval == 0:
            print(f"\n  Updating networks...")
            update_stats = agent.update()
            
            print(f"    Actor Loss:     {update_stats.get('actor_loss', 0):.4f}")
            print(f"    Critic Loss:    {update_stats.get('critic_loss', 0):.4f}")
            print(f"    Entropy:        {update_stats.get('entropy', 0):.4f}")
            print(f"    Clipped Ratio:  {update_stats.get('clipped_ratio', 0):.4f}")
    
    # ============================================================================
    # Summary
    # ============================================================================
    
    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)
    
    if len(agent.episode_rewards) > 0:
        print(f"\nEpisode Statistics:")
        print(f"  Mean Reward:  {np.mean(agent.episode_rewards):>8.4f}")
        print(f"  Std Reward:   {np.std(agent.episode_rewards):>8.4f}")
        print(f"  Max Reward:   {np.max(agent.episode_rewards):>8.4f}")
        print(f"  Min Reward:   {np.min(agent.episode_rewards):>8.4f}")
    
    print(f"\nLogs saved to: {agent.log_dir}")
    print(f"To view Tensorboard: tensorboard --logdir={agent.log_dir}")
    
    # ============================================================================
    # Save Checkpoint
    # ============================================================================
    
    checkpoint_path = "./checkpoints/ppo_agent_reward_optimized.pt"
    os.makedirs("./checkpoints", exist_ok=True)
    agent.save_checkpoint(checkpoint_path)
    print(f"\n✓ Checkpoint saved to {checkpoint_path}")
    
    # ============================================================================
    # Test checkpoint loading
    # ============================================================================
    
    agent2 = PPOAgentPyTorch(observation_dim=456, device=device)
    agent2.load_checkpoint(checkpoint_path)
    print(f"✓ Checkpoint loaded successfully")
    
    # Single evaluation step
    state = np.random.randn(456) * 0.1
    state = np.clip(state, -10, 10)
    action, log_prob, value = agent2.select_action(state, deterministic=True)
    print(f"\nEvaluation (deterministic):")
    print(f"  Action:  {action}")
    print(f"  Value:   {value:.4f}")


if __name__ == "__main__":
    main()
