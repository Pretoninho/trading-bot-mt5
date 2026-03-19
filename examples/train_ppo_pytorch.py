"""
Example Training Loop for PPO+GAE PyTorch Agent

Demonstrates:
- Initialization with PyTorch backend
- Episode simulation
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


def simulate_trading_env_step(
    state: np.ndarray,
    action: int,
) -> Tuple[np.ndarray, float, bool]:
    """
    Simulate one step of trading environment.
    
    In production, this would interface with real MT5 or Gymnasium env.
    """
    # Simplified simulation: random next state, Gaussian reward
    next_state = state + np.random.randn(391) * 0.01
    
    # Reward: based on action (simplified)
    if action == 0:  # HOLD
        reward = 0.0001 * np.random.randn()
    elif action == 1:  # LONG
        reward = 0.001 * np.random.randn() + 0.0005
    elif action == 2:  # SHORT
        reward = 0.001 * np.random.randn() - 0.0005
    else:
        reward = np.random.randn() * 0.0001
    
    # Terminate after 100 steps (simplified episode)
    done = False  # (would check max_steps in real scenario)
    
    return next_state, reward, done


def train_episode(
    agent: PPOAgentPyTorch,
    max_steps: int = 100,
) -> dict:
    """
    Simulate one training episode.
    
    Args:
        agent: PPOAgentPyTorch instance
        max_steps: Maximum steps per episode
        
    Returns:
        episode_stats: Dictionary with episode metrics
    """
    # Initialize state (random)
    state = np.random.randn(391) * 0.1
    
    episode_reward = 0
    episode_length = 0
    
    for step in range(max_steps):
        # Select action
        action, log_prob, value = agent.select_action(state)
        
        # Simulate environment step
        next_state, reward, done = simulate_trading_env_step(state, action)
        
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
        
        if done:
            break
        
        state = next_state
    
    return {
        'reward': episode_reward,
        'length': episode_length,
    }


def main():
    """Main training loop."""
    
    print("=" * 80)
    print("PPO+GAE PyTorch Agent - Training Example")
    print("=" * 80)
    
    # ============================================================================
    # Initialize Agent
    # ============================================================================
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n🖥️  Device: {device}")
    
    agent = PPOAgentPyTorch(
        observation_dim=391,
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
        log_dir="./logs/ppo_training",
    )
    
    print(f"✓ Agent initialized")
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
        # Run episode
        stats = train_episode(agent, max_steps=100)
        
        print(f"\n[Episode {episode:3d}]")
        print(f"  Reward: {stats['reward']:>8.4f}")
        print(f"  Length: {stats['length']:>8d}")
        
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
    
    checkpoint_path = "./checkpoints/ppo_agent.pt"
    os.makedirs("./checkpoints", exist_ok=True)
    agent.save_checkpoint(checkpoint_path)
    print(f"\n✓ Checkpoint saved to {checkpoint_path}")
    
    # ============================================================================
    # Test checkpoint loading
    # ============================================================================
    
    agent2 = PPOAgentPyTorch(device=device)
    agent2.load_checkpoint(checkpoint_path)
    print(f"✓ Checkpoint loaded successfully")
    
    # Single evaluation step
    state = np.random.randn(391) * 0.1
    action, log_prob, value = agent2.select_action(state, deterministic=True)
    print(f"\nEvaluation (deterministic):")
    print(f"  Action:  {action}")
    print(f"  Value:   {value:.4f}")


if __name__ == "__main__":
    main()
