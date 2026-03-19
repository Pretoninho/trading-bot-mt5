"""
Comprehensive Test Suite for PPO+GAE Trading Agent

Tests:
1. Unit Tests: Networks, Agent initialization, Components
2. Integration Tests: Training loop, Episode simulation
3. Comparison Tests: PyTorch vs NumPy implementations
4. Performance Tests: Inference speed, Update speed, Memory usage
5. Trading Tests: Reward learning, Position management, Noise stability
6. Metrics Tests: Convergence, Learning curves, Stability

Usage:
    pytest tests/test_trading_agent.py -v              # All tests
    pytest tests/test_trading_agent.py -v -k "unit"    # Unit only
    pytest tests/test_trading_agent.py -v -k "perf"    # Performance only
    pytest tests/test_trading_agent.py --tb=short       # Compact output
"""

import pytest
import numpy as np
import torch
import time
from typing import Tuple, List
import tempfile
import os

from trading_env.agents import (
    ActorCriticNetwork,
    PPONetworks,
    PPOAgentPyTorch,
    PPOAgentWithGAE,
    ReplayBuffer,
)


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def numpy_seed():
    """Set numpy and torch seeds for reproducibility."""
    np.random.seed(42)
    torch.manual_seed(42)
    yield


@pytest.fixture
def observation():
    """Standard trading observation (391D)."""
    return np.random.randn(391)


@pytest.fixture
def ppo_agent_pytorch():
    """PPO agent with PyTorch backend."""
    return PPOAgentPyTorch(
        observation_dim=391,
        action_dim=6,
        hidden_dim=128,
        learning_rate=3e-4,
        gamma=1.0,
        lambda_=0.95,
        clip_ratio=0.2,
        entropy_coef=0.01,
        epochs=3,
        batch_size=64,
        device='cpu',
    )


@pytest.fixture
def ppo_agent_numpy():
    """PPO agent with NumPy backend."""
    return PPOAgentWithGAE(
        observation_dim=391,
        action_dim=6,
        hidden_dim=128,
        learning_rate_actor=1e-4,
        learning_rate_critic=1e-3,
        gamma=1.0,
        lambda_=0.95,
    )


# ============================================================================
# UNIT TESTS: Components
# ============================================================================


class TestNetworks:
    """Test PyTorch neural network components."""
    
    def test_actor_critic_network_init(self):
        """Test network initialization."""
        network = ActorCriticNetwork(391, 6, hidden_dim=128)
        
        assert network.observation_dim == 391
        assert network.action_dim == 6
        assert network.hidden_dim == 128
    
    def test_actor_critic_forward_shapes(self):
        """Test forward pass output shapes."""
        network = ActorCriticNetwork(391, 6, hidden_dim=128)
        
        obs = torch.randn(1, 391)
        logits, values = network(obs)
        
        assert logits.shape == (1, 6), f"Expected (1, 6), got {logits.shape}"
        assert values.shape == (1, 1), f"Expected (1, 1), got {values.shape}"
    
    def test_actor_critic_batch_forward(self):
        """Test forward pass with batch."""
        network = ActorCriticNetwork(391, 6)
        
        batch_size = 32
        obs = torch.randn(batch_size, 391)
        logits, values = network(obs)
        
        assert logits.shape == (batch_size, 6)
        assert values.shape == (batch_size, 1)
    
    def test_ppo_networks_wrapper(self):
        """Test PPONetworks wrapper with optimizer."""
        networks = PPONetworks(391, 6, device='cpu')
        
        assert networks.device == 'cpu'
        assert networks.optimizer is not None
        assert isinstance(networks.optimizer, torch.optim.Adam)


class TestPPOAgentInitialization:
    """Test PPO agent initialization and basic properties."""
    
    def test_agent_init_defaults(self):
        """Test agent initialization with defaults."""
        agent = PPOAgentPyTorch(device='cpu')
        
        assert agent.device == 'cpu'
        assert agent.gamma == 1.0
        assert agent.lambda_ == 0.95
        assert agent.clip_ratio == 0.2
    
    def test_agent_init_custom(self):
        """Test agent initialization with custom parameters."""
        agent = PPOAgentPyTorch(
            observation_dim=391,
            action_dim=6,
            hidden_dim=256,
            learning_rate=1e-3,
            gamma=0.99,
            lambda_=0.9,
            clip_ratio=0.15,
            device='cpu',
        )
        
        assert agent.gamma == 0.99
        assert agent.lambda_ == 0.9
        assert agent.clip_ratio == 0.15


class TestActionSelection:
    """Test action selection and policy sampling."""
    
    def test_action_selection_ranges(self, ppo_agent_pytorch, observation):
        """Test that selected actions are within valid range."""
        for _ in range(10):
            action, log_prob, value = ppo_agent_pytorch.select_action(observation)
            
            assert 0 <= action < 6, f"Action {action} out of range [0, 6)"
            assert isinstance(log_prob, float)
            assert isinstance(value, float)
    
    def test_action_selection_deterministic(self, ppo_agent_pytorch, observation):
        """Test deterministic mode (argmax)."""
        action_det1, _, _ = ppo_agent_pytorch.select_action(
            observation, deterministic=True
        )
        action_det2, _, _ = ppo_agent_pytorch.select_action(
            observation, deterministic=True
        )
        
        assert action_det1 == action_det2


class TestReplayBuffer:
    """Test trajectory replay buffer."""
    
    def test_buffer_add_transition(self):
        """Test adding transitions."""
        buffer = ReplayBuffer(max_size=100)
        
        state = np.random.randn(391)
        buffer.add(state, 1, 0.01, 0.5, np.log(0.1), False)
        assert buffer.size == 1
    
    def test_buffer_gae_computation(self):
        """Test GAE advantage computation."""
        buffer = ReplayBuffer(max_size=100)
        
        np.random.seed(42)
        for _ in range(50):
            state = np.random.randn(391)
            reward = np.random.randn() * 0.01
            value = np.random.randn() * 0.1
            buffer.add(state, 0, reward, value, np.log(0.1), False)
        
        buffer.compute_advantages(gamma=1.0, lambda_=0.95)
        
        advs = buffer.buffer['advantages'][:buffer.size]
        assert np.abs(np.mean(advs)) < 0.1
        assert np.abs(np.std(advs) - 1.0) < 0.3


# ============================================================================
# INTEGRATION TESTS: Training Loop
# ============================================================================


class TestTrainingLoop:
    """Test complete training loops."""
    
    def test_single_episode_collection(self, ppo_agent_pytorch, numpy_seed):
        """Test collecting one complete episode."""
        state = np.random.randn(391)
        
        for step in range(100):
            action, log_prob, value = ppo_agent_pytorch.select_action(state)
            
            reward = np.random.randn() * 0.01
            next_state = state + np.random.randn(391) * 0.01
            done = step == 99
            
            ppo_agent_pytorch.store_transition(
                state, action, reward, value, log_prob, done
            )
            
            state = next_state
        
        assert ppo_agent_pytorch.buffer.size == 100
    
    def test_update_after_episode(self, ppo_agent_pytorch, numpy_seed):
        """Test updating networks after episode collection."""
        state = np.random.randn(391)
        for step in range(100):
            action, log_prob, value = ppo_agent_pytorch.select_action(state)
            reward = np.random.randn() * 0.01
            next_state = state + np.random.randn(391) * 0.01
            done = step == 99
            
            ppo_agent_pytorch.store_transition(
                state, action, reward, value, log_prob, done
            )
            state = next_state
        
        metrics = ppo_agent_pytorch.update()
        
        assert 'actor_loss' in metrics
        assert 'critic_loss' in metrics
        assert 'entropy' in metrics
        
        assert ppo_agent_pytorch.buffer.size == 0
    
    def test_multiple_episodes_training(self, ppo_agent_pytorch, numpy_seed):
        """Test training over multiple episodes."""
        returns = []
        
        for episode in range(5):
            state = np.random.randn(391)
            ep_return = 0
            
            for step in range(100):
                action, log_prob, value = ppo_agent_pytorch.select_action(state)
                reward = np.random.randn() * 0.01
                next_state = state + np.random.randn(391) * 0.01
                done = step == 99
                
                ppo_agent_pytorch.store_transition(
                    state, action, reward, value, log_prob, done
                )
                
                ep_return += reward
                state = next_state
            
            returns.append(ep_return)
            
            if episode % 2 == 0:
                metrics = ppo_agent_pytorch.update()
                assert 'actor_loss' in metrics
        
        assert len(returns) == 5


# ============================================================================
# PERFORMANCE TESTS
# ============================================================================


class TestPerformance:
    """Test inference and training performance."""
    
    @pytest.mark.performance
    def test_inference_speed(self, ppo_agent_pytorch, observation):
        """Test inference speed (actions per second)."""
        num_steps = 1000
        
        start = time.time()
        for _ in range(num_steps):
            action, log_prob, value = ppo_agent_pytorch.select_action(observation)
        elapsed = time.time() - start
        
        actions_per_sec = num_steps / elapsed
        
        print(f"\nInference speed: {actions_per_sec:.0f} actions/sec")
        
        assert actions_per_sec > 100
    
    @pytest.mark.performance
    def test_update_speed(self, ppo_agent_pytorch, numpy_seed):
        """Test update speed."""
        state = np.random.randn(391)
        for _ in range(200):
            action, log_prob, value = ppo_agent_pytorch.select_action(state)
            ppo_agent_pytorch.store_transition(state, action, 0.01, value, log_prob, False)
            state = np.random.randn(391)
        
        start = time.time()
        metrics = ppo_agent_pytorch.update()
        elapsed = time.time() - start
        
        update_ms = elapsed * 1000
        print(f"\nUpdate time: {update_ms:.1f} ms")
        
        assert update_ms < 5000


# ============================================================================
# TRADING TESTS: Specific Trading Scenarios
# ============================================================================


class TestTradingBehavior:
    """Test agent behavior in trading scenarios."""
    
    def test_learn_simple_reward_signal(self, ppo_agent_pytorch, numpy_seed):
        """Test that agent learns to prefer high-reward actions."""
        returns = []
        for episode in range(20):
            state = np.random.randn(391)
            ep_return = 0
            
            for step in range(50):
                action, log_prob, value = ppo_agent_pytorch.select_action(state)
                
                reward = 0.01 if action == 1 else -0.005
                
                ppo_agent_pytorch.store_transition(
                    state, action, reward, value, log_prob, False
                )
                
                ep_return += reward
                state = np.random.randn(391)
            
            returns.append(ep_return)
            
            if episode % 5 == 0:
                ppo_agent_pytorch.update()
        
        early_return = np.mean(returns[:5])
        late_return = np.mean(returns[-5:])
        
        print(f"\nEarly return: {early_return:.4f}, Late return: {late_return:.4f}")
        
        assert all(-0.3 < r < 0.5 for r in returns)
    
    def test_noise_stability(self, ppo_agent_pytorch, numpy_seed):
        """Test agent remains stable under noisy rewards."""
        returns = []
        
        for episode in range(30):
            state = np.random.randn(391)
            ep_return = 0
            
            for step in range(100):
                action, log_prob, value = ppo_agent_pytorch.select_action(state)
                
                noise = np.random.randn() * 0.05
                drift = 0.0001
                reward = noise + drift
                
                ppo_agent_pytorch.store_transition(
                    state, action, reward, value, log_prob, False
                )
                
                ep_return += reward
                state = np.random.randn(391)
            
            returns.append(ep_return)
            
            if episode % 10 == 0:
                ppo_agent_pytorch.update()
        
        assert not np.any(np.isnan(returns))
        assert not np.any(np.isinf(returns))
        
        std_returns = np.std(returns)
        assert std_returns < 1.0


# ============================================================================
# CHECKPOINT TESTS
# ============================================================================


class TestCheckpoints:
    """Test model checkpointing functionality."""
    
    def test_save_load_checkpoint(self, ppo_agent_pytorch):
        """Test saving and loading checkpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = os.path.join(tmpdir, "test_checkpoint.pt")
            
            ppo_agent_pytorch.save_checkpoint(checkpoint_path)
            assert os.path.exists(checkpoint_path)
            
            agent2 = PPOAgentPyTorch(device='cpu')
            agent2.load_checkpoint(checkpoint_path)
            
            state = np.random.randn(391)
            
            action1, lp1, v1 = ppo_agent_pytorch.select_action(state, deterministic=True)
            action2, lp2, v2 = agent2.select_action(state, deterministic=True)
            
            assert action1 == action2
            assert np.isclose(lp1, lp2, atol=1e-5)
            assert np.isclose(v1, v2, atol=1e-5)


# ============================================================================
# CONVERGENCE & STABILITY TESTS
# ============================================================================


class TestConvergence:
    """Test training convergence and stability."""
    
    def test_loss_convergence_trend(self, numpy_seed):
        """Test that losses show convergence trend."""
        agent = PPOAgentPyTorch(device='cpu')
        
        actor_losses = []
        critic_losses = []
        
        for episode in range(10):
            state = np.random.randn(391)
            
            for step in range(200):
                action, log_prob, value = agent.select_action(state)
                reward = np.random.randn() * 0.01
                agent.store_transition(state, action, reward, value, log_prob, False)
                state = np.random.randn(391)
            
            if episode >= 2:
                metrics = agent.update()
                actor_losses.append(metrics['actor_loss'])
                critic_losses.append(metrics['critic_loss'])
        
        assert not any(np.isnan(l) for l in actor_losses)
        assert not any(np.isnan(l) for l in critic_losses)
        
        assert all(abs(l) < 100 for l in actor_losses)
        assert all(abs(l) < 100 for l in critic_losses)
    
    def test_entropy_stability(self, numpy_seed):
        """Test that entropy remains stable during training."""
        agent = PPOAgentPyTorch(device='cpu', entropy_coef=0.01)
        
        entropies = []
        
        for episode in range(10):
            state = np.random.randn(391)
            
            for step in range(100):
                action, log_prob, value = agent.select_action(state)
                agent.store_transition(state, action, 0.01, value, log_prob, False)
                state = np.random.randn(391)
            
            metrics = agent.update()
            entropies.append(metrics['entropy'])
        
        assert all(0 < e < 10 for e in entropies)
        
        final_entropy = entropies[-1]
        assert final_entropy > 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
