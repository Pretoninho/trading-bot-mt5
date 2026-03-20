# DEPLOYMENT GUIDE

## Production Deployment Checklist ✓

### Phase 1: Pre-Deployment Validation (COMPLETE)
- [x] All 126 unit tests passing
- [x] Reward system integration verified
- [x] Training pipeline validated
- [x] Convergence criteria met
- [x] Deployment configuration created

### Phase 2: Environment Setup

#### 2.1 System Requirements
```bash
# GPU support (recommended)
python -c "import torch; print(f'GPU Available: {torch.cuda.is_available()}')"

# Python version (3.10+)
python --version

# Memory check
free -h  # At least 8GB RAM recommended, 16GB+ recommended
```

#### 2.2 Dependencies
```bash
# Install/update required packages
pip install -r requirements.txt

# Verify installation
python -c "import trading_env; import torch; print('All dependencies OK')"
```

#### 2.3 Data Setup
```bash
# Ensure market data is available
ls -la trading_env/data/

# Generate/download EURUSD M1 data
python -c "from trading_env.data.market_loader import load_mt5_m1_csv; print('Data loader ready')"
```

### Phase 3: Model Training

#### 3.1 Training with Production Config
```bash
# Train with balanced reward profile (recommended)
python examples/train_ppo_pytorch.py

# Monitor training
tensorboard --logdir=./logs/production
```

#### 3.2 Custom Training Script
```python
from DEPLOYMENT_CONFIG import get_agent_config, TRAINING_CONFIG
from trading_env.agents import PPOAgentPyTorch
from trading_env.utils.reward_functions import RewardCalculator

# Get configuration
agent_config = get_agent_config(profile="balanced")
reward_config = agent_config.pop("reward_config")

# Initialize agent with production settings
agent = PPOAgentPyTorch(**agent_config)
reward_calc = RewardCalculator(reward_config)

# Train for specified episodes
num_episodes = TRAINING_CONFIG["num_episodes"]
for episode in range(num_episodes):
    # ... training loop ...
    pass

# Save final model
agent.save_checkpoint("./checkpoints/production/final_model.pt")
```

### Phase 4: Validation & Testing

#### 4.1 Unit Tests
```bash
# Run full test suite
pytest tests/ -v

# Run specific test files
pytest tests/test_reward.py -v
pytest tests/test_features.py -v
```

#### 4.2 Integration Tests
```bash
# Test reward system integration
python validate_convergence.py

# Test benchmarking
python benchmark_reward_configs.py
```

#### 4.3 Production Validation
```bash
# Verify deployment configuration
python DEPLOYMENT_CONFIG.py

# Check model checkpoint
python -c "
from trading_env.agents import PPOAgentPyTorch
agent = PPOAgentPyTorch(observation_dim=456)
agent.load_checkpoint('./checkpoints/production/final_model.pt')
print('✓ Checkpoint loaded successfully')
"
```

### Phase 5: Deployment

#### 5.1 Backward Compatibility Check
```bash
# Ensure legacy code still works
python -c "
import numpy as np
from trading_env.env.trading_env import EURUSDTradingEnv
from trading_env.data.market_loader import load_mt5_m1_csv

print('✓ Environment compatible')
print('✓ Data loader compatible')
"
```

#### 5.2 Production Checkpoints
```bash
# Create production checkpoint directory
mkdir -p ./checkpoints/production

# Copy trained model
cp ./checkpoints/ppo_agent_reward_optimized.pt ./checkpoints/production/model_v1.pt

# Archive for backup
tar -czf model_backup_$(date +%Y%m%d).tar.gz ./checkpoints/production/
```

#### 5.3 Configuration Deployment
```bash
# Review production config
grep -A5 "PRODUCTION\|RISK\|MONITORING" DEPLOYMENT_CONFIG.py

# Customize for your environment (if needed)
# Edit DEPLOYMENT_CONFIG.py with your settings
```

### Phase 6: Monitoring & Operations

#### 6.1 Real-Time Monitoring
```bash
# Start TensorBoard for live monitoring
tensorboard --logdir=./logs/production

# Monitor system resources
watch -n 5 'nvidia-smi'  # GPU if available
top -u $(whoami)         # CPU/Memory

# Check log files
tail -f logs/production/*.log
```

#### 6.2 Risk Management
- Daily equity checks: Monitor absolute loss/gain vs `RISK_CONFIG`
- Drawdown alerts: Alert if equity drawdown > 15%
- Position limits: Enforce max 1 concurrent position
- Trading hours: Only trade 08:00-22:00 UTC

#### 6.3 Performance Tracking
```bash
# Track key metrics
python -c "
from DEPLOYMENT_CONFIG import MONITORING_CONFIG, TRAINING_CONFIG
print('Monitoring Config:', MONITORING_CONFIG)
print('Training Config:', TRAINING_CONFIG)
"
```

### Phase 7: Rollback Plan

#### 7.1 Checkpoint Management
```bash
# List available checkpoints
ls -la checkpoints/

# Restore previous version if needed
cp ./checkpoints/backup/model_old.pt ./checkpoints/production/model_v1.pt
```

#### 7.2 Quick Rollback
```bash
# Stop current model
# ... stop trading bot ...

# Revert to known-good checkpoint
# ... restart with previous config ...

# Verify operation
# ... run validation tests ...
```

## Reward System Details

### Multi-Component Reward (Enhanced Profile)
```
Total Reward = 1.0 × equity_return
              + 0.05 × sharpe_ratio_bonus
              + 0.03 × streak_bonus
              + 0.02 × duration_reward
              - 0.05 × drawdown_penalty
              - 0.005 × opportunity_cost_penalty
              - 1e-4 × invalid_action_penalty
              (clipped to [-1.0, +1.0])
```

### Profiles Available
- **conservative** (baseline): Original equity-only reward
- **balanced** (enhanced): All components with moderate weights ✓
- **aggressive**: All components with higher weights

## Troubleshooting

### Common Issues

**Q: GPU not being used**
```bash
python -c "import torch; print(torch.cuda.is_available())"
# If False: Install proper CUDA/PyTorch version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Q: Out of memory errors**
- Reduce `batch_size` in DEPLOYMENT_CONFIG
- Reduce `num_episodes` for initial tests
- Use CPU if GPU memory insufficient

**Q: Model not converging**
- Check reward components are being calculated
- Verify training data quality
- Increase `num_episodes` or `epochs`
- Try `conservative` reward profile first

**Q: Tests failing after deployment**
```bash
# Run individual test suites
pytest tests/test_reward.py::TestRewardCalculator -v
pytest tests/test_features.py::TestFeatureEngineer -v
```

## Support & Monitoring

### Key Metrics to Track
- Episode returns (should trend upward)
- Actor/Critic losses (should converge)
- Policy entropy (should stabilize)
- Drawdown vs threshold (alert if exceeded)

### Production Health Checks
```bash
python validate_convergence.py  # Daily
python benchmark_reward_configs.py  # Weekly
pytest tests/ -q  # Before each deployment
```

### Deployment Rollback Criteria
- Unreasonable number of invalid actions (>50%)
- Convergence divergence (losses increasing)
- Consecutive losses >5 trades
- Equity drawdown >15%

## Version Info
- **Agent Version**: v1.0.0
- **Reward System**: v1.0.0
- **Features**: 456D (391 base + 65 engineered)
- **Deployment Date**: [Set at deployment]
- **Python**: 3.10+
- **PyTorch**: 2.0+

---

**Status**: ✅ READY FOR PRODUCTION DEPLOYMENT

For questions or issues, refer to:
- [README.md](README.md) - Project overview
- [DEPLOYMENT_CONFIG.py](DEPLOYMENT_CONFIG.py) - Configuration details
- [trading_env/utils/reward_functions.py](trading_env/utils/reward_functions.py) - Reward system docs
