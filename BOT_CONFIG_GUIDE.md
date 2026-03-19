<!-- markdownlint-disable MD013 MD060 -->

# 🤖 Bot Configuration Interface

Configuration web interactive pour le Trading Bot EURUSD M1 PPO.

## Installation

```bash
# Installer les dépendances
pip install -r requirements.txt

# Ou si vous avez des problèmes avec torch, installer séparément
pip install streamlit>=1.28.0
```

## Lancer l'interface

```bash
streamlit run bot_config_interface.py
```

L'interface s'ouvrira à : [http://localhost:8501](http://localhost:8501)

---

## 📋 Sections de configuration

### 1️⃣ **📊 Data Files**

- **Market Data (EURUSD M1)** : Upload votre CSV MT5 M1
  - Format : `DATE TIME OPEN HIGH LOW CLOSE TICKVOL VOL SPREAD` (séparé par espaces)
  - Exemple : `2025.01.06 00:00:00 1.02345 1.02390 1.02310 1.02360 312 0 8`

- **Unsafe Weeks** (optionnel) : Semaines à éviter (CPI/NFP/FOMC)

- **FXStreet Calendar** : Calendrier économique par trimestre (q1-q4)

### 2️⃣ **⚙️ Environment**

Configuration de l'environnement de trading :

| Paramètre             | Range       | Défaut | Effet                              |
| --------------------- | ----------- | ------ | ---------------------------------- |
| **Equity gain limit** | 0.01-0.1%   | 2%     | Episode s'arrête si +X%            |
| **Equity loss limit** | -0.1 à -0.01% | -2%    | Episode s'arrête si -X%            |
| **ATR Period**       | 5-30 bars   | 14     | Période pour calcul stop-loss       |
| **ATR SL Multiplier** | 1-4x        | 2.0    | SL = ATR × ce multiple             |
| **TP1 R-multiple**   | 0.5-3       | 1.0    | Premier take-profit à 1R           |
| **TP2 R-multiple**   | 1-5         | 2.0    | Deuxième take-profit à 2R          |
| **Partial close**    | 10-50%      | 25%    | % position fermée à chaque TP      |
| **Observation window** | 32-256 bars | 64     | Contexte historique pour l'agent   |

### 3️⃣ **💰 Position Sizing**

Gestion du sizing des positions :

| Paramètre          | Min  | Max | Défaut |
| ------------------ | ---- | --- | ------ |
| **Min lot**        | 0.01 | 1.0 | 0.01   |
| **Lot step**       | 0.01 | 0.1 | 0.01   |
| **Max lot**        | 1.0  | 100 | 100    |
| **Risk per trade** | 0.1% | 5%  | 1%     |

**Formule de calcul** :

```text
Equity Risk = Equity × Risk%
Stop Loss Distance = |Entry - SL| (pips)
Loss per Lot = SL_Pips × $10/pip
Lots = Equity Risk / Loss per Lot
```

### 4️⃣ **🧠 Training**

Paramètres d'entraînement PPO+GAE :

| Paramètre            | Min    | Max  | Défaut | Impact                  |
| -------------------- | ------ | ---- | ------ | ----------------------- |
| **Initial equity**   | -      | -    | $10k   | Capital de départ       |
| **Device**           | cpu/cuda | -    | cpu    | GPU ou CPU              |
| **Hidden dim**       | 64     | 512  | 128    | Capacité réseau         |
| **Learning rate**    | 1e-5   | 1e-3 | 3e-4   | Vitesse convergence     |
| **Batch size**       | 32     | 256  | 64     | Samples par update      |
| **Gamma**            | 0.9    | 1.0  | 1.0    | Valeur futur vs présent |
| **Lambda GAE**       | 0.9    | 0.99 | 0.95   | Lissage avantages       |
| **Epochs per update** | 1      | 10   | 4      | Passes d'entraînement   |
| **Max grad norm**    | 0.1    | 1.0  | 0.5    | Clipping gradients      |

### 5️⃣ **🎯 Gates & Features**

Trading gates et features avancées :

| Feature                   | Type  | Défaut | Effet                              |
| ------------------------- | ----- | ------ | ---------------------------------- |
| **Safe week gating**      | Toggle | ✅ ON  | Bloque trades semaines CPI/NFP     |
| **Tradable window**       | Toggle | ✅ ON  | Trades seulement mer/jeu 13-21 UTC |
| **Breakout detection**    | Toggle | ✅ ON  | Détecte breakouts hebdo            |
| **Trailing stop**         | Toggle | ✅ ON  | Trail SL dans direction profit     |
| **Break-even buffer**     | Pips  | 0      | Buffer break-even move             |
| **Invalid action penalty** | Float  | -1e-4  | Pénalité actions bloquées          |

---

## 💾 Utilisation de la config

### Sauvegarder

```python
from bot_config import BotConfig

config = BotConfig()
config.learning_rate = 1e-4
config.initial_equity = 50_000
config.save("mon_config.json")
```

### Charger

```python
from bot_config import BotConfig

config = BotConfig.load("mon_config.json")
print(f"Learning rate: {config.learning_rate}")
print(f"Initial equity: ${config.initial_equity}")
```

### Utiliser dans training

```python
from bot_config import BotConfig
from trading_env.env.trading_env import EURUSDTradingEnv
from trading_env.agents.ppo_pytorch import PPOAgentPyTorch

# Charger config
config = BotConfig.load("bot_config.json")

# Créer environnement
env = EURUSDTradingEnv(
    df,
    initial_equity=config.initial_equity,
    render_mode=config.render_mode,
)

# Créer agent
agent = PPOAgentPyTorch(
    observation_dim=config.observation_window * 6 + 7,
    action_dim=6,
    hidden_dim=config.hidden_dim,
    learning_rate=config.learning_rate,
    gamma=config.gamma,
)

# Entraîner...
```

---

## 🎁 Paramètres "cachés" (que vous ne pensiez pas à ajuster)

| Paramètre             | Défaut | Min | Max | Suggestion                                              |
| --------------------- | ------ | --- | --- | ------------------------------------------------------- |
| **Observation window** | 64 bars | 32  | 256 | ↑ Plus de contexte = meilleure décision mais plus lent  |
| **Hidden dim**        | 128    | 64  | 512 | 64=rapide/léger, 256=balancé, 512=puissant             |
| **Lambda GAE**        | 0.95   | 0.9 | 0.99 | 0.99=moins variance, 0.9=plus lissé                   |
| **Max grad norm**     | 0.5    | 0.1 | 1.0 | 0.5=conservateur, 1.0=agressif                        |
| **Break-even buffer** | 0 pips | 0   | 10  | 0=exact, 5+=plus safe                                  |
| **Batch size**        | 64     | 32  | 256 | ↑ Plus gros = moins bruyant mais plus RAM              |
| **Epochs per update** | 4      | 1   | 10  | ↑ Plus d'epochs = meilleur fit mais lent               |

---

## 🚀 Workflow recommandé

1. **Ouvrir l'interface** → `streamlit run bot_config_interface.py`
2. **Section Data Files** → Uploader EURUSD CSV + calendrier FXStreet
3. **Tuner sections** ⚙️ → Environment / Training / Position Sizing
4. **Valider gates** 🎯 → Activer/désactiver gating features
5. **Sauvegarder** 💾 → "Save Configuration" ou "Download JSON"
6. **Utiliser dans code** → `BotConfig.load("bot_config.json")`

---

## 🐛 Troubleshooting

### Module not found: bot_config

Assurez-vous que `bot_config.py` est à la racine du projet avec `bot_config_interface.py`

### Streamlit demande des dependences PyTorch

Installez torch séparément :

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Les fichiers uploadés ne sont pas sauvegardés

Files uploadés via Streamlit ne sont que dans la session. Copiez-les manuellement dans le projet.

---

## 📖 Références

- [Streamlit Docs](https://docs.streamlit.io)
- [BotConfig Dataclass](./bot_config.py)
- [Training Example](./examples/train_ppo_pytorch.py)
