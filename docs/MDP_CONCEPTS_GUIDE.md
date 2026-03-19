# MDP Concepts Guide - Trading Bot

## 1. État Initial & Trajectoire

Votre exemple est **exact** :

```python
S(0) → a(0) → r(0) → S(1) → a(1) → r(1) → S(2) → a(2) → r(2) → ...
```

### Exemple Concret : Safe Week Check

```python
S(0) = {
  equity: 10,000 USD,
  is_safe_week: 1,  ← état initial vérifie si semaine sûre
  position: 0 (flat),
  bars_history: [...],
  ...
}

a(0) = OPEN_LONG  ← agent décide d'ouvrir une position
       (peut ouvrir car is_safe_week = 1)

r(0) = 0.0005  ← récompense: +0.05% equity change

S(1) = {
  equity: 10,005 USD,  ← equity a augmenté
  is_safe_week: 1,
  position: 1 (long open),
  entry_price: 1.0850,
  sl: 1.0830,
  ...
}

a(1) = MANAGE_TP  ← agent gère les take-profits

r(1) = 0.0002  ← récompense: +0.02%

S(2) = {...}
```

---

## 2. Propriété de Markov

### Définition

L'état futur dépend **uniquement** de l'état courant & l'action, **pas** de l'histoire passée :

$$P(S_{t+1} | S_t, a_t) = P(S_{t+1} | S_t, a_t, S_{t-1}, a_{t-1}, ..., S_0)$$

### Pourquoi c'est important ?

- Rend le problème **résoluble** (sinon infinité d'états différents)
- Permet la programmation dynamique
- Justifie l'utilisation d'un réseau de neurones (qui résume l'historique en `h_t`)

### Dans votre MDP

✅ **Propriété respectée** :

- L'historique de 64 bars est **encodé** dans `h_t` (observation history)
- L'état `S(t) = (h_t, c_t)` contient toute l'info nécessaire
- Futur ne dépend que de `S(t)` et `a(t)`, pas des états avant `t-1`

---

## 3. Randomness (Stochastic vs Deterministic)

### Deux types de MDP

#### A) Deterministic MDP (votre cas)

```python
P(S' | S, a) = 1 si S' = f(S, a)
             = 0 sinon
```

- Même action → toujours même résultat
- **Votre trading-bot** : données historiques fixes → pas d'aléatoire
- Les prix sont donnés d'avance (backtest)

#### B) Stochastic MDP

```python
P(S' | S, a) = probabilité(S')
```

- Même action → résultats possibles différents
- Exemple : dans un vrai marché, même OPEN_LONG → prix peut monter/descendre différemment

### Dans votre implémentation

```python
# Deterministic transition
next_state, reward = env.step(action)
# Même action & état → toujours même next_state & reward
# Important : simplifie l'apprentissage !
```

---

## 4. État, Action, Récompense : Definitions 1 par 1

### A) État S

**Ne pas définir 1 par 1** → définir l'espace entier

$$\mathcal{S} = \{(h_t, c_t) : h_t \in \mathbb{R}^{384}, c_t \in \mathbb{R}^{7}\}$$

Chaque composant :

- `h_t` : 64 bars avec 6 features (OHLC, spread, ATR)
- `c_t` : contexte (is_tradable, is_safe_week, has_breakout, tp_state, etc.)

### B) Action A

**Ensemble fini** :

```python
a ∈ {0, 1, 2, 3, 4, 5}
   = {HOLD, OPEN_LONG, OPEN_SHORT, CLOSE, PROTECT, MANAGE_TP}
```

Contrainte : ne peut pas OPEN_LONG si une position existe → **gating**

### C) Récompense R

**Règle déterministe** :
$$r(s, a, s') = \frac{\text{equity}(s') - \text{equity}(s)}{\text{equity}(s)} - 10^{-4} \times \mathbb{1}[\text{action invalide}]$$

---

## 5. Stratégie (Policy) π

### Qu'est-ce qu'une Policy

Mappe état → action :

$$\pi: \mathcal{S} \to \mathcal{A}$$

Deux types :

#### Deterministic Policy

$$\pi(s) = a \quad \text{(une seule action par état)}$$

Exemple :

```python
π(S_safe_week) = OPEN_LONG
π(S_not_safe_week) = HOLD
```

#### Stochastic Policy

$$\pi(a|s) = P(a | s) \quad \text{(probabilité sur les actions)}$$

Exemple :

```python
π(OPEN_LONG | S_safe_week) = 0.7
π(HOLD | S_safe_week) = 0.3
```

### Dans votre agent

- **Pendant entraînement** : stochastic (exploration, softmax sur Q-values)
- **Après entraînement** : deterministic (argmax action)

---

## 6. Return & Discount Factor

### Return à temps t

$$G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \gamma^3 r_{t+3} + ...$$

### Exemple avec γ = 0.99

```python
Étape 0: r = +0.001  → contribution = 0.001
Étape 1: r = -0.0002 → contribution = 0.99 × (-0.0002) = -0.000198
Étape 2: r = +0.0005 → contribution = 0.99² × 0.0005 ≈ 0.00049
Étape 3: r = +0.0003 → contribution = 0.99³ × 0.0003 ≈ 0.00029

G(0) = 0.001 - 0.000198 + 0.00049 + 0.00029 + ... ≈ 0.00148
```

### Dans votre trading-bot

$$\gamma = 1.0 \quad \text{(horizon fini, un jour)}$$

Justification :

- Épisode dure ~1 jour (1440 bars M1)
- Récompenses court-terme et long-terme ont égale importance
- Pas de "vraiment loin du futur"

---

## 7. Propriété de Markov : Formalisation

$$P(S_{t+1} | S_t, a_t) = P(S_{t+1} | H_t)$$

où $H_t = (S_0, a_0, r_0, ..., S_t, a_t)$ (l'histoire complète)

### Implications

1. **Sans Markov** : besoin de connaître tout le passé
2. **Avec Markov** : l'observation history `h_t` **suffit**

Votre MDP encode le passé dans `h_t`, donc la propriété est respectée ✓

---

## 8. Objectif Global

Trouver la **policy optimale** $\pi^*$ qui maximise la **expected return** :

$$\pi^* = \arg\max_{\pi} \mathbb{E}[G_0 | \pi]$$

### Deux approches

#### Value-Based (Q-Learning)

- Apprend : $Q^*(s, a)$ = return maximal en prenant action `a` dans état `s`
- Policy : $\pi^*(s) = \arg\max_a Q^*(s, a)$

#### Policy-Based (Policy Gradient)

- Apprend directement : $\pi(a|s)$
- Optimise : $\max_{\pi} \mathbb{E}[G_t]$

---

## 9. Résumé : Votre Séquence

Pour un jour de trading (episode = 1 jour) :

```python
Day Start
├─ S(0) = {equity: 10000, is_safe_week: 1, ...}
├─ a(0) = π(S(0)) = OPEN_LONG  (par policy)
├─ r(0) = +0.0005
├─ S(1) = next_state after OPEN_LONG
│
├─ a(1) = π(S(1)) = MANAGE_TP
├─ r(1) = +0.0002
├─ S(2) = next_state after MANAGE_TP
│
├─ ... (plusieurs autres étapes)
│
└─ Day End
   ├─ S(N) = {equity: 10,050, position: 0, ...}  [closed]
   ├─ a(N) = terminal action (episode fini)
   ├─ G(0) = r(0) + r(1) + ... + r(N)  [total return du jour]
   └─ Agent apprend : cette séquence génère return = +0.005 (0.5%)
```

### Boucle d'apprentissage

```python
for episode in episodes:
    S = env.reset()  # S(0)
    G = 0
    while not done:
        a = π(S)  # policy selects action
        S', r, done = env.step(a)
        G += r
        S = S'
    
    # Améliorer π pour maximiser G
    update_policy(π, G)
```

---

## 10. Prochaine Étape

Pour Phase 2 (README), ajoutez :

1. **Trajectoire exemple** : S(0) → a(0) → r(0) → S(1) → ...
2. **Propriété de Markov** : Simple illustration
3. **Policy** : Schéma "agent voit état, choisit action"
4. **Return** : Calcul avec exemple concret

Voulez-vous que je crée ces exemples dans README ?
