# Détection de Fraude par Carte Bancaire avec Deep Q-Network (DQN)

## Description du Projet

Ce projet implémente un système innovant de détection de fraude utilisant le **Reinforcement Learning** (Apprentissage par Renforcement) avec un algorithme **DQN (Deep Q-Network)**. Contrairement aux approches traditionnelles de classification supervisée, l'agent apprend à identifier les transactions frauduleuses en maximisant une récompense au fil des interactions.

## Objectifs

- Détecter les transactions frauduleuses en temps réel
- Utiliser le Reinforcement Learning pour une approche adaptative
- Gérer le déséquilibre des classes via un système de récompenses pondéré
- Apprendre de manière continue à partir des patterns de fraude

## Architecture du Modèle

### Deep Q-Network (DQN)
```python
Architecture: [Input] → [Dense(64)] → [ReLU] → [Dropout(0.2)] 
              → [Dense(32)] → [ReLU] → [Dropout(0.2)] 
              → [Output(2)]
```

**Composants principaux:**
- **Q-Network**: Réseau principal pour estimer les valeurs Q
- **Target Network**: Réseau cible pour stabiliser l'apprentissage
- **Replay Buffer**: Mémoire d'expérience (capacité: 5,000 transitions)
- **Epsilon-Greedy**: Exploration vs Exploitation

### Système de Récompenses
```python
Détection correcte de fraude:     +10.0
Transaction légitime correcte:    +1.0
Fraude manquée:                   -5.0
Faux positif:                     -1.0
```

## Dataset

**Fichier source:** `fraudTrain_balanced_smote.csv`

**Caractéristiques après prétraitement:**
- Nombre total de transactions: 70,000
- Distribution des classes:
  - Classe 0 (légitime): 50,000 (71.4%)
  - Classe 1 (fraude): 20,000 (28.6%)
- Features: 10 variables numériques

**Variables principales:**
```
- cc_num: Numéro de carte bancaire
- amt: Montant de la transaction
- zip: Code postal
- lat, long: Coordonnées géographiques
- city_pop: Population de la ville
- unix_time: Timestamp
- merch_lat, merch_long: Coordonnées du marchand
```

## Technologies Utilisées

```python
Python 3.x
- PyTorch: Deep Learning et DQN
- NumPy: Calculs numériques
- Pandas: Manipulation de données
- Scikit-learn: Prétraitement et métriques
- Matplotlib: Visualisation
```

## Installation

```bash
# Créer un environnement virtuel
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows

# Installer les dépendances
pip install torch numpy pandas scikit-learn matplotlib
```

## Utilisation

### 1. Structure du Code

Le notebook est organisé en 10 étapes:

1. **Imports et Configuration**
2. **Définition du Réseau DQN**
3. **Replay Buffer**
4. **Agent DQN**
5. **Environnement de Fraude**
6. **Préparation des Données**
7. **Entraînement**
8. **Évaluation**
9. **Visualisation**
10. **Sauvegarde du Modèle**

### 2. Exécution

```python
# Charger le dataset
df = pd.read_csv('fraudTrain_balanced_smote.csv')

# Split Train/Test (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Créer l'environnement
env_train = FraudDetectionEnv(X_train, y_train)

# Créer l'agent DQN
agent = DQNAgent(state_size=10, lr=0.001, gamma=0.95)

# Entraîner sur 10 épisodes
episodes = 10
for episode in range(episodes):
    state = env_train.reset()
    while not done:
        action = agent.select_action(state)
        next_state, reward, done = env_train.step(action)
        agent.memory.push(state, action, reward, next_state, done)
        agent.train_step(batch_size=128)
        state = next_state
```

### 3. Évaluation sur le Test Set

```python
env_test = FraudDetectionEnv(X_test, y_test)
state = env_test.reset()

predictions = []
for i in range(len(X_test)):
    action = agent.select_action(state, training=False)
    predictions.append(action)
    state, _, done = env_test.step(action)
```

## Résultats

### Métriques de Performance (Test Set)

```
                Precision    Recall    F1-Score    Support
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Non-Fraude        0.94      0.68      0.79       10,000
Fraude            0.53      0.89      0.66        4,000
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Accuracy                              0.74       14,000
Macro Avg         0.73      0.79      0.73       14,000
Weighted Avg      0.82      0.74      0.75       14,000
```

### Matrice de Confusion

```
                    Predicted
                Non-Fraud    Fraud
Actual  ┌─────────────────────────┐
Non-F.  │   6,830      3,170      │
Fraud   │     434      3,566      │
        └─────────────────────────┘
```

### Métriques Détaillées

- **Accuracy**: 74.26%
- **Precision (Fraude)**: 52.94%
- **Recall (Fraude)**: 89.15% (Excellente détection des fraudes)
- **F1-Score**: 66.43%

### Courbes d'Apprentissage

Les graphiques générés montrent:
1. **Récompense Totale par Episode**: Progression ~42,000 → ~43,000
2. **Moyenne Mobile (10 épisodes)**: Tendance stable à ~42,500
3. **Loss d'Entraînement**: Convergence progressive

## Hyperparamètres

```python
# Agent DQN
learning_rate = 0.001
gamma = 0.95              # Facteur de discount
epsilon = 1.0             # Exploration initiale
epsilon_decay = 0.995
epsilon_min = 0.01

# Entraînement
episodes = 10
batch_size = 128
replay_buffer_size = 5000
hidden_layers = [64, 32]
dropout_rate = 0.2
```

## Sauvegarde et Chargement

### Sauvegarder le Modèle

```python
torch.save({
    'model_state_dict': agent.q_network.state_dict(),
    'optimizer_state_dict': agent.optimizer.state_dict(),
    'episode_rewards': episode_rewards,
    'losses': agent.losses
}, 'dqn_fraud_detection_model.pth')
```

### Charger le Modèle

```python
checkpoint = torch.load('dqn_fraud_detection_model.pth')
agent.q_network.load_state_dict(checkpoint['model_state_dict'])
agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
```

## Avantages du DQN pour la Détection de Fraude

### Points Forts

1. **Apprentissage Adaptatif**: L'agent s'améliore continuellement
2. **Gestion du Déséquilibre**: Système de récompenses pondéré
3. **Détection en Temps Réel**: Inférence rapide
4. **Haute Sensibilité**: Recall de 89.15% (minimise les fraudes manquées)

### Limitations

1. **Précision Modérée**: 52.94% (taux de faux positifs élevé)
2. **Coût Computationnel**: Entraînement plus lourd que ML classique
3. **Données Nécessaires**: Requiert beaucoup de transitions

## Améliorations Futures

- [ ] **Double DQN**: Réduire la surestimation des valeurs Q
- [ ] **Dueling DQN**: Séparer l'estimation de la valeur et de l'avantage
- [ ] **Prioritized Experience Replay**: Apprentissage sur transitions importantes
- [ ] **Rainbow DQN**: Combiner toutes les améliorations
- [ ] **Ajustement des Récompenses**: Optimiser le ratio de pénalités
- [ ] **Features Engineering**: Ajouter des variables temporelles
- [ ] **Transfer Learning**: Pré-entraîner sur d'autres datasets
- [ ] **Ensemble Methods**: Combiner plusieurs agents

## Ressources

### Papers de Référence
- [Playing Atari with Deep Reinforcement Learning (Mnih et al., 2013)](https://arxiv.org/abs/1312.5602)
- [Human-level control through deep RL (Nature, 2015)](https://www.nature.com/articles/nature14236)
- [Rainbow: Combining Improvements in DRL (2017)](https://arxiv.org/abs/1710.02298)

### Tutoriels
- [PyTorch DQN Tutorial](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)
- [Spinning Up in Deep RL (OpenAI)](https://spinningup.openai.com/)

## Contribution

Les contributions sont les bienvenues ! Pour contribuer:

1. Fork le projet
2. Créer une branche (`git checkout -b feature/amelioration`)
3. Commit les changements (`git commit -m 'Ajout nouvelle feature'`)
4. Push vers la branche (`git push origin feature/amelioration`)
5. Ouvrir une Pull Request

## Licence

Ce projet est sous licence MIT.

## Contact

Pour toute question:
- Email: votre.email@example.com
- GitHub: [@votre-username](https://github.com/votre-username)

---

**Dernière mise à jour:** Janvier 2025  
**Version:** 1.0.0  
**Recall (Fraude):** 89.15%

**Note**: Ce projet utilise le GPU (CUDA) si disponible pour accélérer l'entraînement.
