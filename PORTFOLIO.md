# CAS D'USAGE 8 — Apprentissage par Renforcement
## Optimiser une décision séquentielle par essais et erreurs dans un environnement simulé

> **Auteur :** TSAGUE EMMANUEL — Data Scientist / Data Analyst  
> **Domaine :** Machine Learning, Optimisation, Prise de décision  
> **Repository GitHub :** `reinforcement-learning-business-case`  
> **Statut :** Portfolio — données simulées  
> **Date :** Juin 2026

---
## 1. TITRE ET RÉSUMÉ EXÉCUTIF

**"Apprentissage par Renforcement pour optimiser la gestion d'énergie d'un bâtiment — Un agent apprend à minimiser les coûts par essais et erreurs"**

> **Apprentissage par Renforcement (Reinforcement Learning — RL) :** branche du Machine Learning où un agent apprend à prendre des décisions en interagissant avec un environnement. À chaque étape, l'agent choisit une action, reçoit une récompense, et met à jour sa stratégie pour maximiser la récompense totale sur le long terme.

Un système de gestion d'énergie d'un bâtiment doit décider à chaque heure : chauffer, maintenir ou refroidir. L'objectif est de minimiser les coûts d'énergie tout en maintenant le confort. Ce projet démontre les concepts fondamentaux du RL avec Q-Learning et les compare à une règle fixe.

**Résultats hypothétiques :** réduction des coûts énergétiques de 18 % vs stratégie fixe, convergence en 5 000 épisodes.

---
## 2. CONTEXTE MÉTIER

> **Décision séquentielle :** suite de décisions interdépendantes où chaque choix influence les options futures. La gestion d'énergie, la gestion de stock, le routage, le trading sont des problèmes de décision séquentielle.

**Pourquoi le RL pour la gestion d'énergie ?**

| Approche | Limite |
|----------|--------|
| Règles fixes | Ne s'adapte pas aux changements (météo, occupation) |
| Optimisation classique | Nécessite un modèle exact du système (souvent inconnu) |
| Apprentissage supervisé | Nécessite des exemples "étiquetés" des bonnes décisions |
| **Apprentissage par renforcement** | Apprend par essais/erreurs sans modèle ni exemples étiquetés |

---
## 3. LES 6 COMPOSANTS DU RL

> **Agent :** l'entité qui prend les décisions (ici : le système de gestion du bâtiment).

> **Environnement :** le monde dans lequel l'agent évolue (ici : le bâtiment avec sa thermodynamique, la météo, l'occupation).

> **État (State) :** la description complète de la situation actuelle (température intérieure, température extérieure, heure de la journée, tarif électrique).

> **Action :** ce que l'agent peut faire à chaque étape (ici : augmenter le chauffage, maintenir, réduire).

> **Récompense (Reward) :** signal numérique qui évalue la qualité de l'action (ici : négatif si énergie dépensée, très négatif si inconfort).

> **Politique (Policy) :** la stratégie de l'agent — pour chaque état, quelle action choisir. C'est ce que l'agent apprend.

---
## 4. PROBLÈME MÉTIER

> "Notre système de chauffage utilise une règle fixe : chauffer si T < 19°C, ne pas chauffer si T ≥ 21°C. Cette règle ignore les tarifs d'électricité (heures creuses/pleines), la météo et l'occupation. Peut-on faire mieux ?"

**Défis :**
1. Modéliser l'environnement bâtiment de façon réaliste
2. Définir les actions et la récompense
3. Implémenter et faire converger un algorithme RL
4. Comparer la politique apprise à la règle fixe
5. Interpréter ce que l'agent a appris

---
## 5. ENVIRONNEMENT SIMULÉ — BÂTIMENT

```python
import numpy as np
import random

class BatimentEnergie:
    """
    Environnement simulé d'un bâtiment pour le RL.
    L'agent contrôle le chauffage/climatisation heure par heure.
    """

    def __init__(self):
        # Paramètres physiques du bâtiment
        self.temp_cible_min = 19.0  # °C — confort minimum
        self.temp_cible_max = 22.0  # °C — confort maximum
        self.inertie        = 0.95  # Coefficient de refroidissement naturel
        self.puissance_kw   = 3.0   # Puissance de la PAC (kW)

        # Paramètres de simulation
        self.reset()

    def reset(self):
        """Réinitialise l'environnement pour un nouvel épisode."""
        self.heure        = 0
        self.temp_int     = random.uniform(17, 24)  # Température initiale aléatoire
        return self._get_state()

    def _get_temp_ext(self, heure):
        """Température extérieure simulée : plus froid la nuit."""
        return 10 + 5 * np.sin((heure - 6) * np.pi / 12)

    def _get_tarif(self, heure):
        """Tarif électrique : heures creuses (nuit) vs heures pleines."""
        return 0.12 if 22 <= heure or heure < 6 else 0.18  # €/kWh

    def _get_state(self):
        """
        État = (temp_intérieure discrétisée, heure, tarif).
        La discrétisation est nécessaire pour le Q-Learning tabulaire.
        """
        temp_disc  = int(np.clip(self.temp_int, 10, 30))   # 10-30°C
        heure_disc = self.heure % 24
        tarif_disc = 1 if self._get_tarif(self.heure) > 0.15 else 0  # 0=creux, 1=plein
        return (temp_disc, heure_disc, tarif_disc)

    def step(self, action):
        """
        Exécute une action et retourne (nouvel_état, récompense, terminé).

        Actions :
            0 = Refroidir / Éteindre
            1 = Maintenir (pas de chauffage)
            2 = Chauffer (puissance maximale)
        """
        temp_ext = self._get_temp_ext(self.heure)
        tarif    = self._get_tarif(self.heure)

        # ─── Dynamique thermique simplifiée ──────────────
        delta_ext     = (temp_ext - self.temp_int) * (1 - self.inertie)
        delta_action  = {0: -0.5, 1: 0.0, 2: 2.0}[action]
        noise         = np.random.normal(0, 0.1)
        self.temp_int = self.temp_int + delta_ext + delta_action + noise

        # ─── Calcul de la récompense ──────────────────────
        cout_energie = 0.0
        if action == 2:
            cout_energie = self.puissance_kw * tarif  # €/heure

        # Pénalité d'inconfort
        inconfort = 0.0
        if self.temp_int < self.temp_cible_min:
            inconfort = 5.0 * (self.temp_cible_min - self.temp_int) ** 2
        elif self.temp_int > self.temp_cible_max:
            inconfort = 5.0 * (self.temp_int - self.temp_cible_max) ** 2

        reward = -(cout_energie + inconfort)  # Négatif : on minimise le coût

        # ─── Avancer le temps ─────────────────────────────
        self.heure += 1
        done = self.heure >= 24 * 7  # 1 semaine = 168 heures

        return self._get_state(), reward, done
```

---
## 6. Q-LEARNING — ALGORITHME

> **Q-Learning :** algorithme de RL qui apprend une table Q(état, action) représentant la valeur attendue de chaque action dans chaque état. La mise à jour est : Q(s,a) ← Q(s,a) + α × [r + γ × max Q(s',a') - Q(s,a)]

> **Table Q (Q-Table) :** tableau qui associe à chaque paire (état, action) une valeur numérique estimant la récompense future espérée. L'agent choisit l'action avec la valeur Q maximale.

> **α (alpha — taux d'apprentissage) :** contrôle la vitesse de mise à jour des valeurs Q. Trop grand = instable, trop petit = apprentissage très lent.

> **γ (gamma — facteur de décompte) :** pondération des récompenses futures. γ = 0,9 signifie que 1€ demain vaut 0,9€ aujourd'hui pour l'agent.

> **ε-greedy (epsilon-greedy) :** stratégie d'exploration/exploitation. Avec une probabilité ε, l'agent choisit une action aléatoire (exploration). Sinon, il choisit la meilleure action connue (exploitation).

> **Exploration vs Exploitation :** dilemme fondamental du RL. Exploiter = utiliser ce qu'on sait déjà faire. Explorer = essayer de nouvelles actions pour peut-être découvrir mieux. Un agent purement exploiteur converge vers un optimum local. Un agent purement explorateur ne converge pas.

```python
import numpy as np

class QLearningAgent:
    """Agent Q-Learning pour la gestion d'énergie."""

    def __init__(self,
                 n_actions=3,
                 alpha=0.1,     # Taux d'apprentissage
                 gamma=0.95,    # Facteur de décompte
                 epsilon=1.0,   # Exploration initiale (100%)
                 epsilon_min=0.01,
                 epsilon_decay=0.995):

        self.n_actions     = n_actions
        self.alpha         = alpha
        self.gamma         = gamma
        self.epsilon       = epsilon
        self.epsilon_min   = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Table Q : dictionnaire {état : array[n_actions]}
        self.q_table = {}

    def _get_q(self, state):
        """Retourne les valeurs Q pour un état, initialise si inconnu."""
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.n_actions)
        return self.q_table[state]

    def choisir_action(self, state):
        """Politique ε-greedy : exploration ou exploitation."""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)  # Exploration aléatoire
        return np.argmax(self._get_q(state))           # Exploitation

    def apprendre(self, state, action, reward, next_state, done):
        """Mise à jour de la table Q par l'équation de Bellman."""
        q_actuel = self._get_q(state)[action]
        q_max_suivant = 0 if done else np.max(self._get_q(next_state))

        # Équation de Bellman
        q_cible = reward + self.gamma * q_max_suivant
        self._get_q(state)[action] += self.alpha * (q_cible - q_actuel)

        # Décroissance de l'exploration
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


def entrainer(n_episodes=5000, verbose=True):
    """Boucle d'entraînement principale."""
    env   = BatimentEnergie()
    agent = QLearningAgent()

    historique = {"rewards": [], "epsilons": []}

    for episode in range(n_episodes):
        state  = env.reset()
        total_reward = 0

        while True:
            action              = agent.choisir_action(state)
            next_state, reward, done = env.step(action)
            agent.apprendre(state, action, reward, next_state, done)
            state        = next_state
            total_reward += reward
            if done:
                break

        historique["rewards"].append(total_reward)
        historique["epsilons"].append(agent.epsilon)

        if verbose and episode % 500 == 0:
            moy = np.mean(historique["rewards"][-100:])
            print(f"Épisode {episode:5d} | "
                  f"Reward moyen (100 derniers) : {moy:8.2f} | "
                  f"ε = {agent.epsilon:.3f}")

    return agent, historique


# Entraînement
agent_rl, historique = entrainer(n_episodes=5000)
```

---
## 7. COMPARAISON AVEC RÈGLES FIXES

```python
def evaluer_politique(politique, n_eval=100):
    """Évalue une politique sur plusieurs épisodes."""
    env = BatimentEnergie()
    rewards_eval = []

    for _ in range(n_eval):
        state = env.reset()
        total_reward = 0
        while True:
            action = politique(state, env)
            state, reward, done = env.step(action)
            total_reward += reward
            if done:
                break
        rewards_eval.append(total_reward)

    return np.mean(rewards_eval), np.std(rewards_eval)

# Règle fixe : chauffer si T < 19, pas si T >= 21
def regle_fixe(state, env):
    temp = state[0]
    if temp < 19:    return 2  # Chauffer
    elif temp > 21:  return 0  # Refroidir
    else:            return 1  # Maintenir

# Politique aléatoire (baseline)
def politique_aleatoire(state, env):
    return np.random.randint(3)

# Politique RL apprise
def politique_rl(state, env):
    return agent_rl.choisir_action_greedy(state)

# Évaluation comparative
r_fixe,    s_fixe    = evaluer_politique(regle_fixe,          100)
r_aleat,   s_aleat   = evaluer_politique(politique_aleatoire, 100)
r_rl,      s_rl      = evaluer_politique(politique_rl,        100)

print("\n=== COMPARAISON DES POLITIQUES ===")
print(f"Règle fixe        : {r_fixe:8.2f} ± {s_fixe:.2f}")
print(f"Aléatoire         : {r_aleat:8.2f} ± {s_aleat:.2f}")
print(f"Q-Learning (RL)   : {r_rl:8.2f} ± {s_rl:.2f}")
gain = (r_rl - r_fixe) / abs(r_fixe) * 100
print(f"\nGain RL vs règle fixe : {gain:+.1f} %")
```

---
## 8. VISUALISATION DE L'APPRENTISSAGE

```python
import matplotlib.pyplot as plt
import pandas as pd

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Apprentissage par Renforcement — Gestion Énergie Bâtiment",
             fontsize=13, fontweight="bold")

# 1. Courbe d'apprentissage (reward cumulé lissé)
rewards = historique["rewards"]
rolling = pd.Series(rewards).rolling(window=200).mean()
axes[0,0].plot(rewards, alpha=0.3, color="steelblue", label="Épisode")
axes[0,0].plot(rolling, color="steelblue", linewidth=2, label="Moyenne 200")
axes[0,0].set_title("Courbe d'apprentissage")
axes[0,0].set_xlabel("Épisode")
axes[0,0].set_ylabel("Reward cumulé")
axes[0,0].legend()

# 2. Décroissance de ε (exploration → exploitation)
axes[0,1].plot(historique["epsilons"], color="orange")
axes[0,1].set_title("Décroissance de ε (exploration → exploitation)")
axes[0,1].set_xlabel("Épisode")
axes[0,1].set_ylabel("ε (probabilité d'exploration)")

# 3. Comparaison des politiques
politiques = ["Aléatoire", "Règle fixe", "Q-Learning"]
rewards_comp = [r_aleat, r_fixe, r_rl]
colors = ["red", "orange", "green"]
axes[1,0].bar(politiques, rewards_comp, color=colors, edgecolor="white")
axes[1,0].set_title("Comparaison des politiques")
axes[1,0].set_ylabel("Reward moyen (100 évaluations)")

# 4. Température et actions sur un épisode type
env_demo = BatimentEnergie()
state    = env_demo.reset()
temps_hist = [env_demo.temp_int]
actions_hist = []
agent_rl.epsilon = 0.0  # Mode évaluation pure

for _ in range(167):
    action = agent_rl.choisir_action(state)
    state, _, done = env_demo.step(action)
    temps_hist.append(env_demo.temp_int)
    actions_hist.append(action)
    if done: break

heures = range(len(temps_hist))
axes[1,1].plot(heures, temps_hist, color="steelblue", label="Température (°C)")
axes[1,1].axhline(y=19, color="red",   linestyle="--", alpha=0.5, label="Min confort")
axes[1,1].axhline(y=22, color="green", linestyle="--", alpha=0.5, label="Max confort")
axes[1,1].fill_between(heures, 19, 22, alpha=0.1, color="green")
axes[1,1].set_title("Température gérée par l'agent (1 semaine)")
axes[1,1].set_xlabel("Heure")
axes[1,1].legend()

plt.tight_layout()
plt.savefig("figures/rl_apprentissage.png", dpi=150, bbox_inches="tight")
```

---
## 9. DÉMARCHE ÉTAPE PAR ÉTAPE

```
ÉTAPE 1 : Définir le problème (agent, environnement, état, action, récompense)
ÉTAPE 2 : Implémenter l'environnement simulé
ÉTAPE 3 : Choisir l'algorithme (Q-Learning pour discret, DQN pour continu)
ÉTAPE 4 : Implémenter l'agent avec ε-greedy
ÉTAPE 5 : Entraîner l'agent (5 000 épisodes)
ÉTAPE 6 : Visualiser la courbe d'apprentissage
ÉTAPE 7 : Évaluer sans exploration (ε=0)
ÉTAPE 8 : Comparer à la règle fixe et à l'aléatoire
ÉTAPE 9 : Interpréter la politique apprise
```

---
## 10. MÉTRIQUES RL

| Métrique | Définition | Valeur simulée |
|----------|------------|----------------|
| **Reward cumulé** | Somme des récompenses sur un épisode | -245 (RL) vs -300 (fixe) |
| **Convergence** | Épisode à partir duquel le reward se stabilise | ~3 000 épisodes |
| **ε final** | Probabilité d'exploration en fin d'entraînement | 0,01 (1 %) |
| **Gain vs règle fixe** | Amélioration du reward | +18 % (simulé) |
| **Taux d'inconfort** | % d'heures hors zone de confort | 4,2 % (RL) vs 8,7 % (fixe) |

---
## 11. RÉSULTATS SIMULÉS

| Politique | Reward moyen | Taux inconfort | Coût énergie |
|-----------|-------------|----------------|-------------|
| Aléatoire | -420 | 22 % | Élevé |
| Règle fixe | -300 | 8,7 % | Moyen |
| Q-Learning | -245 | 4,2 % | Optimisé heures creuses |

**Comportement appris (simulé) :**
- L'agent apprend à préchauffer légèrement avant les heures pleines
- Il exploite les heures creuses (nuit) pour accumuler de la chaleur
- Il maintient la température dans la zone de confort sans gaspillage

---
## 12. VALEUR MÉTIER

| Valeur | Description |
|--------|-------------|
| **Économie** | Réduction de 15-20 % des coûts vs règle fixe (simulé) |
| **Confort** | Meilleure régulation thermique, moins d'inconfort |
| **Adaptabilité** | L'agent s'adapte à des conditions changeantes (météo, occupation) |
| **Automatisation** | Pas d'intervention humaine une fois la politique apprise |

---
## 13. LIMITES

| Limite | Description |
|--------|-------------|
| Environnement simulé | Le vrai bâtiment est plus complexe que le modèle |
| Q-Learning tabulaire | Ne passe pas à l'échelle si l'espace d'états est continu ou grand |
| Instabilité | Le RL peut converger vers des optimums locaux |
| Sécurité | Un agent RL peut explorer des actions dangereuses pendant l'apprentissage |
| Récompense difficile à définir | Le compromis confort/coût est subjectif |

---
## 14. AMÉLIORATIONS

- **DQN (Deep Q-Network) :** Q-Learning avec réseau de neurones pour les espaces d'états continus
- **gymnasium :** bibliothèque Python d'environnements standards pour le RL
- **PPO / A3C :** algorithmes RL plus avancés pour les environnements complexes
- **Multi-agent RL :** plusieurs bâtiments coopèrent pour optimiser le réseau
- **Transfer Learning RL :** transférer la politique apprise sur un bâtiment vers un autre

> **DQN (Deep Q-Network) :** extension du Q-Learning où la table Q est remplacée par un réseau de neurones. Permet de gérer des espaces d'états très grands ou continus. AlphaGo et les agents Atari de DeepMind utilisent cette approche.

---
## 15. ARCHITECTURE GITHUB

```
reinforcement-learning-business-case/
├── README.md
├── requirements.txt
├── notebooks/
│   ├── 01_concepts_rl.ipynb
│   ├── 02_environnement_batiment.ipynb
│   ├── 03_qlearning_agent.ipynb
│   ├── 04_entrainement_evaluation.ipynb
│   └── 05_comparaison_politiques.ipynb
├── src/
│   ├── environment.py         ← BatimentEnergie
│   ├── agent_qlearning.py     ← QLearningAgent
│   └── evaluation.py          ← Fonctions d'évaluation
├── figures/
│   ├── rl_apprentissage.png
│   └── comparaison_politiques.png
└── docs/
    └── concepts_rl_vulgarises.md
```

---
## 22-23. COMPÉTENCES DÉMONTRÉES

| Compétence | Preuve | Valeur | Phrase CV |
|-----------|--------|--------|-----------|
| Q-Learning | Implémentation complète | Optimisation séquentielle | "Q-Learning pour optimisation décision séquentielle" |
| Modélisation | Environnement bâtiment simulé | Abstraction métier → code | "Modélisation d'environnement RL pour cas industriel" |
| Python OOP | Classes Agent et Environnement | Code structuré | "POO Python : agent et environnement RL encapsulés" |
| Évaluation | Comparaison 3 politiques | Rigueur expérimentale | "Comparaison expérimentale de politiques RL" |
| Visualisation | Courbe apprentissage, comparaison | Communication | "Visualisation convergence et performance RL" |

---

*Fin du document — TSAGUE EMMANUEL — CAS 8 — Apprentissage par Renforcement*
---

## Contact & Liens

**TSAGUE EMMANUEL** - Data Scientist

| | |
|---|---|
| Email | [emmatsague@yahoo.fr](mailto:emmatsague@yahoo.fr) |
| GitHub | [github.com/TSAGUE25](https://github.com/TSAGUE25) |
| Formation | Datascientest 2024 |
| Experience | EDF MAD EDVANCE |
| Domaines | Machine Learning - Data Analysis - Energie |

---

> Toutes les donnees de ce depot sont simulees et anonymisees.  
> Aucune donnee reelle ou confidentielle n'est presente.
