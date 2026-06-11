# Apprentissage par Renforcement — Gestion d'Énergie

> **Un agent Q-Learning apprend à minimiser les coûts énergétiques par essais et erreurs**

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Domaine](https://img.shields.io/badge/Domaine-Machine%20Learning-green)
![Statut](https://img.shields.io/badge/Statut-Portfolio-orange)
![Données](https://img.shields.io/badge/Données-Simulées%2FAnonymisées-lightgrey)

---

## Contexte métier

L'apprentissage par renforcement (RL) permet à un agent d'apprendre une stratégie optimale par interaction avec un environnement, sans connaissance a priori du modèle. Applications : gestion d'énergie, robotique, trading, jeux.

---

## Problème traité

Un système de gestion du chauffage/climatisation d'un bâtiment doit décider chaque heure l'action optimale (chauffer/maintenir/refroidir) pour minimiser les coûts en heures creuses/pleines tout en maintenant le confort.

---

## Solution proposée

Environnement bâtiment simulé (thermodynamique + tarifs), agent Q-Learning avec stratégie ε-greedy et décroissance de l'exploration, convergence en ~3000 épisodes, comparaison vs règle fixe.

---

## Technologies utilisées

| Outil | Usage |
|-------|-------|
| Python 3.10+ | Langage principal |
| pandas / numpy | Manipulation des données |
| scikit-learn | Machine Learning & preprocessing |
| matplotlib / seaborn | Visualisation |
| Jupyter Notebook | Exploration interactive |

> Voir `requirements.txt` pour la liste complète.

---

## Structure du projet

```
reinforcement-learning-business-case/
├── README.md              ← Ce fichier
├── PORTFOLIO.md           ← Documentation complète du cas d'usage
├── .gitignore
├── requirements.txt
├── notebooks/             ← Jupyter Notebooks d'exploration
├── src/                   ← Code Python modulaire
├── data_sample/           ← Données simulées (anonymisées)
├── figures/               ← Graphiques et visualisations
├── reports/               ← Rapports et synthèses
└── docs/                  ← Documentation complémentaire
```

---

## Installation

```bash
# 1. Cloner le dépôt
git clone https://github.com/TSAGUE25/reinforcement-learning-business-case.git
cd reinforcement-learning-business-case

# 2. Créer un environnement virtuel
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate    # Windows

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Lancer Jupyter
jupyter notebook
```

---

## Métriques clés (données simulées)

```
Reward cumulé, taux d'inconfort, KS statistic convergence, gain vs règle fixe
```

---

## Valeur métier

Réduction de 18 % des coûts énergétiques vs stratégie fixe (simulé).

---

## Limites

Q-Learning tabulaire : ne passe pas à l'échelle (espace d'états continu). Environnement simplifié.

---

## Prochaines améliorations

DQN (Deep Q-Network) pour espace d'états continu. Multi-agent pour bâtiments multiples.

---

## Avertissement — Confidentialité

> **Toutes les données utilisées dans ce projet sont simulées, synthétiques ou anonymisées.**
> Aucune donnée réelle, confidentielle ou propriétaire n'est présente dans ce dépôt.
> Ce projet est un cas d'usage pédagogique à destination du portfolio professionnel d'Emmanuel TSAGUE.

---

## Contributors

**TSAGUE EMMANUEL** - Data Scientist  
Specialise en Machine Learning, Data Analysis et systemes decisionnels.  
Formation Datascientest 2024 | EDF MAD EDVANCE  
Email : [emmatsague@yahoo.fr](mailto:emmatsague@yahoo.fr)  
GitHub : [github.com/TSAGUE25](https://github.com/TSAGUE25)

