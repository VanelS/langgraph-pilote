# Agent LangGraph Modulaire

Ce projet implémente un agent conversationnel modulaire basé sur LangGraph qui peut répondre à des questions en utilisant différents outils comme la météo et une calculatrice.

## Structure du Projet

Le projet a été organisé de manière modulaire pour une meilleure maintenabilité:

```
src/
├── main.py                  # Point d'entrée principal
├── modules/                 # Package contenant les modules
│   ├── __init__.py          # Exports du package
│   ├── state.py             # Définition de l'état de l'agent
│   ├── tools.py             # Outils disponibles (météo, calculatrice)
│   ├── reasoning.py         # Fonctions de raisonnement
│   ├── graph.py             # Construction du graphe d'agent
│   └── visualization.py     # Visualisation du graphe
└── 03_test.py               # Version monolithique d'origine
```

## Fonctionnalités

L'agent peut:
- Analyser des questions en langage naturel
- Choisir l'outil approprié pour répondre
- Rechercher la météo d'une ville
- Calculer des expressions mathématiques
- Répondre directement à des questions générales
- Générer une visualisation du graphe d'agent

## Utilisation

1. Assurez-vous d'avoir les dépendances installées
2. Exécutez le fichier principal:

```bash
python src/main.py
```

Le programme teste automatiquement deux questions (météo et calcul) et génère une visualisation du graphe d'agent.

## Organisation des modules

- **state.py**: Définit la structure de données qui représente l'état de l'agent
- **tools.py**: Implémente les outils que l'agent peut utiliser
- **reasoning.py**: Contient les fonctions de raisonnement et le routeur
- **graph.py**: Assemble le graphe d'agent avec ses nœuds et arêtes
- **visualization.py**: Fournit des fonctions pour visualiser le graphe

## Ajouter de nouveaux outils

Pour ajouter un nouvel outil:
1. Ajoutez une fonction avec décorateur `@tool` dans `tools.py`
2. Ajoutez une fonction de nœud correspondante dans `reasoning.py`
3. Mettez à jour le routeur dans `reasoning.py`
4. Ajoutez le nœud et les arêtes dans `graph.py`
