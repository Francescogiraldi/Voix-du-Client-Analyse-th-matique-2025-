# 💬 Voix du Client - Analyse des Irritants Clients

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

**Voix du Client** est un outil d'analyse NLP moderne et robuste qui détecte automatiquement les irritants majeurs du parcours client à partir de verbatims textuels. Utilisant des techniques avancées de clustering K-means et de vectorisation TF-IDF, il révèle les sujets les plus problématiques pour prioriser efficacement les actions correctives.

## ✨ Fonctionnalités

### 🔍 Analyse Avancée
- **Pré-traitement linguistique intelligent** avec spaCy (nettoyage, lemmatisation, stopwords)
- **Vectorisation TF-IDF optimisée** avec support des n-grammes (1-3)
- **Clustering K-means robuste** avec validation automatique
- **Score silhouette** pour évaluer la qualité du clustering
- **Détection automatique du nombre optimal de clusters**

### 🎨 Interface Utilisateur
- **Dashboard Streamlit interactif** avec visualisations modernes
- **Nuages de mots dynamiques** par cluster
- **Graphiques interactifs** avec Plotly
- **Export des résultats** en CSV
- **Interface responsive** et accessible

### 🛠️ Outils de Développement
- **API CLI complète** pour l'intégration CI/CD
- **Configuration flexible** via variables d'environnement
- **Logging structuré** avec Loguru
- **Validation des données** avec Pydantic
- **Tests complets** avec pytest

### 🚀 Qualité et Performance
- **Architecture modulaire** et extensible
- **Gestion d'erreurs robuste**
- **Cache intelligent** pour les performances
- **Support multi-format** (CSV, JSON, Excel)
- **Sauvegarde/chargement des modèles**

## 🗂️ Structure du Projet

```text
voix-du-client/
├── src/
│   └── voix_du_client/
│       ├── __init__.py          # Package principal
│       ├── analyzer.py          # Moteur d'analyse NLP
│       ├── config.py            # Configuration avec Pydantic
│       ├── dashboard.py         # Interface Streamlit
│       ├── main.py              # Point d'entrée CLI
│       └── models.py            # Modèles de données
├── tests/
│   ├── test_analyzer.py         # Tests du moteur d'analyse
│   ├── test_models.py           # Tests des modèles
│   └── __init__.py
├── data/
│   └── feedback.csv             # Données d'exemple
├── pyproject.toml               # Configuration moderne du projet
├── requirements.txt             # Dépendances
├── Makefile                     # Tâches de développement
├── .pre-commit-config.yaml      # Hooks de qualité de code
├── .env.example                 # Variables d'environnement
└── README.md                    # Documentation
```

## 🚀 Installation

### Prérequis
- Python 3.9 ou supérieur
- pip ou uv pour la gestion des packages

### Installation Rapide

```bash
# Cloner le repository
git clone https://github.com/votre-org/voix-du-client.git
cd voix-du-client

# Créer un environnement virtuel
python -m venv .venv

# Activer l'environnement (Windows)
.venv\Scripts\activate
# Ou sur Linux/Mac
# source .venv/bin/activate

# Installation complète avec Make
make setup

# Ou installation manuelle
pip install -e ".[dev]"
python -m spacy download fr_core_news_sm
```

### Installation avec uv (Recommandé)

```bash
# Installation avec uv (plus rapide)
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv
source .venv/bin/activate  # ou .venv\Scripts\activate sur Windows
uv pip install -e ".[dev]"
python -m spacy download fr_core_news_sm
```

## 🏃 Utilisation

### Interface Web (Recommandé)

```bash
# Lancer le dashboard
make dashboard
# ou
streamlit run src/voix_du_client/dashboard.py
```

1. 📂 **Chargez vos données** : Uploadez un fichier CSV avec une colonne 'text'
2. ⚙️ **Configurez l'analyse** : Ajustez le nombre de clusters et les paramètres
3. 🚀 **Lancez l'analyse** : Cliquez sur "Lancer l'analyse"
4. 📊 **Explorez les résultats** : Visualisez les clusters et exportez les données

### Ligne de Commande

```bash
# Analyse basique
voix-du-client analyze --csv data/feedback.csv --k 5

# Analyse avec sauvegarde
voix-du-client analyze --csv data/feedback.csv --k 5 --output results.csv --save-model

# Analyse avec paramètres avancés
voix-du-client analyze \
  --csv data/feedback.csv \
  --k 7 \
  --max-features 10000 \
  --ngram-min 1 \
  --ngram-max 3 \
  --output detailed_results.csv
```

### API Python

```python
from voix_du_client import FeedbackAnalyzer, Config
from voix_du_client.models import FeedbackItem, AnalysisConfig

# Configuration
config = Config()
analyzer = FeedbackAnalyzer(config)

# Préparation des données
feedback_items = [
    FeedbackItem(id="1", text="Le service client est très lent"),
    FeedbackItem(id="2", text="Le site web ne fonctionne pas"),
    # ...
]

# Configuration de l'analyse
analysis_config = AnalysisConfig(
    k=5,
    max_features=5000,
    ngram_range=(1, 2),
    random_state=42
)

# Analyse
result = analyzer.analyze(feedback_items, analysis_config)

# Résultats
print(f"Score silhouette: {result.silhouette_score:.3f}")
for cluster in result.clusters:
    print(f"Cluster {cluster.id}: {cluster.size} items")
    print(f"Termes: {', '.join(cluster.top_terms[:5])}")
```

## ⚙️ Configuration

### Variables d'Environnement

Copiez `.env.example` vers `.env` et ajustez les valeurs :

```bash
# Configuration NLP
VOIX_NLP_MODEL=fr_core_news_sm
VOIX_MAX_FEATURES=5000
VOIX_DEFAULT_K=5

# Chemins
VOIX_DATA_DIR=data
VOIX_MODELS_DIR=models
VOIX_LOGS_DIR=logs

# Logging
VOIX_LOG_LEVEL=INFO
```

### Format des Données

Le fichier CSV doit contenir au minimum une colonne `text` :

```csv
id,text
1,"Le temps d'attente au service client est trop long"
2,"Impossible de trouver l'information de livraison"
3,"Le produit ne correspond pas à la description"
```

Colonnes optionnelles supportées :
- `id` : Identifiant unique
- `timestamp` : Date/heure du feedback
- `source` : Source du feedback (email, chat, etc.)
- `priority` : Priorité du feedback

## 🧪 Tests et Qualité

```bash
# Tests complets
make test

# Tests avec couverture
make test-cov

# Vérification de la qualité du code
make lint

# Formatage automatique
make format

# Vérification complète
make dev-check-all
```

## 📊 Métriques et Performance

### Métriques d'Évaluation
- **Score Silhouette** : Mesure la qualité du clustering (-1 à 1)
- **Taux de rétention** : Pourcentage de textes conservés après nettoyage
- **Temps de traitement** : Performance de l'analyse
- **Distribution des clusters** : Équilibre des groupes

### Optimisation des Performances
- **Cache intelligent** : Mise en cache des modèles spaCy
- **Traitement par batch** : Optimisation pour les gros volumes
- **Vectorisation optimisée** : Paramètres TF-IDF ajustés
- **Mémoire** : Gestion efficace des ressources

## 🛠️ Développement

### Contribution

1. **Fork** le projet
2. **Créez** une branche feature (`git checkout -b feature/amazing-feature`)
3. **Committez** vos changements (`git commit -m 'Add amazing feature'`)
4. **Pushez** vers la branche (`git push origin feature/amazing-feature`)
5. **Ouvrez** une Pull Request

### Standards de Code

- **Black** pour le formatage
- **Ruff** pour le linting
- **MyPy** pour le type checking
- **Pytest** pour les tests
- **Pre-commit hooks** pour la qualité

### Architecture

Le projet suit une architecture modulaire :

- `analyzer.py` : Moteur d'analyse principal
- `models.py` : Modèles de données avec Pydantic
- `config.py` : Configuration centralisée
- `dashboard.py` : Interface utilisateur Streamlit
- `main.py` : Point d'entrée CLI

## 🔧 Dépannage

### Problèmes Courants

**Erreur spaCy model not found**
```bash
python -m spacy download fr_core_news_sm
```

**Erreur de mémoire avec de gros datasets**
```bash
# Réduire max_features
voix-du-client analyze --csv data.csv --max-features 1000
```

**Streamlit ne démarre pas**
```bash
pip install --upgrade streamlit
streamlit run src/voix_du_client/dashboard.py
```

### Logs et Debug

Les logs sont disponibles dans le dossier `logs/` :
- `voix_du_client.log` : Logs détaillés
- Console : Logs en temps réel

## 📈 Roadmap

- [ ] **Support multilingue** (anglais, espagnol)
- [ ] **Clustering hiérarchique** (HDBSCAN)
- [ ] **Topic modeling** (LDA, BERTopic)
- [ ] **API REST** pour intégration
- [ ] **Dashboard temps réel** avec WebSocket
- [ ] **Export avancé** (PDF, PowerPoint)
- [ ] **Intégration cloud** (AWS, Azure, GCP)

## 📝 Licence

Ce projet est sous licence **MIT**. Voir le fichier [LICENSE](LICENSE) pour plus de détails.

## 🙏 Remerciements

- [spaCy](https://spacy.io/) pour le traitement du langage naturel
- [Streamlit](https://streamlit.io/) pour l'interface utilisateur
- [scikit-learn](https://scikit-learn.org/) pour les algorithmes de machine learning
- [Plotly](https://plotly.com/) pour les visualisations interactives

---

**Développé avec ❤️ pour améliorer l'expérience client**

