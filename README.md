# Voix du Client – Analyse des Irritants

**Voix du Client** est un outil d’analyse NLP qui détecte et résume les irritants
majeurs du parcours client à partir de verbatims textuels. Basé sur le clustering
K‑means et la vectorisation TF‑IDF, il révèle automatiquement les 5 (ou plus)
sujets les plus problématiques afin de prioriser les actions correctives.

---

## ✨ Fonctionnalités

* **Pré‑traitement linguistique** (nettoyage, stopwords, lemmatisation spaCy)
* **Vectorisation TF‑IDF** avec n‑grammes 1–2
* **Clustering** K‑means (nombre de clusters réglable)
* **Score silhouette** pour évaluer la cohérence thématique
* **Dashboard** interactif Streamlit (tableaux + nuages de mots)
* **Mode CLI** pour intégration CI/CD ou analyses rapides

---

## 🗂️ Structure du dépôt

```text
project/
├── voix_du_client.py   # script principal
├── feedback.csv        # verbatims (exemple)
└── models/             # artefacts générés (vectoriseur, k‑means)
```

Le **`feedback.csv`** minimal :

| id | text                                                     |
| -- | -------------------------------------------------------- |
| 1  | "Le temps d’attente au service client est trop long."    |
| 2  | "Impossible de trouver l’info de livraison sur le site." |

---

## 🚀 Installation rapide

```bash
git clone https://github.com/votre-org/voix-du-client.git
cd voix-du-client
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install pandas scikit-learn spacy streamlit wordcloud python-dotenv
python -m spacy download fr_core_news_sm
```

---

## 🏃 Lancement

### Interface web

```bash
streamlit run voix_du_client.py
```

1. Chargez votre `feedback.csv` via l’UI.
2. Sélectionnez le nombre de clusters (*slider*).
3. Cliquez sur **Lancer l’analyse**.

### Ligne de commande

```bash
python voix_du_client.py --csv feedback.csv --k 5
```

---

## ⚙️ Variables & Paramètres

| Paramètre      | Description                                 | Défaut         |
| -------------- | ------------------------------------------- | -------------- |
| `--k`          | Nombre de clusters (irritants)              | `5`            |
| `--csv`        | Chemin vers le fichier de verbatims         | `feedback.csv` |
| `MAX_FEATURES` | Taille du vocabulaire TF‑IDF (dans le code) | `5000`         |

---

## 🛠️ Personnalisation

* **Stopwords** : adapter `STOPWORDS_EXTRA` dans le script.
* **Vectorisation** : ajuster le `ngram_range` ou la limite `MAX_FEATURES`.
* **Modèles avancés** : remplacer K‑means par HDBSCAN ou LDA (topic modelling).

---

## 🤝 Contribuer

Les *issues* et *pull requests* sont les bienvenues. Merci !

---

## 📝 Licence

Diffusé sous licence **MIT**. Consultez `LICENSE` pour les détails.

