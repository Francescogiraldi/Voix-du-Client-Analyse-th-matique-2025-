#!/usr/bin/env python3
"""
Voix du Client – Analyse thématique (2025)
=========================================
Ce script réalise une .

Principales étapes
------------------
1. **Chargement** des verbatims depuis un `feedback.csv` (ou fichier uploadé en UI).
2. **Pré‑traitement** : nettoyage, normalisation, lemmatisation (spaCy `fr_core_news_sm`).
3. **Vectorisation** TF‑IDF (scikit‑learn) avec n‑grammes (1–2).
4. **Clustering** K‑means (par défaut *k = 5* pour extraire les 5 irritants majeurs).
5. **Synthèse** : termes les plus représentatifs par cluster + volume de verbatims.
6. **Dashboard Streamlit** pour une exploration interactive.

Structure du projet
-------------------
project/
├── voix_du_client.py          ← *ce fichier*
├── feedback.csv               ← source verbatims (UTF‑8, colonnes `id,text`)
└── models/                    ← modèles/artefacts générés (vectoriseur, k‑means)

Installation rapide
-------------------
```bash
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install pandas scikit-learn spacy streamlit wordcloud python-dotenv
python -m spacy download fr_core_news_sm
```

Exécution
---------
```bash
streamlit run voix_du_client.py
```
"""

from __future__ import annotations

import re
import string
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import spacy
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from wordcloud import WordCloud

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

NLP_MODEL = "fr_core_news_sm"  # modèle spaCy français (léger)
STOPWORDS_EXTRA = {"nous", "vous", "ils", "elles"}  # stopwords spécifiques
DEFAULT_K = 5  # nombre de clusters par défaut
MAX_FEATURES = 5000  # vocabulaire TF‑IDF
SEED = 42  # reproductibilité

# -----------------------------------------------------------------------------
# Pré‑traitement texte
# -----------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def load_spacy() -> spacy.language.Language:
    """Charge (et met en cache) le modèle spaCy français."""
    return spacy.load(NLP_MODEL)


nlp = load_spacy()


def nettoyer_texte(texte: str) -> str:
    """Nettoie et normalise un verbatim."""
    texte = texte.lower()
    texte = re.sub(r"https?://\S+", " ", texte)  # liens
    texte = re.sub(r"\d+", " ", texte)  # chiffres
    texte = texte.translate(str.maketrans("", "", string.punctuation))
    doc = nlp(texte)
    tokens = [t.lemma_ for t in doc if not (t.is_stop or t.is_punct or t.lemma_ in STOPWORDS_EXTRA)]
    return " ".join(tokens)


# -----------------------------------------------------------------------------
# Pipeline analyse
# -----------------------------------------------------------------------------

def vectoriser_textes(textes: List[str]) -> Tuple[TfidfVectorizer, "csr_matrix"]:
    """Transforme les textes en matrice TF‑IDF."""
    vectorizer = TfidfVectorizer(max_features=MAX_FEATURES, ngram_range=(1, 2))
    X = vectorizer.fit_transform(textes)
    return vectorizer, X


def clusteriser(X, k: int = DEFAULT_K) -> KMeans:
    """Applique K‑means et retourne le modèle."""
    km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=SEED)
    km.fit(X)
    return km


def termes_topics(vectorizer: TfidfVectorizer, km: KMeans, top_n: int = 10) -> List[List[str]]:
    """Retourne les termes les plus importants par cluster."""
    termes = []
    vocab = vectorizer.get_feature_names_out()
    for centre in km.cluster_centers_:
        indices = centre.argsort()[::-1][:top_n]
        termes.append([vocab[i] for i in indices])
    return termes


# -----------------------------------------------------------------------------
# Dashboard Streamlit
# -----------------------------------------------------------------------------

st.set_page_config(page_title="Voix du Client", page_icon="💬", layout="wide")
st.title("💬 Analyse Voix du Client – Clustering d'irritants")
st.caption("Téléchargez vos verbatims clients (CSV) pour détecter les irritants majeurs.")

fichier_upload = st.file_uploader("📂 Charger un fichier CSV", type=["csv"])

if fichier_upload:
    df = pd.read_csv(fichier_upload)
else:
    chemin_defaut = Path("feedback.csv")
    if chemin_defaut.exists():
        df = pd.read_csv(chemin_defaut)
        st.info("Exemple `feedback.csv` chargé (démo).")
    else:
        st.warning("Aucun fichier trouvé. Veuillez uploader un CSV avec les colonnes `id,text`.")
        st.stop()

if "text" not in df.columns:
    st.error("La colonne `text` est obligatoire dans le CSV.")
    st.stop()

# Nettoyage + vectorisation
with st.spinner("Nettoyage et vectorisation des verbatims…"):
    df["clean"] = df["text"].fillna("").apply(nettoyer_texte)
    vectorizer, X = vectoriser_textes(df["clean"].tolist())

# Choix du nombre de clusters
k = st.slider("Nombre de clusters (irritants)", 2, 10, DEFAULT_K)

if st.button("🚀 Lancer l'analyse"):

    with st.spinner("Clustering en cours…"):
        km = clusteriser(X, k)
        labels = km.labels_
        df["cluster"] = labels
        score = silhouette_score(X, labels)
        termes = termes_topics(vectorizer, km)

    st.success(f"Clustering terminé ! Score silhouette = {score:.3f}")

    # Tableau résumé
    st.subheader("📊 Résumé des clusters")
    resume = (
        df.groupby("cluster")
        .size()
        .rename("nb_verbatims")
        .to_frame()
        .assign(termes=lambda d: d.index.map(lambda i: ", ".join(termes[i][:6])))
    )
    st.dataframe(resume)

    # Wordclouds par cluster
    st.subheader("☁️ Nuages de mots par cluster")
    for i in range(k):
        texts_cluster = " ".join(df.loc[df["cluster"] == i, "clean"])
        wc = WordCloud(width=800, height=400, background_color="white").generate(texts_cluster)
        st.markdown(f"##### Cluster {i} – Principaux termes")
        st.image(wc.to_array(), use_column_width=True)

# -----------------------------------------------------------------------------
# Mode CLI (pour exécutions rapides)
# -----------------------------------------------------------------------------
if __name__ == "__main__" and not st.runtime.exists():
    import argparse

    parser = argparse.ArgumentParser(description="Analyse Voix du Client (CLI)")
    parser.add_argument("--csv", type=str, default="feedback.csv", help="Chemin vers le CSV de verbatims")
    parser.add_argument("--k", type=int, default=DEFAULT_K, help="Nombre de clusters")
    args = parser.parse_args()

    df_cli = pd.read_csv(args.csv)
    df_cli["clean"] = df_cli["text"].fillna("").apply(nettoyer_texte)
    _, X_cli = vectoriser_textes(df_cli["clean"].tolist())
    km_cli = clusteriser(X_cli, args.k)
    termes_cli = termes_topics(_, km_cli)

    print("\n=== Résumé des clusters ===")
    for i, mots in enumerate(termes_cli):
        n = (km_cli.labels_ == i).sum()
        print(f"Cluster {i} | {n} verbatims | mots clés : {', '.join(mots[:8])}")
