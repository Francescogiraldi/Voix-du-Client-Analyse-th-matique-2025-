"""Streamlit dashboard for Voix du Client."""

import io
import time
from pathlib import Path
from typing import List, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from loguru import logger
from wordcloud import WordCloud

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from voix_du_client.analyzer import FeedbackAnalyzer
from voix_du_client.config import Config
from voix_du_client.models import AnalysisConfig, AnalysisResult, FeedbackItem


class Dashboard:
    """Streamlit dashboard for feedback analysis."""
    
    def __init__(self):
        self.config = Config()
        self.analyzer = FeedbackAnalyzer(self.config)
        self._setup_page()
    
    def _setup_page(self):
        """Configure Streamlit page."""
        st.set_page_config(
            page_title=self.config.page_title,
            page_icon=self.config.page_icon,
            layout=self.config.layout,
            initial_sidebar_state="expanded"
        )
        
        # Custom CSS
        st.markdown("""
        <style>
        .main-header {
            font-size: 2.5rem;
            font-weight: bold;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 2rem;
        }
        .metric-container {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 0.5rem 0;
        }
        .cluster-header {
            background: linear-gradient(90deg, #1f77b4, #ff7f0e);
            color: white;
            padding: 0.5rem;
            border-radius: 0.3rem;
            margin: 1rem 0;
        }
        .stAlert {
            margin-top: 1rem;
        }
        </style>
        """, unsafe_allow_html=True)
    
    def _load_sample_data(self) -> pd.DataFrame:
        """Create sample data if no file is provided."""
        sample_data = {
            'id': range(1, 21),
            'text': [
                "Le temps d'attente au service client est beaucoup trop long",
                "Impossible de trouver l'information de livraison sur le site web",
                "Le produit reçu ne correspond pas à la description",
                "Très déçu de la qualité du service après-vente",
                "Le site web est lent et difficile à naviguer",
                "Problème avec le paiement en ligne, transaction échouée",
                "Livraison en retard sans notification préalable",
                "Interface utilisateur confuse et peu intuitive",
                "Prix trop élevé par rapport à la concurrence",
                "Manque de transparence sur les frais de livraison",
                "Service client peu aimable et non professionnel",
                "Difficile de contacter le support technique",
                "Produit défectueux dès la réception",
                "Processus de retour trop compliqué",
                "Site web souvent en panne ou inaccessible",
                "Informations produit incomplètes ou erronées",
                "Délai de livraison non respecté",
                "Interface mobile peu ergonomique",
                "Facturation incorrecte et difficile à comprendre",
                "Manque de suivi après la vente"
            ]
        }
        return pd.DataFrame(sample_data)
    
    def _validate_dataframe(self, df: pd.DataFrame) -> bool:
        """Validate uploaded dataframe."""
        required_columns = ['text']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            st.error(f"Colonnes manquantes: {', '.join(missing_columns)}")
            st.info("Le fichier CSV doit contenir au minimum une colonne 'text'")
            return False
        
        if df.empty:
            st.error("Le fichier est vide")
            return False
        
        # Check for valid text data
        valid_texts = df['text'].dropna().astype(str).str.strip()
        valid_texts = valid_texts[valid_texts != '']
        
        if len(valid_texts) < 2:
            st.error("Pas assez de textes valides (minimum 2 requis)")
            return False
        
        return True
    
    def _create_feedback_items(self, df: pd.DataFrame) -> List[FeedbackItem]:
        """Convert dataframe to FeedbackItem objects."""
        items = []
        
        for idx, row in df.iterrows():
            # Use provided ID or generate one
            item_id = str(row.get('id', idx))
            text = str(row['text']).strip()
            
            if text:  # Only add non-empty texts
                item = FeedbackItem(id=item_id, text=text)
                items.append(item)
        
        return items
    
    def _display_metrics(self, result: AnalysisResult):
        """Display key metrics."""
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="📊 Nombre de clusters",
                value=result.num_clusters
            )
        
        with col2:
            st.metric(
                label="📝 Total verbatims",
                value=result.total_items
            )
        
        with col3:
            st.metric(
                label="🎯 Score silhouette",
                value=f"{result.silhouette_score:.3f}",
                help="Score de qualité du clustering (-1 à 1, plus élevé = meilleur)"
            )
        
        with col4:
            st.metric(
                label="⏱️ Temps de traitement",
                value=f"{result.processing_time:.1f}s"
            )
    
    def _create_cluster_distribution_chart(self, result: AnalysisResult):
        """Create cluster distribution chart."""
        cluster_data = pd.DataFrame([
            {
                'Cluster': f"Cluster {cluster.id}",
                'Taille': cluster.size,
                'Pourcentage': cluster.percentage,
                'Termes principaux': ', '.join(cluster.top_terms[:3])
            }
            for cluster in result.clusters
        ])
        
        fig = px.bar(
            cluster_data,
            x='Cluster',
            y='Taille',
            color='Pourcentage',
            title="Distribution des clusters",
            hover_data=['Termes principaux'],
            color_continuous_scale='viridis'
        )
        
        fig.update_layout(
            xaxis_title="Clusters",
            yaxis_title="Nombre de verbatims",
            showlegend=False
        )
        
        return fig
    
    def _create_wordcloud(self, texts: List[str], title: str) -> Optional[WordCloud]:
        """Create word cloud from texts."""
        if not texts:
            return None
        
        combined_text = ' '.join(texts)
        if not combined_text.strip():
            return None
        
        try:
            wordcloud = WordCloud(
                width=self.config.wordcloud_width,
                height=self.config.wordcloud_height,
                background_color=self.config.wordcloud_background,
                max_words=100,
                colormap='viridis',
                relative_scaling=0.5,
                min_font_size=10
            ).generate(combined_text)
            
            return wordcloud
        except Exception as e:
            logger.error(f"Error creating wordcloud: {e}")
            return None
    
    def _display_cluster_details(self, result: AnalysisResult, feedback_items: List[FeedbackItem]):
        """Display detailed cluster information."""
        st.subheader("📋 Détails des clusters")
        
        # Create tabs for each cluster
        if result.clusters:
            tab_names = [f"Cluster {cluster.id} ({cluster.size})" for cluster in result.clusters]
            tabs = st.tabs(tab_names)
            
            for tab, cluster in zip(tabs, result.clusters):
                with tab:
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        st.markdown(f"**Taille:** {cluster.size} verbatims ({cluster.percentage:.1f}%)")
                        st.markdown("**Termes principaux:**")
                        for i, term in enumerate(cluster.top_terms[:10], 1):
                            st.write(f"{i}. {term}")
                    
                    with col2:
                        # Get texts for this cluster
                        cluster_texts = [
                            item.clean_text or item.text 
                            for item in feedback_items 
                            if item.cluster == cluster.id
                        ]
                        
                        # Create wordcloud
                        wordcloud = self._create_wordcloud(
                            cluster_texts, 
                            f"Cluster {cluster.id}"
                        )
                        
                        if wordcloud:
                            st.image(wordcloud.to_array(), use_column_width=True)
                        else:
                            st.info("Impossible de générer le nuage de mots")
                    
                    # Representative texts
                    if cluster.representative_texts:
                        st.markdown("**Exemples de verbatims:**")
                        for i, text in enumerate(cluster.representative_texts, 1):
                            st.write(f"{i}. *{text}*")
    
    def _export_results(self, result: AnalysisResult, feedback_items: List[FeedbackItem]):
        """Provide export functionality."""
        st.subheader("📥 Export des résultats")
        
        # Create export dataframe
        export_data = []
        for item in feedback_items:
            cluster_info = next((c for c in result.clusters if c.id == item.cluster), None)
            export_data.append({
                'id': item.id,
                'text': item.text,
                'cluster': item.cluster,
                'cluster_size': cluster_info.size if cluster_info else 0,
                'cluster_percentage': cluster_info.percentage if cluster_info else 0,
                'top_terms': ', '.join(cluster_info.top_terms[:5]) if cluster_info else ''
            })
        
        export_df = pd.DataFrame(export_data)
        
        # Download button
        csv_buffer = io.StringIO()
        export_df.to_csv(csv_buffer, index=False, encoding='utf-8')
        csv_data = csv_buffer.getvalue()
        
        st.download_button(
            label="📊 Télécharger les résultats (CSV)",
            data=csv_data,
            file_name=f"voix_du_client_results_{int(time.time())}.csv",
            mime="text/csv"
        )
    
    def run(self):
        """Run the Streamlit dashboard."""
        # Header
        st.markdown('<h1 class="main-header">💬 Voix du Client - Analyse des Irritants</h1>', 
                   unsafe_allow_html=True)
        st.markdown("---")
        
        # Sidebar configuration
        with st.sidebar:
            st.header("⚙️ Configuration")
            
            # File upload
            uploaded_file = st.file_uploader(
                "📂 Charger un fichier CSV",
                type=['csv'],
                help="Le fichier doit contenir au minimum une colonne 'text'"
            )
            
            # Analysis parameters
            st.subheader("Paramètres d'analyse")
            
            k = st.slider(
                "Nombre de clusters",
                min_value=self.config.min_k,
                max_value=self.config.max_k,
                value=self.config.default_k,
                help="Nombre d'irritants à identifier"
            )
            
            max_features = st.selectbox(
                "Taille du vocabulaire",
                options=[1000, 2000, 5000, 10000],
                index=2,
                help="Nombre maximum de termes dans le vocabulaire TF-IDF"
            )
            
            advanced = st.expander("🔧 Paramètres avancés")
            with advanced:
                ngram_min = st.selectbox("N-gram minimum", [1, 2], index=0)
                ngram_max = st.selectbox("N-gram maximum", [1, 2, 3], index=1)
                random_state = st.number_input("Graine aléatoire", value=42, min_value=0)
        
        # Main content
        try:
            # Load data
            if uploaded_file is not None:
                df = pd.read_csv(uploaded_file)
                st.success(f"✅ Fichier chargé: {len(df)} lignes")
            else:
                df = self._load_sample_data()
                st.info("📋 Données d'exemple chargées (20 verbatims de démonstration)")
            
            # Validate data
            if not self._validate_dataframe(df):
                st.stop()
            
            # Show data preview
            with st.expander("👀 Aperçu des données"):
                st.dataframe(df.head(10))
                st.write(f"Total: {len(df)} verbatims")
            
            # Analysis configuration
            analysis_config = AnalysisConfig(
                k=k,
                max_features=max_features,
                ngram_range=(ngram_min, ngram_max),
                random_state=random_state,
                min_text_length=self.config.min_text_length
            )
            
            # Analysis button
            if st.button("🚀 Lancer l'analyse", type="primary"):
                with st.spinner("Analyse en cours... Cela peut prendre quelques instants."):
                    try:
                        # Convert to feedback items
                        feedback_items = self._create_feedback_items(df)
                        
                        if len(feedback_items) < k:
                            st.error(f"Pas assez de textes valides ({len(feedback_items)}) pour {k} clusters")
                            st.stop()
                        
                        # Perform analysis
                        result = self.analyzer.analyze(feedback_items, analysis_config)
                        
                        # Store results in session state
                        st.session_state['analysis_result'] = result
                        st.session_state['feedback_items'] = feedback_items
                        
                        st.success("✅ Analyse terminée avec succès!")
                        
                    except Exception as e:
                        st.error(f"❌ Erreur lors de l'analyse: {str(e)}")
                        logger.error(f"Analysis error: {e}")
                        st.stop()
            
            # Display results if available
            if 'analysis_result' in st.session_state:
                result = st.session_state['analysis_result']
                feedback_items = st.session_state['feedback_items']
                
                st.markdown("---")
                st.header("📊 Résultats de l'analyse")
                
                # Metrics
                self._display_metrics(result)
                
                # Distribution chart
                st.subheader("📈 Distribution des clusters")
                fig = self._create_cluster_distribution_chart(result)
                st.plotly_chart(fig, use_container_width=True)
                
                # Cluster details
                self._display_cluster_details(result, feedback_items)
                
                # Export
                self._export_results(result, feedback_items)
        
        except Exception as e:
            st.error(f"❌ Erreur inattendue: {str(e)}")
            logger.error(f"Dashboard error: {e}")


def main():
    """Main entry point for the dashboard."""
    dashboard = Dashboard()
    dashboard.run()


if __name__ == "__main__":
    main()