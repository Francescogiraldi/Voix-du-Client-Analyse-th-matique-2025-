"""Core feedback analysis functionality."""

import re
import string
import time
from pathlib import Path
from typing import List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import spacy
from loguru import logger
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from spacy.lang.fr.stop_words import STOP_WORDS as FRENCH_STOP_WORDS

from .config import Config
from .models import AnalysisConfig, AnalysisResult, ClusterInfo, FeedbackItem, ProcessingStats


class TextProcessor:
    """Handles text preprocessing and cleaning."""
    
    def __init__(self, config: Config):
        self.config = config
        self._nlp = None
        self._stopwords = self._build_stopwords()
    
    @property
    def nlp(self) -> spacy.language.Language:
        """Lazy loading of spaCy model."""
        if self._nlp is None:
            try:
                self._nlp = spacy.load(self.config.nlp_model)
                logger.info(f"Loaded spaCy model: {self.config.nlp_model}")
            except OSError as e:
                logger.error(f"Failed to load spaCy model {self.config.nlp_model}: {e}")
                logger.info("Attempting to download the model...")
                import subprocess
                subprocess.run(["python", "-m", "spacy", "download", self.config.nlp_model])
                self._nlp = spacy.load(self.config.nlp_model)
        return self._nlp
    
    def _build_stopwords(self) -> set:
        """Build comprehensive stopwords set."""
        stopwords = set(FRENCH_STOP_WORDS)
        stopwords.update(self.config.extra_stopwords)
        # Add common noise words
        stopwords.update({
            "c", "d", "j", "l", "m", "n", "s", "t", "y", "qu", "si", "ca", "ça",
            "alors", "donc", "enfin", "puis", "aussi", "ainsi", "car", "or",
            "mais", "et", "ou", "ni", "comme", "quand", "où", "comment", "pourquoi"
        })
        return stopwords
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize a single text."""
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower().strip()
        
        # Remove URLs
        text = re.sub(r"https?://\S+|www\.\S+", " ", text)
        
        # Remove email addresses
        text = re.sub(r"\S+@\S+", " ", text)
        
        # Remove phone numbers
        text = re.sub(r"\b\d{2}[\s.-]?\d{2}[\s.-]?\d{2}[\s.-]?\d{2}[\s.-]?\d{2}\b", " ", text)
        
        # Remove standalone numbers but keep numbers within words
        text = re.sub(r"\b\d+\b", " ", text)
        
        # Remove excessive punctuation
        text = re.sub(r"[!]{2,}", "!", text)
        text = re.sub(r"[?]{2,}", "?", text)
        text = re.sub(r"[.]{2,}", ".", text)
        
        # Remove punctuation except sentence endings
        text = re.sub(r"[^\w\s.!?]", " ", text)
        
        # Process with spaCy
        try:
            doc = self.nlp(text)
            tokens = []
            
            for token in doc:
                # Skip if token is stop word, punctuation, space, or too short
                if (token.is_stop or token.is_punct or token.is_space or 
                    len(token.text) < 2 or token.lemma_ in self._stopwords):
                    continue
                
                # Use lemma and clean it
                lemma = token.lemma_.lower().strip()
                if lemma and len(lemma) >= 2 and lemma.isalpha():
                    tokens.append(lemma)
            
            result = " ".join(tokens)
            return result if len(result) >= self.config.min_text_length else ""
            
        except Exception as e:
            logger.warning(f"Error processing text with spaCy: {e}")
            # Fallback to simple processing
            words = text.split()
            clean_words = [w for w in words if len(w) >= 2 and w.isalpha() and w not in self._stopwords]
            return " ".join(clean_words)
    
    def process_texts(self, texts: List[str]) -> Tuple[List[str], ProcessingStats]:
        """Process multiple texts and return statistics."""
        start_time = time.time()
        
        original_count = len(texts)
        original_lengths = [len(text) for text in texts if isinstance(text, str)]
        avg_length_before = np.mean(original_lengths) if original_lengths else 0
        
        logger.info(f"Processing {original_count} texts...")
        
        processed_texts = []
        for i, text in enumerate(texts):
            if i % 1000 == 0 and i > 0:
                logger.info(f"Processed {i}/{original_count} texts")
            
            clean_text = self.clean_text(text)
            if clean_text:  # Only keep non-empty texts
                processed_texts.append(clean_text)
        
        processed_count = len(processed_texts)
        filtered_count = original_count - processed_count
        processed_lengths = [len(text) for text in processed_texts]
        avg_length_after = np.mean(processed_lengths) if processed_lengths else 0
        processing_time = time.time() - start_time
        
        stats = ProcessingStats(
            original_count=original_count,
            processed_count=processed_count,
            filtered_count=filtered_count,
            avg_length_before=avg_length_before,
            avg_length_after=avg_length_after,
            processing_time=processing_time
        )
        
        logger.info(f"Text processing completed: {stats.retention_rate:.1f}% retention rate")
        return processed_texts, stats


class FeedbackAnalyzer:
    """Main analyzer class for customer feedback."""
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.text_processor = TextProcessor(self.config)
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.kmeans: Optional[KMeans] = None
        
        logger.info("FeedbackAnalyzer initialized")
    
    def _create_vectorizer(self, analysis_config: AnalysisConfig) -> TfidfVectorizer:
        """Create TF-IDF vectorizer with given configuration."""
        return TfidfVectorizer(
            max_features=analysis_config.max_features,
            ngram_range=analysis_config.ngram_range,
            min_df=2,  # Ignore terms that appear in less than 2 documents
            max_df=0.95,  # Ignore terms that appear in more than 95% of documents
            strip_accents='unicode',
            lowercase=True,
            token_pattern=r'\b[a-zA-ZÀ-ÿ]{2,}\b'  # French characters support
        )
    
    def _create_kmeans(self, analysis_config: AnalysisConfig) -> KMeans:
        """Create K-means clusterer with given configuration."""
        return KMeans(
            n_clusters=analysis_config.k,
            init='k-means++',
            n_init=10,
            max_iter=300,
            random_state=analysis_config.random_state,
            algorithm='lloyd'
        )
    
    def _extract_top_terms(self, cluster_id: int, n_terms: int = 10) -> List[str]:
        """Extract top terms for a specific cluster."""
        if self.vectorizer is None or self.kmeans is None:
            return []
        
        feature_names = self.vectorizer.get_feature_names_out()
        cluster_center = self.kmeans.cluster_centers_[cluster_id]
        top_indices = cluster_center.argsort()[::-1][:n_terms]
        
        return [feature_names[i] for i in top_indices]
    
    def _get_representative_texts(self, feedback_items: List[FeedbackItem], 
                                cluster_id: int, n_texts: int = 3) -> List[str]:
        """Get representative texts for a cluster."""
        cluster_texts = [
            item.text for item in feedback_items 
            if item.cluster == cluster_id
        ]
        
        # Return shortest texts as they're often more concise
        cluster_texts.sort(key=len)
        return cluster_texts[:n_texts]
    
    def analyze(self, feedback_items: List[FeedbackItem], 
               analysis_config: Optional[AnalysisConfig] = None) -> AnalysisResult:
        """Perform complete feedback analysis."""
        start_time = time.time()
        
        if not feedback_items:
            raise ValueError("No feedback items provided")
        
        # Use default config if none provided
        if analysis_config is None:
            analysis_config = AnalysisConfig(
                k=self.config.default_k,
                max_features=self.config.max_features,
                ngram_range=(self.config.ngram_range_min, self.config.ngram_range_max),
                random_state=self.config.random_state,
                min_text_length=self.config.min_text_length
            )
        
        logger.info(f"Starting analysis with {len(feedback_items)} items, k={analysis_config.k}")
        
        # Extract and process texts
        texts = [item.text for item in feedback_items]
        processed_texts, processing_stats = self.text_processor.process_texts(texts)
        
        if len(processed_texts) < analysis_config.k:
            raise ValueError(f"Not enough valid texts ({len(processed_texts)}) for {analysis_config.k} clusters")
        
        # Update feedback items with clean text
        valid_items = []
        processed_idx = 0
        for item in feedback_items:
            clean_text = self.text_processor.clean_text(item.text)
            if clean_text:
                item.clean_text = clean_text
                valid_items.append(item)
                processed_idx += 1
        
        # Vectorization
        logger.info("Vectorizing texts...")
        self.vectorizer = self._create_vectorizer(analysis_config)
        X = self.vectorizer.fit_transform(processed_texts)
        
        # Clustering
        logger.info("Performing clustering...")
        self.kmeans = self._create_kmeans(analysis_config)
        cluster_labels = self.kmeans.fit_predict(X)
        
        # Assign clusters to feedback items
        for i, item in enumerate(valid_items):
            item.cluster = int(cluster_labels[i])
        
        # Calculate silhouette score
        silhouette = silhouette_score(X, cluster_labels) if len(set(cluster_labels)) > 1 else 0.0
        
        # Create cluster information
        clusters = []
        total_items = len(valid_items)
        
        for cluster_id in range(analysis_config.k):
            cluster_items = [item for item in valid_items if item.cluster == cluster_id]
            cluster_size = len(cluster_items)
            
            if cluster_size > 0:
                cluster_info = ClusterInfo(
                    id=cluster_id,
                    size=cluster_size,
                    percentage=(cluster_size / total_items) * 100,
                    top_terms=self._extract_top_terms(cluster_id, self.config.top_terms_display),
                    representative_texts=self._get_representative_texts(valid_items, cluster_id)
                )
                clusters.append(cluster_info)
        
        # Sort clusters by size (largest first)
        clusters.sort(key=lambda c: c.size, reverse=True)
        
        processing_time = time.time() - start_time
        
        result = AnalysisResult(
            clusters=clusters,
            total_items=total_items,
            silhouette_score=silhouette,
            processing_time=processing_time,
            config_snapshot=analysis_config.dict()
        )
        
        logger.info(f"Analysis completed in {processing_time:.2f}s, silhouette score: {silhouette:.3f}")
        return result
    
    def save_model(self, filepath: Path) -> None:
        """Save the trained model to disk."""
        if self.vectorizer is None or self.kmeans is None:
            raise ValueError("No trained model to save")
        
        model_data = {
            'vectorizer': self.vectorizer,
            'kmeans': self.kmeans,
            'config': self.config.dict()
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: Path) -> None:
        """Load a trained model from disk."""
        model_data = joblib.load(filepath)
        
        self.vectorizer = model_data['vectorizer']
        self.kmeans = model_data['kmeans']
        
        logger.info(f"Model loaded from {filepath}")
    
    def predict_cluster(self, text: str) -> int:
        """Predict cluster for a new text."""
        if self.vectorizer is None or self.kmeans is None:
            raise ValueError("No trained model available")
        
        clean_text = self.text_processor.clean_text(text)
        if not clean_text:
            raise ValueError("Text is empty after cleaning")
        
        X = self.vectorizer.transform([clean_text])
        cluster = self.kmeans.predict(X)[0]
        
        return int(cluster)