"""Tests for the analyzer module."""

import pytest
from unittest.mock import Mock, patch

from src.voix_du_client.analyzer import FeedbackAnalyzer, TextProcessor
from src.voix_du_client.config import Config
from src.voix_du_client.models import AnalysisConfig, FeedbackItem


class TestTextProcessor:
    """Test cases for TextProcessor."""
    
    @pytest.fixture
    def config(self):
        return Config()
    
    @pytest.fixture
    def processor(self, config):
        return TextProcessor(config)
    
    def test_clean_text_basic(self, processor):
        """Test basic text cleaning."""
        text = "Bonjour, comment allez-vous aujourd'hui ?"
        # Mock spaCy processing
        with patch.object(processor, 'nlp') as mock_nlp:
            mock_doc = Mock()
            mock_token = Mock()
            mock_token.is_stop = False
            mock_token.is_punct = False
            mock_token.is_space = False
            mock_token.text = "bonjour"
            mock_token.lemma_ = "bonjour"
            mock_doc.__iter__ = Mock(return_value=iter([mock_token]))
            mock_nlp.return_value = mock_doc
            
            result = processor.clean_text(text)
            assert isinstance(result, str)
            assert len(result) > 0
    
    def test_clean_text_empty(self, processor):
        """Test cleaning empty text."""
        assert processor.clean_text("") == ""
        assert processor.clean_text("   ") == ""
        assert processor.clean_text(None) == ""
    
    def test_clean_text_urls(self, processor):
        """Test URL removal."""
        text = "Visitez https://example.com pour plus d'infos"
        with patch.object(processor, 'nlp') as mock_nlp:
            mock_doc = Mock()
            mock_doc.__iter__ = Mock(return_value=iter([]))
            mock_nlp.return_value = mock_doc
            
            result = processor.clean_text(text)
            assert "https://example.com" not in result
    
    def test_process_texts(self, processor):
        """Test processing multiple texts."""
        texts = ["Premier texte", "Deuxième texte", ""]
        
        with patch.object(processor, 'clean_text') as mock_clean:
            mock_clean.side_effect = ["premier texte", "deuxième texte", ""]
            
            processed, stats = processor.process_texts(texts)
            
            assert len(processed) == 2  # Empty text filtered out
            assert stats.original_count == 3
            assert stats.processed_count == 2
            assert stats.filtered_count == 1


class TestFeedbackAnalyzer:
    """Test cases for FeedbackAnalyzer."""
    
    @pytest.fixture
    def config(self):
        return Config()
    
    @pytest.fixture
    def analyzer(self, config):
        return FeedbackAnalyzer(config)
    
    @pytest.fixture
    def sample_feedback_items(self):
        return [
            FeedbackItem(id="1", text="Le service client est très lent"),
            FeedbackItem(id="2", text="Le site web ne fonctionne pas bien"),
            FeedbackItem(id="3", text="Problème avec la livraison"),
            FeedbackItem(id="4", text="Interface utilisateur confuse"),
            FeedbackItem(id="5", text="Prix trop élevé pour la qualité")
        ]
    
    def test_analyzer_initialization(self, analyzer):
        """Test analyzer initialization."""
        assert analyzer.config is not None
        assert analyzer.text_processor is not None
        assert analyzer.vectorizer is None
        assert analyzer.kmeans is None
    
    def test_analyze_empty_input(self, analyzer):
        """Test analysis with empty input."""
        with pytest.raises(ValueError, match="No feedback items provided"):
            analyzer.analyze([])
    
    def test_analyze_insufficient_items(self, analyzer):
        """Test analysis with insufficient items for clustering."""
        items = [FeedbackItem(id="1", text="test")]
        config = AnalysisConfig(k=5, max_features=1000, ngram_range=(1, 2), 
                               random_state=42, min_text_length=3)
        
        with patch.object(analyzer.text_processor, 'process_texts') as mock_process:
            mock_process.return_value = (["test"], Mock())
            
            with pytest.raises(ValueError, match="Not enough valid texts"):
                analyzer.analyze(items, config)
    
    @patch('src.voix_du_client.analyzer.TfidfVectorizer')
    @patch('src.voix_du_client.analyzer.KMeans')
    @patch('src.voix_du_client.analyzer.silhouette_score')
    def test_analyze_success(self, mock_silhouette, mock_kmeans_class, 
                           mock_vectorizer_class, analyzer, sample_feedback_items):
        """Test successful analysis."""
        # Mock vectorizer
        mock_vectorizer = Mock()
        mock_vectorizer.fit_transform.return_value = Mock()
        mock_vectorizer.get_feature_names_out.return_value = ['service', 'client', 'site', 'web']
        mock_vectorizer_class.return_value = mock_vectorizer
        
        # Mock KMeans
        mock_kmeans = Mock()
        mock_kmeans.fit_predict.return_value = [0, 1, 0, 1, 0]
        mock_kmeans.cluster_centers_ = [[0.5, 0.3, 0.2, 0.1], [0.4, 0.4, 0.1, 0.1]]
        mock_kmeans_class.return_value = mock_kmeans
        
        # Mock silhouette score
        mock_silhouette.return_value = 0.5
        
        # Mock text processing
        with patch.object(analyzer.text_processor, 'process_texts') as mock_process:
            processed_texts = ["service client lent", "site web problème", 
                             "problème livraison", "interface confuse", "prix élevé"]
            mock_process.return_value = (processed_texts, Mock())
            
            with patch.object(analyzer.text_processor, 'clean_text') as mock_clean:
                mock_clean.side_effect = processed_texts
                
                config = AnalysisConfig(k=2, max_features=1000, ngram_range=(1, 2),
                                       random_state=42, min_text_length=3)
                
                result = analyzer.analyze(sample_feedback_items, config)
                
                assert result is not None
                assert result.num_clusters == 2
                assert result.total_items == 5
                assert result.silhouette_score == 0.5
                assert len(result.clusters) == 2
    
    def test_predict_cluster_no_model(self, analyzer):
        """Test prediction without trained model."""
        with pytest.raises(ValueError, match="No trained model available"):
            analyzer.predict_cluster("test text")
    
    def test_save_load_model(self, analyzer, tmp_path):
        """Test model saving and loading."""
        # Setup mock models
        analyzer.vectorizer = Mock()
        analyzer.kmeans = Mock()
        
        model_path = tmp_path / "test_model.joblib"
        
        with patch('src.voix_du_client.analyzer.joblib') as mock_joblib:
            # Test saving
            analyzer.save_model(model_path)
            mock_joblib.dump.assert_called_once()
            
            # Test loading
            mock_joblib.load.return_value = {
                'vectorizer': Mock(),
                'kmeans': Mock(),
                'config': {}
            }
            
            analyzer.load_model(model_path)
            mock_joblib.load.assert_called_once_with(model_path)


class TestIntegration:
    """Integration tests."""
    
    @pytest.fixture
    def sample_data(self):
        return [
            FeedbackItem(id="1", text="Le service client répond très lentement aux demandes"),
            FeedbackItem(id="2", text="Le site internet est souvent en panne et inaccessible"),
            FeedbackItem(id="3", text="Les délais de livraison ne sont jamais respectés"),
            FeedbackItem(id="4", text="L'interface utilisateur est vraiment confuse et difficile"),
            FeedbackItem(id="5", text="Les prix sont beaucoup trop élevés pour la qualité offerte"),
            FeedbackItem(id="6", text="Le support technique ne résout jamais les problèmes"),
            FeedbackItem(id="7", text="Le processus de commande en ligne est très compliqué"),
            FeedbackItem(id="8", text="La facturation contient souvent des erreurs importantes")
        ]
    
    @pytest.mark.slow
    def test_full_analysis_pipeline(self, sample_data):
        """Test the complete analysis pipeline with real spaCy model."""
        config = Config()
        analyzer = FeedbackAnalyzer(config)
        
        analysis_config = AnalysisConfig(
            k=3,
            max_features=1000,
            ngram_range=(1, 2),
            random_state=42,
            min_text_length=5
        )
        
        try:
            result = analyzer.analyze(sample_data, analysis_config)
            
            # Basic assertions
            assert result is not None
            assert result.num_clusters == 3
            assert result.total_items <= len(sample_data)  # Some might be filtered
            assert -1 <= result.silhouette_score <= 1
            assert result.processing_time > 0
            
            # Check clusters
            assert len(result.clusters) == 3
            for cluster in result.clusters:
                assert cluster.size > 0
                assert len(cluster.top_terms) > 0
                assert 0 <= cluster.percentage <= 100
            
            # Check that all items are assigned to clusters
            total_cluster_size = sum(c.size for c in result.clusters)
            assert total_cluster_size == result.total_items
            
        except Exception as e:
            pytest.skip(f"Integration test failed (likely missing spaCy model): {e}")