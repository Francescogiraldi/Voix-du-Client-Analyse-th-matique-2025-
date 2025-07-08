"""Tests for the models module."""

import pytest
from datetime import datetime
from pydantic import ValidationError

from src.voix_du_client.models import (
    FeedbackItem, ClusterInfo, AnalysisResult, AnalysisConfig, ProcessingStats
)


class TestFeedbackItem:
    """Test cases for FeedbackItem model."""
    
    def test_valid_feedback_item(self):
        """Test creating a valid feedback item."""
        item = FeedbackItem(id="1", text="Test feedback")
        assert item.id == "1"
        assert item.text == "Test feedback"
        assert item.clean_text is None
        assert item.cluster is None
        assert item.timestamp is None
        assert item.metadata == {}
    
    def test_feedback_item_with_all_fields(self):
        """Test feedback item with all fields."""
        timestamp = datetime.now()
        metadata = {"source": "email", "priority": "high"}
        
        item = FeedbackItem(
            id="1",
            text="Test feedback",
            clean_text="test feedback",
            cluster=0,
            timestamp=timestamp,
            metadata=metadata
        )
        
        assert item.id == "1"
        assert item.text == "Test feedback"
        assert item.clean_text == "test feedback"
        assert item.cluster == 0
        assert item.timestamp == timestamp
        assert item.metadata == metadata
    
    def test_empty_text_validation(self):
        """Test validation of empty text."""
        with pytest.raises(ValidationError):
            FeedbackItem(id="1", text="")
        
        with pytest.raises(ValidationError):
            FeedbackItem(id="1", text="   ")
    
    def test_text_stripping(self):
        """Test that text is automatically stripped."""
        item = FeedbackItem(id="1", text="  Test feedback  ")
        assert item.text == "Test feedback"
    
    def test_negative_cluster_validation(self):
        """Test validation of negative cluster ID."""
        with pytest.raises(ValidationError):
            FeedbackItem(id="1", text="Test", cluster=-1)


class TestClusterInfo:
    """Test cases for ClusterInfo model."""
    
    def test_valid_cluster_info(self):
        """Test creating valid cluster info."""
        cluster = ClusterInfo(
            id=0,
            size=10,
            percentage=25.0,
            top_terms=["service", "client", "problème"]
        )
        
        assert cluster.id == 0
        assert cluster.size == 10
        assert cluster.percentage == 25.0
        assert cluster.top_terms == ["service", "client", "problème"]
        assert cluster.representative_texts == []
        assert cluster.centroid_distance is None
    
    def test_cluster_with_representative_texts(self):
        """Test cluster with representative texts."""
        cluster = ClusterInfo(
            id=0,
            size=5,
            percentage=50.0,
            top_terms=["service"],
            representative_texts=["Le service est lent", "Problème de service"]
        )
        
        assert len(cluster.representative_texts) == 2
    
    def test_empty_top_terms_validation(self):
        """Test validation of empty top terms."""
        with pytest.raises(ValidationError):
            ClusterInfo(
                id=0,
                size=10,
                percentage=25.0,
                top_terms=[]
            )
    
    def test_negative_values_validation(self):
        """Test validation of negative values."""
        with pytest.raises(ValidationError):
            ClusterInfo(
                id=-1,
                size=10,
                percentage=25.0,
                top_terms=["test"]
            )
        
        with pytest.raises(ValidationError):
            ClusterInfo(
                id=0,
                size=-1,
                percentage=25.0,
                top_terms=["test"]
            )
    
    def test_percentage_range_validation(self):
        """Test validation of percentage range."""
        with pytest.raises(ValidationError):
            ClusterInfo(
                id=0,
                size=10,
                percentage=-5.0,
                top_terms=["test"]
            )
        
        with pytest.raises(ValidationError):
            ClusterInfo(
                id=0,
                size=10,
                percentage=105.0,
                top_terms=["test"]
            )


class TestAnalysisResult:
    """Test cases for AnalysisResult model."""
    
    def test_valid_analysis_result(self):
        """Test creating valid analysis result."""
        clusters = [
            ClusterInfo(id=0, size=5, percentage=50.0, top_terms=["service"]),
            ClusterInfo(id=1, size=5, percentage=50.0, top_terms=["site"])
        ]
        
        result = AnalysisResult(
            clusters=clusters,
            total_items=10,
            silhouette_score=0.5,
            processing_time=2.5,
            config_snapshot={"k": 2}
        )
        
        assert len(result.clusters) == 2
        assert result.total_items == 10
        assert result.silhouette_score == 0.5
        assert result.processing_time == 2.5
        assert result.config_snapshot == {"k": 2}
        assert isinstance(result.timestamp, datetime)
    
    def test_cluster_size_validation(self):
        """Test validation that cluster sizes sum to total items."""
        clusters = [
            ClusterInfo(id=0, size=5, percentage=50.0, top_terms=["service"]),
            ClusterInfo(id=1, size=3, percentage=30.0, top_terms=["site"])  # Sum = 8, not 10
        ]
        
        with pytest.raises(ValidationError):
            AnalysisResult(
                clusters=clusters,
                total_items=10,
                silhouette_score=0.5,
                processing_time=2.5,
                config_snapshot={"k": 2}
            )
    
    def test_silhouette_score_range(self):
        """Test silhouette score range validation."""
        clusters = [ClusterInfo(id=0, size=10, percentage=100.0, top_terms=["test"])]
        
        with pytest.raises(ValidationError):
            AnalysisResult(
                clusters=clusters,
                total_items=10,
                silhouette_score=1.5,  # Out of range
                processing_time=2.5,
                config_snapshot={"k": 1}
            )
    
    def test_properties(self):
        """Test computed properties."""
        clusters = [
            ClusterInfo(id=0, size=8, percentage=80.0, top_terms=["service"]),
            ClusterInfo(id=1, size=2, percentage=20.0, top_terms=["site"])
        ]
        
        result = AnalysisResult(
            clusters=clusters,
            total_items=10,
            silhouette_score=0.5,
            processing_time=2.5,
            config_snapshot={"k": 2}
        )
        
        assert result.num_clusters == 2
        assert result.largest_cluster.id == 0
        assert result.largest_cluster.size == 8
        assert result.smallest_cluster.id == 1
        assert result.smallest_cluster.size == 2
    
    def test_empty_clusters_properties(self):
        """Test properties with empty clusters."""
        result = AnalysisResult(
            clusters=[],
            total_items=0,
            silhouette_score=0.0,
            processing_time=1.0,
            config_snapshot={}
        )
        
        assert result.num_clusters == 0
        assert result.largest_cluster is None
        assert result.smallest_cluster is None


class TestAnalysisConfig:
    """Test cases for AnalysisConfig model."""
    
    def test_valid_analysis_config(self):
        """Test creating valid analysis config."""
        config = AnalysisConfig(
            k=5,
            max_features=1000,
            ngram_range=(1, 2),
            random_state=42,
            min_text_length=10
        )
        
        assert config.k == 5
        assert config.max_features == 1000
        assert config.ngram_range == (1, 2)
        assert config.random_state == 42
        assert config.min_text_length == 10
    
    def test_k_range_validation(self):
        """Test validation of k range."""
        with pytest.raises(ValidationError):
            AnalysisConfig(
                k=1,  # Too small
                max_features=1000,
                ngram_range=(1, 2),
                random_state=42,
                min_text_length=10
            )
        
        with pytest.raises(ValidationError):
            AnalysisConfig(
                k=100,  # Too large
                max_features=1000,
                ngram_range=(1, 2),
                random_state=42,
                min_text_length=10
            )
    
    def test_max_features_range_validation(self):
        """Test validation of max_features range."""
        with pytest.raises(ValidationError):
            AnalysisConfig(
                k=5,
                max_features=50,  # Too small
                ngram_range=(1, 2),
                random_state=42,
                min_text_length=10
            )
    
    def test_ngram_range_validation(self):
        """Test validation of n-gram range."""
        with pytest.raises(ValidationError):
            AnalysisConfig(
                k=5,
                max_features=1000,
                ngram_range=(2, 1),  # Invalid range
                random_state=42,
                min_text_length=10
            )
        
        with pytest.raises(ValidationError):
            AnalysisConfig(
                k=5,
                max_features=1000,
                ngram_range=(0, 2),  # Invalid minimum
                random_state=42,
                min_text_length=10
            )


class TestProcessingStats:
    """Test cases for ProcessingStats model."""
    
    def test_valid_processing_stats(self):
        """Test creating valid processing stats."""
        stats = ProcessingStats(
            original_count=100,
            processed_count=95,
            filtered_count=5,
            avg_length_before=50.5,
            avg_length_after=45.2,
            processing_time=2.5
        )
        
        assert stats.original_count == 100
        assert stats.processed_count == 95
        assert stats.filtered_count == 5
        assert stats.avg_length_before == 50.5
        assert stats.avg_length_after == 45.2
        assert stats.processing_time == 2.5
    
    def test_retention_rate_property(self):
        """Test retention rate calculation."""
        stats = ProcessingStats(
            original_count=100,
            processed_count=80,
            filtered_count=20,
            avg_length_before=50.0,
            avg_length_after=45.0,
            processing_time=1.0
        )
        
        assert stats.retention_rate == 80.0
    
    def test_retention_rate_zero_original(self):
        """Test retention rate with zero original count."""
        stats = ProcessingStats(
            original_count=0,
            processed_count=0,
            filtered_count=0,
            avg_length_before=0.0,
            avg_length_after=0.0,
            processing_time=0.0
        )
        
        assert stats.retention_rate == 0.0
    
    def test_negative_values_validation(self):
        """Test validation of negative values."""
        with pytest.raises(ValidationError):
            ProcessingStats(
                original_count=-1,
                processed_count=0,
                filtered_count=0,
                avg_length_before=0.0,
                avg_length_after=0.0,
                processing_time=0.0
            )