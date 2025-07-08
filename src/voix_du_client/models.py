"""Data models for Voix du Client."""

from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, validator


class FeedbackItem(BaseModel):
    """Individual feedback item."""
    
    id: str = Field(..., description="Unique identifier")
    text: str = Field(..., min_length=1, description="Original feedback text")
    clean_text: Optional[str] = Field(None, description="Cleaned and processed text")
    cluster: Optional[int] = Field(None, ge=0, description="Assigned cluster ID")
    timestamp: Optional[datetime] = Field(None, description="Feedback timestamp")
    metadata: Optional[Dict] = Field(default_factory=dict, description="Additional metadata")
    
    @validator("text")
    def validate_text(cls, v):
        """Ensure text is not empty after stripping."""
        if not v.strip():
            raise ValueError("Text cannot be empty")
        return v.strip()


class ClusterInfo(BaseModel):
    """Information about a cluster."""
    
    id: int = Field(..., ge=0, description="Cluster ID")
    size: int = Field(..., ge=0, description="Number of items in cluster")
    percentage: float = Field(..., ge=0, le=100, description="Percentage of total items")
    top_terms: List[str] = Field(..., description="Most representative terms")
    representative_texts: List[str] = Field(default_factory=list, description="Sample texts")
    centroid_distance: Optional[float] = Field(None, description="Average distance to centroid")
    
    @validator("top_terms")
    def validate_top_terms(cls, v):
        """Ensure we have at least one term."""
        if not v:
            raise ValueError("Cluster must have at least one top term")
        return v


class AnalysisResult(BaseModel):
    """Complete analysis result."""
    
    clusters: List[ClusterInfo] = Field(..., description="Cluster information")
    total_items: int = Field(..., ge=0, description="Total number of feedback items")
    silhouette_score: float = Field(..., ge=-1, le=1, description="Clustering quality score")
    processing_time: float = Field(..., ge=0, description="Processing time in seconds")
    timestamp: datetime = Field(default_factory=datetime.now, description="Analysis timestamp")
    config_snapshot: Dict = Field(..., description="Configuration used for analysis")
    
    @validator("clusters")
    def validate_clusters(cls, v, values):
        """Ensure cluster sizes sum to total items."""
        if "total_items" in values:
            total_cluster_items = sum(cluster.size for cluster in v)
            if total_cluster_items != values["total_items"]:
                raise ValueError("Cluster sizes must sum to total items")
        return v
    
    @property
    def num_clusters(self) -> int:
        """Number of clusters."""
        return len(self.clusters)
    
    @property
    def largest_cluster(self) -> Optional[ClusterInfo]:
        """Get the largest cluster."""
        return max(self.clusters, key=lambda c: c.size) if self.clusters else None
    
    @property
    def smallest_cluster(self) -> Optional[ClusterInfo]:
        """Get the smallest cluster."""
        return min(self.clusters, key=lambda c: c.size) if self.clusters else None


class AnalysisConfig(BaseModel):
    """Configuration for a specific analysis run."""
    
    k: int = Field(..., ge=2, le=50, description="Number of clusters")
    max_features: int = Field(..., ge=100, le=50000, description="TF-IDF vocabulary size")
    ngram_range: tuple = Field(..., description="N-gram range")
    random_state: int = Field(..., description="Random seed")
    min_text_length: int = Field(..., ge=1, description="Minimum text length")
    
    @validator("ngram_range")
    def validate_ngram_range(cls, v):
        """Validate n-gram range."""
        if len(v) != 2 or v[0] > v[1] or v[0] < 1:
            raise ValueError("Invalid n-gram range")
        return v


class ProcessingStats(BaseModel):
    """Statistics about text processing."""
    
    original_count: int = Field(..., ge=0, description="Original number of texts")
    processed_count: int = Field(..., ge=0, description="Number of texts after processing")
    filtered_count: int = Field(..., ge=0, description="Number of texts filtered out")
    avg_length_before: float = Field(..., ge=0, description="Average length before processing")
    avg_length_after: float = Field(..., ge=0, description="Average length after processing")
    processing_time: float = Field(..., ge=0, description="Processing time in seconds")
    
    @property
    def retention_rate(self) -> float:
        """Percentage of texts retained after processing."""
        if self.original_count == 0:
            return 0.0
        return (self.processed_count / self.original_count) * 100