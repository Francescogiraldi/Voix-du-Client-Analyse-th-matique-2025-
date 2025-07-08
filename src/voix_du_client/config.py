"""Configuration management for Voix du Client."""

from pathlib import Path
from typing import List, Optional

from pydantic import Field, validator
from pydantic_settings import BaseSettings


class Config(BaseSettings):
    """Application configuration with validation."""

    # NLP Configuration
    nlp_model: str = Field(default="fr_core_news_sm", description="spaCy model name")
    max_features: int = Field(default=5000, ge=100, le=50000, description="TF-IDF vocabulary size")
    ngram_range_min: int = Field(default=1, ge=1, le=3, description="Minimum n-gram size")
    ngram_range_max: int = Field(default=2, ge=1, le=5, description="Maximum n-gram size")
    
    # Clustering Configuration
    default_k: int = Field(default=5, ge=2, le=20, description="Default number of clusters")
    min_k: int = Field(default=2, ge=2, le=10, description="Minimum number of clusters")
    max_k: int = Field(default=15, ge=5, le=50, description="Maximum number of clusters")
    random_state: int = Field(default=42, description="Random seed for reproducibility")
    
    # Text Processing
    extra_stopwords: List[str] = Field(
        default=["nous", "vous", "ils", "elles", "être", "avoir", "faire", "dire"],
        description="Additional French stopwords"
    )
    min_text_length: int = Field(default=10, ge=1, description="Minimum text length after cleaning")
    
    # File Paths
    data_dir: Path = Field(default=Path("data"), description="Data directory")
    models_dir: Path = Field(default=Path("models"), description="Models directory")
    logs_dir: Path = Field(default=Path("logs"), description="Logs directory")
    
    # Streamlit Configuration
    page_title: str = Field(default="Voix du Client", description="Streamlit page title")
    page_icon: str = Field(default="💬", description="Streamlit page icon")
    layout: str = Field(default="wide", description="Streamlit layout")
    
    # Logging
    log_level: str = Field(default="INFO", description="Logging level")
    log_format: str = Field(
        default="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}",
        description="Log format"
    )
    
    # Performance
    cache_ttl: int = Field(default=3600, ge=60, description="Cache TTL in seconds")
    max_file_size_mb: int = Field(default=50, ge=1, le=500, description="Maximum file size in MB")
    
    # Visualization
    wordcloud_width: int = Field(default=800, ge=400, description="WordCloud width")
    wordcloud_height: int = Field(default=400, ge=200, description="WordCloud height")
    wordcloud_background: str = Field(default="white", description="WordCloud background color")
    top_terms_display: int = Field(default=10, ge=3, le=20, description="Number of top terms to display")
    
    @validator("ngram_range_max")
    def validate_ngram_range(cls, v, values):
        """Ensure max n-gram is >= min n-gram."""
        if "ngram_range_min" in values and v < values["ngram_range_min"]:
            raise ValueError("ngram_range_max must be >= ngram_range_min")
        return v
    
    @validator("max_k")
    def validate_k_range(cls, v, values):
        """Ensure max_k is >= min_k."""
        if "min_k" in values and v < values["min_k"]:
            raise ValueError("max_k must be >= min_k")
        return v
    
    @validator("data_dir", "models_dir", "logs_dir")
    def create_directories(cls, v):
        """Create directories if they don't exist."""
        v.mkdir(parents=True, exist_ok=True)
        return v
    
    class Config:
        """Pydantic configuration."""
        env_file = ".env"
        env_prefix = "VOIX_"
        case_sensitive = False
        validate_assignment = True


# Global configuration instance
config = Config()