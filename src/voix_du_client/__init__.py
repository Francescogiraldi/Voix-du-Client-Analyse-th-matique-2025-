"""Voix du Client - Customer feedback analysis tool."""

__version__ = "1.0.0"
__author__ = "Voix du Client Team"
__email__ = "contact@voixduclient.com"

from .analyzer import FeedbackAnalyzer
from .config import Config
from .models import AnalysisResult, ClusterInfo

__all__ = ["FeedbackAnalyzer", "Config", "AnalysisResult", "ClusterInfo"]