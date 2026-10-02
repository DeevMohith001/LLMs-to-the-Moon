"""
VADER Lexicon-Based Baseline for Financial Sentiment Analysis.

Re-exports VADERBaseline from the canonical implementation in src.models.vader_model.

NOTE: The canonical implementation lives in src/models/vader_model.py.
This module provides a convenience import path: src.baselines.vader.VADERBaseline
"""

# The VADERBaseline class is canonically defined in src/models/vader_model.py.
# Both import paths are valid:
#   from src.baselines.vader import VADERBaseline
#   from src.models.vader_model import VADERBaseline
from src.models.vader_model import VADERBaseline

__all__ = ["VADERBaseline"]
