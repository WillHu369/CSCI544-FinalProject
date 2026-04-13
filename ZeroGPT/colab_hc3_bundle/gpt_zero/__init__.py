"""Local GPTZero-style detector and classical baseline package."""

from gpt_zero.config import DEFAULT_GPTZERO_LM, PROJECT_ROOT
from gpt_zero.gptzero_like import GPTZeroLikeDetector, ScorerConfig
from gpt_zero.classical import ClassicalBaselineSuite

__all__ = [
    "ClassicalBaselineSuite",
    "DEFAULT_GPTZERO_LM",
    "GPTZeroLikeDetector",
    "PROJECT_ROOT",
    "ScorerConfig",
]
