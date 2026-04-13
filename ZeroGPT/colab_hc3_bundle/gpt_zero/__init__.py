"""Local GPTZero-style detector and classical baseline package."""

from gpt_zero.config import DEFAULT_GPTZERO_LM, PROJECT_ROOT

__all__ = ["ClassicalBaselineSuite", "DEFAULT_GPTZERO_LM", "GPTZeroLikeDetector", "PROJECT_ROOT", "ScorerConfig"]


def __getattr__(name: str):
    if name in {"GPTZeroLikeDetector", "ScorerConfig"}:
        from gpt_zero.gptzero_like import GPTZeroLikeDetector, ScorerConfig

        return {"GPTZeroLikeDetector": GPTZeroLikeDetector, "ScorerConfig": ScorerConfig}[name]
    if name == "ClassicalBaselineSuite":
        from gpt_zero.classical import ClassicalBaselineSuite

        return ClassicalBaselineSuite
    raise AttributeError(f"module 'gpt_zero' has no attribute {name!r}")
