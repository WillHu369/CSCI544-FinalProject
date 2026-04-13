from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
DATA_DIR = ARTIFACTS_DIR / "data"
MODELS_DIR = ARTIFACTS_DIR / "models"
RUNS_DIR = ARTIFACTS_DIR / "runs"

DEFAULT_HC3_HF_DATASET = "Hello-SimpleAI/HC3"
DEFAULT_GPTZERO_LM = "gpt2"
DEFAULT_RANDOM_STATE = 42
DEFAULT_TARGET_FPR = 0.01


def ensure_default_directories() -> None:
    for path in (ARTIFACTS_DIR, DATA_DIR, MODELS_DIR, RUNS_DIR):
        path.mkdir(parents=True, exist_ok=True)
