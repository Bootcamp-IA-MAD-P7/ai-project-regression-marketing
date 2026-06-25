"""Project configuration and reusable filesystem paths."""

from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"

RAW_DATA_PATH = DATA_DIR / "digital_marketing_dataset_30k.csv"
MODEL_PATH = MODELS_DIR / "revenue_model.pkl"
