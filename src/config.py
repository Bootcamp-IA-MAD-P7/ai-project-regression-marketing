"""Project configuration for pre-launch revenue forecasting."""

from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"

# Ensure the models directory exists
MODELS_DIR.mkdir(parents=True, exist_ok=True)

RAW_DATA_PATH = DATA_DIR / "digital_marketing_dataset_30k.csv"

MODEL_FILENAME = "campaign_revenue_forecast_model.joblib"
MODEL_PATH = MODELS_DIR / MODEL_FILENAME

TARGET = "revenue"
TARGET_COLUMN = TARGET
RANDOM_STATE = 42
TEST_SIZE = 0.20

PLANNING_TIME_FEATURES = [
    "month",
    "week",
    "day_of_week",
    "post_hour",
    "season",
    "is_holiday",
    "is_weekend",
    "country",
    "market_tier",
    "account",
    "account_type",
    "platform",
    "placement",
    "funnel_stage",
    "objective",
    "theme",
    "spend",
]

POST_LAUNCH_METRICS = [
    "impressions",
    "reach",
    "frequency",
    "clicks",
    "conversions",
    "video_views",
]

IDENTIFIER_COLUMNS = [
    "date",
    "campaign_id",
    "campaign_name",
    "ad_group_id",
    "ad_group_name",
    "ad_id",
    "ad_name",
]

EXCLUDED_COLUMNS = [
    TARGET,
    *POST_LAUNCH_METRICS,
    *IDENTIFIER_COLUMNS,
]

RF_PARAM_DISTRIBUTIONS = {
    "model__n_estimators": [100, 150],
    "model__max_depth": [10, 15, 20],
    "model__min_samples_split": [5, 10, 15],
    "model__min_samples_leaf": [2, 4, 6],
    "model__max_features": ["sqrt", "log2", 0.7],
}
FEATURE_POLICY = {
    "planning": PLANNING_TIME_FEATURES,
    "excluded": POST_LAUNCH_METRICS,
    "identifiers": IDENTIFIER_COLUMNS,
}
# =====================================================
# Feature Policy
# =====================================================
#
# Only planning-time variables are allowed during model
# training in order to simulate a real campaign planning
# scenario.
#
# Post-launch metrics are intentionally excluded to
# prevent data leakage.
#