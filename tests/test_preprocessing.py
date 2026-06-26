import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from src.config import (
    IDENTIFIER_COLUMNS,
    MODEL_PATH,
    PLANNING_TIME_FEATURES,
    POST_LAUNCH_METRICS,
    TARGET,
)
from src.prediction import load_model, predict_revenue, save_model
from src.preprocessing import (
    build_preprocessor_from_features,
    prepare_features_and_target,
)
from src.model import build_model_pipeline


class ForecastPreprocessingTests(unittest.TestCase):
    def setUp(self):
        self.data = pd.DataFrame(
            {
                "month": [1, 2, 3, 4],
                "week": [1, 5, 9, 13],
                "day_of_week": ["Mon", "Tue", "Wed", "Thu"],
                "post_hour": [9, 12, 15, 18],
                "season": ["Winter", "Winter", "Spring", "Spring"],
                "is_holiday": [0, 0, 1, 0],
                "is_weekend": [0, 0, 0, 0],
                "country": ["KSA", "UAE", "Qatar", "KSA"],
                "market_tier": ["Tier 1", "Tier 1", "Tier 2", "Tier 1"],
                "account": ["A", "B", "C", "A"],
                "account_type": ["Brand", "Creator", "Brand", "Brand"],
                "platform": ["Meta", "TikTok", "Google Search", "Meta"],
                "placement": ["Feed", "In-Feed", "Search", "Stories"],
                "funnel_stage": [
                    "Awareness",
                    "Consideration",
                    "Conversion",
                    "Conversion",
                ],
                "objective": ["Reach", "Traffic", "Sales", "Leads"],
                "theme": ["Promo", "Seasonal", "Launch", "Promo"],
                "spend": [100.0, 250.0, 500.0, 350.0],
                "impressions": [10000, 20000, 30000, 40000],
                "reach": [5000, 9000, 15000, 21000],
                "frequency": [2.0, 2.2, 2.1, 1.9],
                "clicks": [100, 250, 700, 450],
                "conversions": [0, 2, 30, 12],
                "video_views": [1200, 3000, 0, 500],
                "date": [
                    "2025-01-01",
                    "2025-02-01",
                    "2025-03-01",
                    "2025-04-01",
                ],
                "campaign_id": ["c1", "c2", "c3", "c4"],
                "campaign_name": [
                    "Campaign 1",
                    "Campaign 2",
                    "Campaign 3",
                    "Campaign 4",
                ],
                "ad_group_id": ["g1", "g2", "g3", "g4"],
                "ad_group_name": ["Group 1", "Group 2", "Group 3", "Group 4"],
                "ad_id": ["a1", "a2", "a3", "a4"],
                "ad_name": ["Ad 1", "Ad 2", "Ad 3", "Ad 4"],
                "revenue": [0.0, 120.0, 2500.0, 900.0],
            }
        )

    def test_prepare_features_uses_only_planning_time_columns(self):
        features, target = prepare_features_and_target(self.data)

        self.assertEqual(features.columns.tolist(), PLANNING_TIME_FEATURES)
        self.assertEqual(target.name, TARGET)

    def test_prepare_features_excludes_post_launch_metrics_and_identifiers(self):
        features, _ = prepare_features_and_target(self.data)

        forbidden_columns = set(POST_LAUNCH_METRICS + IDENTIFIER_COLUMNS + [TARGET])
        self.assertTrue(forbidden_columns.isdisjoint(features.columns))

    def test_saved_forecast_pipeline_loads_and_predicts(self):
        features, target = prepare_features_and_target(self.data)
        preprocessor, _, _ = build_preprocessor_from_features(features)
        pipeline = build_model_pipeline(
            preprocessor,
            RandomForestRegressor(n_estimators=5, random_state=42),
        )
        pipeline.fit(features, target)

        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / MODEL_PATH.name
            save_model(pipeline, model_path=model_path)
            loaded_model = load_model(model_path=model_path)

        predictions = predict_revenue(features.head(2), model=loaded_model)

        self.assertEqual(predictions.shape, (2,))
        self.assertTrue(np.isfinite(predictions).all())


if __name__ == "__main__":
    unittest.main()
