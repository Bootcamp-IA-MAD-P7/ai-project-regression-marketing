"""Preprocessing utilities for pre-launch campaign revenue forecasting."""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from pandas.api.types import is_numeric_dtype

from src.config import PLANNING_TIME_FEATURES, RANDOM_STATE, TARGET, TEST_SIZE


def prepare_features_and_target(
    data,
    target_column=TARGET,
    feature_columns=None,
):
    """Create a planning-time feature matrix and target vector from raw data."""
    if target_column not in data.columns:
        raise ValueError(f"Target column `{target_column}` is not present in the data.")

    selected_features = (
        list(feature_columns) if feature_columns is not None else PLANNING_TIME_FEATURES
    )
    missing_features = [
        column for column in selected_features if column not in data.columns
    ]

    if missing_features:
        raise ValueError(
            "Missing required planning-time feature columns: "
            + ", ".join(missing_features)
        )

    features = data.loc[:, selected_features].copy()
    target = data[target_column].copy()

    return features, target


def identify_feature_types(features):
    """Return numerical and categorical feature column names."""
    numerical_features = [
    column
    for column in features.columns
    if is_numeric_dtype(features[column]) and features[column].dtype != "bool"
    ]
    categorical_features = [
        column for column in features.columns if column not in numerical_features
    ]

    return numerical_features, categorical_features


def build_preprocessor(numerical_features, categorical_features):
    """Build the sklearn preprocessing transformer used before modeling."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numerical_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )


def build_preprocessor_from_features(features):
    """Identify column types and return a fitted-ready preprocessing transformer."""
    numerical_features, categorical_features = identify_feature_types(features)
    preprocessor = build_preprocessor(numerical_features, categorical_features)

    return preprocessor, numerical_features, categorical_features


def split_features_target(
    features,
    target,
    *,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
):
    """Split feature matrix and target vector for model evaluation."""
    return train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
    )
