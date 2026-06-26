"""Training workflow for the pre-launch campaign revenue forecast model."""

from src.data_loader import load_raw_data
from src.evaluation import evaluate_regression, overfitting_summary
from src.model import compare_models, optimize_random_forest
from src.prediction import save_model
from src.preprocessing import (
    build_preprocessor_from_features,
    prepare_features_and_target,
    split_features_target,
)


def train_forecast_model(data=None, *, model_path=None):
    """Train, evaluate, optimize, and save the revenue forecast pipeline."""
    raw_data = load_raw_data() if data is None else data
    features, target = prepare_features_and_target(raw_data)
    preprocessor, numerical_features, categorical_features = (
        build_preprocessor_from_features(features)
    )

    features_train, features_test, target_train, target_test = split_features_target(
        features,
        target,
    )

    model_results, fitted_models = compare_models(
        preprocessor,
        features_train,
        features_test,
        target_train,
        target_test,
    )

    random_search, optimization_time = optimize_random_forest(
        preprocessor,
        features_train,
        target_train,
    )

    forecast_model = random_search.best_estimator_
    predictions = forecast_model.predict(features_test)
    metrics = evaluate_regression(target_test, predictions)
    overfitting = overfitting_summary(
        forecast_model,
        features_train,
        target_train,
        features_test,
        target_test,
    )
    saved_model_path = save_model(forecast_model, model_path=model_path)

    return {
        "model": forecast_model,
        "model_path": saved_model_path,
        "metrics": metrics,
        "overfitting": overfitting,
        "model_comparison": model_results,
        "fitted_models": fitted_models,
        "best_params": random_search.best_params_,
        "best_cv_score": random_search.best_score_,
        "optimization_time": optimization_time,
        "features": features,
        "numerical_features": numerical_features,
        "categorical_features": categorical_features,
    }


if __name__ == "__main__":
    results = train_forecast_model()

    print("=" * 60)
    print("Training completed successfully")
    print("=" * 60)

    print(f"Saved model: {results['model_path']}")

    print("\nMetrics:")
    for metric, value in results["metrics"].items():
        print(f"{metric}: {value:.4f}")

    print("\nOverfitting:")
    print(results["overfitting"].to_string(index=False))
