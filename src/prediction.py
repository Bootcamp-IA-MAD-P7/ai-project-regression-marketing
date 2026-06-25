"""Prediction utilities prepared for future model integration."""

from pathlib import Path

from src.config import MODEL_PATH


def load_model(model_path=None):
    """Load a trained revenue prediction model.

    This placeholder defines the project interface for model loading. The
    implementation should be connected once the modeling workflow persists a
    trained model artifact.
    """
    resolved_model_path = Path(model_path) if model_path is not None else MODEL_PATH
    raise NotImplementedError(
        f"Model loading is not implemented yet. Expected model path: {resolved_model_path}"
    )


def predict_revenue(input_data, model=None):
    """Predict revenue for application input data.

    This placeholder defines the prediction interface for Streamlit, tests, and
    future database-backed workflows.
    """
    if model is None:
        model = load_model()

    raise NotImplementedError(
        "Revenue prediction is not implemented yet. Connect this function to the trained model."
    )
