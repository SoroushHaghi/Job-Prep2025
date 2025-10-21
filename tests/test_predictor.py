# tests/test_predictor.py
import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from pathlib import Path

# Import the class we want to test
from src.predictor import ActivityPredictor


@pytest.fixture
def sample_data_window():
    """Provides a sample 10-item data window for tests."""
    # A simple, consistent window of data
    return [
        {"x": 1.0, "y": 2.0, "z": 3.0},
        {"x": 1.1, "y": 2.1, "z": 3.1},
        {"x": 1.2, "y": 2.2, "z": 3.2},
        {"x": 1.3, "y": 2.3, "z": 3.3},
        {"x": 1.4, "y": 2.4, "z": 3.4},
        {"x": 1.5, "y": 2.5, "z": 3.5},
        {"x": 1.6, "y": 2.6, "z": 3.6},
        {"x": 1.7, "y": 2.7, "z": 3.7},
        {"x": 1.8, "y": 2.8, "z": 3.8},
        {"x": 1.9, "y": 2.9, "z": 3.9},
    ]


# We use 'patch' to replace 'joblib.load' with a mock
@patch("joblib.load")
def test_predictor_init_loads_model(mock_joblib_load):
    """
    Tests if the ActivityPredictor class attempts to load a model
    from the correct path during initialization.
    """
    # 1. Setup: Create a fake model object
    mock_model = MagicMock()
    mock_joblib_load.return_value = mock_model

    model_path = Path("fake/model.joblib")

    # 2. Execute: Create an instance of the predictor
    predictor = ActivityPredictor(model_path)

    # 3. Assert: Check if joblib.load was called exactly once with the correct path
    mock_joblib_load.assert_called_once_with(model_path)
    # Check if the loaded model was stored in the instance
    assert predictor.model is mock_model


def test_predictor_init_file_not_found():
    """
    Tests if the constructor correctly raises FileNotFoundError
    if the model file does not exist.
    """
    # Use pytest.raises to check if the expected exception is raised
    with pytest.raises(FileNotFoundError):
        ActivityPredictor(Path("non/existent/path/model.joblib"))


# We need to mock multiple things:
# 1. 'joblib.load' (so the class can be created)
# 2. The feature calculation functions (so we don't recalculate them)
@patch("joblib.load")
@patch("src.features.calculate_mean", return_value=1.0)
@patch("src.features.calculate_std_dev", return_value=0.1)
@patch("src.features.calculate_rms", return_value=1.5)
def test_predictor_predict_method(
    mock_rms, mock_std, mock_mean, mock_joblib_load, sample_data_window
):
    """
    Tests the 'predict' method logic.
    Ensures it extracts features correctly, builds the DataFrame in the
    correct order, and calls the model's predict method.
    """
    # 1. Setup:
    # Create a mock model and set its 'predict' method to return a known value, e.g., [1]
    mock_model = MagicMock()
    mock_model.predict.return_value = [1]  # [1] represents 'Movement'
    mock_joblib_load.return_value = mock_model

    # 2. Execute:
    predictor = ActivityPredictor(Path("fake/model.joblib"))
    prediction = predictor.predict(sample_data_window)

    # 3. Assert:
    # Check if the method returned the correct prediction
    assert prediction == 1

    # Check if the model's predict method was called once
    predictor.model.predict.assert_called_once()

    # --- Advanced Assert ---
    # Check if the model was called with the correctly formatted DataFrame

    # Get the arguments that 'model.predict' was called with
    call_args = predictor.model.predict.call_args
    input_df = call_args[0][0]  # The first positional argument

    # Check if it's a DataFrame
    assert isinstance(input_df, pd.DataFrame)
    # Check if it has only one row
    assert len(input_df) == 1
    # Check if the columns are in the correct, predefined order
    assert list(input_df.columns) == predictor.feature_order

    # Check if the values match our mocked feature values
    # (We mocked mean=1.0, std=0.1, rms=1.5 for all axes)
    assert input_df["acc_x_mean"].iloc[0] == 1.0
    assert input_df["acc_y_std"].iloc[0] == 0.1
    assert input_df["acc_z_rms"].iloc[0] == 1.5
