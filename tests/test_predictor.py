# tests/test_predictor.py
import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from pathlib import Path

from src.predictor import ActivityPredictor

# Note: 'from src import features' is no longer needed here


@pytest.fixture
def sample_data_window():
    """Provides a sample 10-item data window for tests (using uppercase keys)."""
    return [
        {"X": 1.0, "Y": 2.0, "Z": 3.0},  # Uppercase
        {"X": 1.1, "Y": 2.1, "Z": 3.1},  # Uppercase
        {"X": 1.2, "Y": 2.2, "Z": 3.2},
        {"X": 1.3, "Y": 2.3, "Z": 3.3},
        {"X": 1.4, "Y": 2.4, "Z": 3.4},
        {"X": 1.5, "Y": 2.5, "Z": 3.5},
        {"X": 1.6, "Y": 2.6, "Z": 3.6},
        {"X": 1.7, "Y": 2.7, "Z": 3.7},
        {"X": 1.8, "Y": 2.8, "Z": 3.8},
        {"X": 1.9, "Y": 2.9, "Z": 3.9},
    ]


@patch("joblib.load")
def test_predictor_init_loads_model(mock_joblib_load):
    # This test remains unchanged
    mock_model = MagicMock()
    mock_joblib_load.return_value = mock_model
    model_path = Path("fake/model.joblib")
    predictor = ActivityPredictor(model_path)
    mock_joblib_load.assert_called_once_with(model_path)
    assert predictor.model is mock_model


def test_predictor_init_file_not_found():
    # This test remains unchanged
    with pytest.raises(FileNotFoundError):
        ActivityPredictor(Path("non/existent/path/model.joblib"))


@patch("joblib.load")
@patch("src.features.calculate_mean", return_value=1.0)
@patch("src.features.calculate_std_dev", return_value=0.1)
@patch("src.features.calculate_rms", return_value=1.5)
def test_predictor_predict_method(
    mock_rms, mock_std, mock_mean, mock_joblib_load, sample_data_window
):
    """Tests the 'predict' method logic with corrected feature names."""
    mock_model = MagicMock()
    mock_model.predict.return_value = [1]
    mock_joblib_load.return_value = mock_model

    predictor = ActivityPredictor(Path("fake/model.joblib"))
    prediction = predictor.predict(sample_data_window)

    assert prediction == "drinking"
    predictor.model.predict.assert_called_once()

    call_args = predictor.model.predict.call_args
    input_df = call_args[0][0]

    assert isinstance(input_df, pd.DataFrame)
    assert len(input_df) == 1
    # Check if columns match the new uppercase feature order
    assert list(input_df.columns) == predictor.feature_order

    # Check values using the new uppercase feature names
    assert input_df["X_mean"].iloc[0] == 1.0
    assert input_df["Y_std"].iloc[0] == 0.1
    assert input_df["Z_rms"].iloc[0] == 1.5
