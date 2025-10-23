# tests/test_predictor.py
import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from pathlib import Path
import torch  # <-- NEW: Needed for CNN testing

from src.predictor import ActivityPredictor, CLASS_NAMES

# We need to patch the CNNModel and torch.load *where they are used*
# (i.e., inside the 'src.predictor' module)
CNN_MODEL_PATH = "src.predictor.CNNModel"
TORCH_LOAD_PATH = "src.predictor.torch.load"
JOBLIB_LOAD_PATH = "joblib.load"


@pytest.fixture
def sample_data_window():
    """Provides a sample 10-item data window for tests."""
    return [
        {"X": 1.0, "Y": 2.0, "Z": 3.0},
        {"X": 1.1, "Y": 2.1, "Z": 3.1},
        {"X": 1.2, "Y": 2.2, "Z": 3.2},
        {"X": 1.3, "Y": 2.3, "Z": 3.3},
        {"X": 1.4, "Y": 2.4, "Z": 3.4},
        {"X": 1.5, "Y": 2.5, "Z": 3.5},
        {"X": 1.6, "Y": 2.6, "Z": 3.6},
        {"X": 1.7, "Y": 2.7, "Z": 3.7},
        {"X": 1.8, "Y": 2.8, "Z": 3.8},
        {"X": 1.9, "Y": 2.9, "Z": 3.9},
    ]


# --- Test RandomForest (RF) Model Path ---


@patch(JOBLIB_LOAD_PATH)
def test_predictor_init_loads_rf_model(mock_joblib_load):
    """Tests that the RF model is loaded correctly."""
    mock_model = MagicMock()
    mock_joblib_load.return_value = mock_model
    model_path = Path("fake/model.joblib")

    # --- MODIFIED: Added model_type='rf' ---
    predictor = ActivityPredictor(model_path, model_type="rf")

    mock_joblib_load.assert_called_once_with(model_path)
    assert predictor.model is mock_model
    assert predictor.model_type == "rf"


@patch(JOBLIB_LOAD_PATH, side_effect=FileNotFoundError)
def test_predictor_init_rf_file_not_found(mock_joblib_load):
    """Tests FileNotFoundError for the RF model path."""
    with pytest.raises(FileNotFoundError):
        # --- MODIFIED: Added model_type='rf' ---
        ActivityPredictor(Path("non/existent/path/model.joblib"), model_type="rf")


@patch(JOBLIB_LOAD_PATH)
@patch("src.features.calculate_mean", return_value=1.0)
@patch("src.features.calculate_std_dev", return_value=0.1)
@patch("src.features.calculate_rms", return_value=1.5)
def test_predictor_predict_method_rf(
    mock_rms, mock_std, mock_mean, mock_joblib_load, sample_data_window
):
    """Tests the 'predict' method logic for the RF model."""
    mock_model = MagicMock()
    # Predict class index 1 ('drinking')
    mock_model.predict.return_value = [1]
    mock_joblib_load.return_value = mock_model

    # --- MODIFIED: Added model_type='rf' ---
    predictor = ActivityPredictor(Path("fake/model.joblib"), model_type="rf")
    prediction = predictor.predict(sample_data_window)

    # Check that index 1 maps to 'drinking'
    assert prediction == CLASS_NAMES[1]
    predictor.model.predict.assert_called_once()

    # Check that the feature extraction was correct
    call_args = predictor.model.predict.call_args
    input_df = call_args[0][0]
    assert isinstance(input_df, pd.DataFrame)
    assert list(input_df.columns) == predictor.feature_order
    assert input_df["X_mean"].iloc[0] == 1.0
    assert input_df["Y_std"].iloc[0] == 0.1
    assert input_df["Z_rms"].iloc[0] == 1.5


# --- Test CNN Model Path (NEW TESTS) ---


@patch(CNN_MODEL_PATH)
@patch(TORCH_LOAD_PATH)
def test_predictor_init_loads_cnn_model(mock_torch_load, mock_cnn_class):
    """Tests that the CNN model is loaded and set to eval mode."""
    mock_model_instance = MagicMock()
    mock_cnn_class.return_value = mock_model_instance
    mock_state_dict = MagicMock()
    mock_torch_load.return_value = mock_state_dict
    model_path = Path("fake/model.pth")

    predictor = ActivityPredictor(model_path, model_type="cnn")

    # Check model was instantiated correctly
    # TODO: Adjust (3, 3) if your model class takes different args
    mock_cnn_class.assert_called_once_with(input_channels=3, num_classes=3)

    # Check model weights were loaded
    mock_torch_load.assert_called_once_with(model_path)
    mock_model_instance.load_state_dict.assert_called_once_with(mock_state_dict)

    # Check model was set to evaluation mode
    mock_model_instance.eval.assert_called_once()
    assert predictor.model is mock_model_instance
    assert predictor.model_type == "cnn"


@patch(CNN_MODEL_PATH)
@patch(TORCH_LOAD_PATH, side_effect=FileNotFoundError)
def test_predictor_init_cnn_file_not_found(mock_torch_load, mock_cnn_class):
    """Tests FileNotFoundError for the CNN model path."""
    with pytest.raises(FileNotFoundError):
        ActivityPredictor(Path("non/existent/path/model.pth"), model_type="cnn")


@patch(CNN_MODEL_PATH)
@patch(TORCH_LOAD_PATH)
def test_predictor_predict_method_cnn(
    mock_torch_load, mock_cnn_class, sample_data_window
):
    """Tests the 'predict' method logic for the CNN model."""
    mock_model_instance = MagicMock()
    # Simulate model predicting class 2 ('driving')
    # Output is raw logits: (batch_size, num_classes)
    mock_model_instance.return_value = torch.tensor([[0.1, 0.5, 2.5]])
    mock_cnn_class.return_value = mock_model_instance
    mock_torch_load.return_value = MagicMock()

    predictor = ActivityPredictor(Path("fake/model.pth"), model_type="cnn")
    prediction = predictor.predict(sample_data_window)

    # Check that the max logit (index 2) maps to 'driving'
    assert prediction == CLASS_NAMES[2]

    # Check that the model was called once
    predictor.model.assert_called_once()

    # --- Check that the data preprocessing was correct ---
    call_args = predictor.model.call_args
    input_tensor = call_args[0][0]

    # 1. Check type and shape: (batch=1, channels=3, window_size=10)
    assert isinstance(input_tensor, torch.Tensor)
    assert input_tensor.shape == (1, 3, 10)

    # 2. Check that the data was transposed correctly
    # X data (channel 0)
    expected_x = torch.tensor([d["X"] for d in sample_data_window])
    assert torch.all(torch.eq(input_tensor[0, 0, :], expected_x))

    # Y data (channel 1)
    expected_y = torch.tensor([d["Y"] for d in sample_data_window])
    assert torch.all(torch.eq(input_tensor[0, 1, :], expected_y))


# --- Test General Error Cases (NEW TEST) ---


def test_predictor_init_invalid_type():
    """Tests that an unknown model_type raises a ValueError."""
    with pytest.raises(ValueError) as e:
        ActivityPredictor(Path("fake/model"), model_type="svm")

    assert "Unknown model_type" in str(e.value)
