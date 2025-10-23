# src/train_dl.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from rich import print
import numpy as np

# --- Project Imports ---
from .utils import (
    load_and_segment_all,
    CLASS_MAP,
    WINDOW_SIZE,
    PROJECT_ROOT,
)
from .model import CNNModel  # Import the new model class

# --- Constants ---
# Hyperparameters
LEARNING_RATE = 0.001
BATCH_SIZE = 64
EPOCHS = 20

# Paths
# --- THIS LINE IS NOW CORRECTED ---
DATA_DIR = PROJECT_ROOT / "data"
MODEL_SAVE_PATH = PROJECT_ROOT / "models" / "cnn_model.pth"

# Get class names in the correct order (matching CLASS_MAP)
CLASS_NAMES = list(CLASS_MAP.keys())


def create_sequences(data, labels, window_size):
    """
    Converts raw data and labels into sequences (windows).
    """
    X, y = [], []
    # This logic assumes 'data' is a list of DataFrames, one for each file
    for df, label_index in zip(data, labels):
        # Create sliding windows
        for i in range(len(df) - window_size + 1):
            window = df.iloc[i : i + window_size][["X", "Y", "Z"]].values
            # Transpose to (channels, sequence_length) -> (3, 10)
            X.append(window.T)
            y.append(label_index)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


def train_deep_learning_model():
    """
    Main function to train and evaluate the 1D-CNN model.
    """
    print("🧠 [bold]Starting 1D-CNN Deep Learning Training Process...[/bold]")

    # --- 1. Load Data ---
    print(f"Loading and segmenting raw data from [cyan]{DATA_DIR}[/cyan]...")
    try:
        # Use window_size=1 (no segmentation) to load full files
        # We will create our own windows
        all_data, all_labels = load_and_segment_all(DATA_DIR, window_size=1)
    except FileNotFoundError:
        print(f"[bold red]Error: Raw data not found in {DATA_DIR}[/bold red]")
        print("Please ensure 'throwing.csv', 'drinking.csv', etc., exist.")
        return
    except Exception as e:
        print(f"[bold red]Error loading data: {e}[/bold red]")
        return

    # --- 2. Create Sequences (Windows) ---
    print(f"Creating sequences with window size {WINDOW_SIZE}...")
    # This creates the (N, 3, 10) shaped dataset
    X, y = create_sequences(all_data, all_labels, WINDOW_SIZE)
    print(f"Total sequences created: {X.shape[0]}")

    # --- 3. Split Data ---
    print("Splitting data into training and testing sets (80/20 split)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Training samples: {X_train.shape[0]}")
    print(f"Testing samples: {X_test.shape[0]}")

    # --- 4. Create PyTorch DataLoaders ---
    # Convert numpy arrays to PyTorch Tensors
    train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    test_dataset = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # --- 5. Initialize Model, Loss, and Optimizer ---
    # input_channels=3 (X, Y, Z), num_classes=3
    model = CNNModel(input_channels=3, num_classes=len(CLASS_NAMES))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"Model: {model.__class__.__name__}")
    print(f"Epochs: {EPOCHS}, Batch Size: {BATCH_SIZE}, LR: {LEARNING_RATE}")

    # --- 6. Training Loop ---
    print("[yellow]Starting model training...[/yellow]")
    for epoch in range(EPOCHS):
        model.train()  # Set model to training mode
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Print loss for this epoch
        print(
            f"Epoch [{epoch + 1}/{EPOCHS}], "
            f"Loss: {running_loss / len(train_loader):.4f}"
        )

    print("[bold green]Training complete.[/bold green]")

    # --- 7. Evaluation ---
    print("\n--- Model Evaluation ---")
    model.eval()  # Set model to evaluation mode
    all_preds = []
    all_true = []

    with torch.no_grad():  # Disable gradient calculation
        for inputs, labels in test_loader:
            outputs = model(inputs)
            # Get the index of the max log-probability (our prediction)
            _, predicted = torch.max(outputs.data, 1)

            all_preds.extend(predicted.numpy())
            all_true.extend(labels.numpy())

    # --- 8. Print Classification Report ---
    # This is the report we need for Part 2!
    print(classification_report(all_true, all_preds, target_names=CLASS_NAMES))

    # --- 9. Save Model ---
    print(f"Saving model to [cyan]{MODEL_SAVE_PATH}[/cyan]...")
    MODEL_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print("Model saved successfully.")
    print("DL training script finished.")


if __name__ == "__main__":
    # This allows running the script directly
    # (e.g., python src/train_dl.py)
    train_deep_learning_model()
