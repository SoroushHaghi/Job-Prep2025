# src/train_dl.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from rich import print
import numpy as np

import json

import matplotlib.pyplot as plt

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
LOSS_CURVE_PATH = PROJECT_ROOT / "docs" / "cnn_loss_curves.png"
ACCURACY_CURVE_PATH = PROJECT_ROOT / "docs" / "cnn_accuracy_curves.png"
RESULTS_FILE = PROJECT_ROOT / "models" / "cnn_results.json"

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
    print(
        "Splitting data into training, validation, and testing sets (70/15/15 split)..."
    )
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    print(f"Training samples: {X_train.shape[0]}")
    print(f"Validation samples: {X_val.shape[0]}")
    print(f"Testing samples: {X_test.shape[0]}")

    # --- 4. Create PyTorch DataLoaders ---
    # Convert numpy arrays to PyTorch Tensors
    train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    val_dataset = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))
    test_dataset = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
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
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(EPOCHS):
        model.train()  # Set model to training mode
        running_loss = 0.0
        correct_train = 0
        total_train = 0

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
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = correct_train / total_train
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)

        # Validation loop
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        val_loss /= len(val_loader)
        val_acc = correct_val / total_val
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch [{epoch + 1}/{EPOCHS}], "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
        )

    print("[bold green]Training complete.[/bold green]")

    # --- 7. Plotting Training History ---
    print("\nGenerating training history plots...")
    # Plot Loss
    plt.figure(figsize=(10, 5))
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Validation Loss")
    plt.title("Loss vs. Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(LOSS_CURVE_PATH)
    plt.close()
    print(f"Loss curve plot saved to [cyan]{LOSS_CURVE_PATH}[/cyan]")

    # Plot Accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(history["train_acc"], label="Train Accuracy")
    plt.plot(history["val_acc"], label="Validation Accuracy")
    plt.title("Accuracy vs. Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(ACCURACY_CURVE_PATH)
    plt.close()
    print(f"Accuracy curve plot saved to [cyan]{ACCURACY_CURVE_PATH}[/cyan]")

    # --- 8. Evaluation ---
    print("\n--- Model Evaluation on Test Set ---")
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

    # --- 9. Print and Save Classification Report ---
    report = classification_report(all_true, all_preds, target_names=CLASS_NAMES)
    report_dict = classification_report(
        all_true, all_preds, target_names=CLASS_NAMES, output_dict=True
    )
    print(report)

    with open(RESULTS_FILE, "w") as f:
        json.dump(report_dict, f, indent=4)
    print(f"Classification report saved to {RESULTS_FILE}")

    # --- 10. Save Model ---
    print(f"\nSaving model to [cyan]{MODEL_SAVE_PATH}[/cyan]...")
    MODEL_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print("Model saved successfully.")
    print("DL training script finished.")


if __name__ == "__main__":
    # This allows running the script directly
    # (e.g., python src/train_dl.py)
    train_deep_learning_model()
