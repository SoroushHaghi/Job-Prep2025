# src/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNModel(nn.Module):
    """
    A simple 1D-CNN for activity recognition.
    """

    def __init__(self, input_channels, num_classes):
        super(CNNModel, self).__init__()

        # --- Convolutional Layers ---
        # Input shape: (batch_size, 3, 10)
        # (batch_size, num_channels, sequence_length)

        # Layer 1
        self.conv1 = nn.Conv1d(
            in_channels=input_channels, out_channels=64, kernel_size=3
        )
        # Shape after conv1: (batch_size, 64, 8)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        # Shape after pool1: (batch_size, 64, 4)

        # Layer 2
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3)
        # Shape after conv2: (batch_size, 128, 2)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        # Shape after pool2: (batch_size, 128, 1)

        # --- Fully Connected (Linear) Layers ---
        # Flatten the output from the conv layers
        # The size is 128 (from out_channels) * 1 (from final pool)
        self.fc1 = nn.Linear(in_features=128 * 1, out_features=128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(in_features=128, out_features=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the model.
        """
        # Conv block 1
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)

        # Conv block 2
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)

        # --- Flattening ---
        # .view(batch_size, -1) flattens all dims except batch
        x = x.view(x.size(0), -1)

        # --- FC block ---
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)

        # Output layer (raw logits)
        x = self.fc2(x)
        return x
