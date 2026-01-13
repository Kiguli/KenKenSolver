# -*- coding: utf-8 -*-
"""
Improved CNN architecture for unified Sudoku/HexaSudoku character recognition.

This model handles ALL puzzle types:
- Sudoku 4x4 and 9x9 (digits 1-9)
- HexaSudoku A-G (digits 0-9 + letters A-G)
- HexaSudoku numeric (digits 0-9 for two-digit numbers 10-16)

Output Classes (17 total):
- 0-9: Digits
- 10-16: Letters A-G (A=10, B=11, ..., G=16)

Key improvements over CNN_v2:
- 4 convolutional layers (vs 2) for more capacity
- BatchNorm after each conv for stable training
- 256 FC hidden units (vs 128)
- Dropout 0.4 (vs 0.25)
- ~850K parameters (vs ~400K)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ImprovedCNN(nn.Module):
    """
    Enhanced CNN with BatchNorm and deeper architecture.

    Architecture:
    - Block 1: 2 conv layers (1->32->32), BatchNorm, MaxPool (28->14)
    - Block 2: 2 conv layers (32->64->64), BatchNorm, MaxPool (14->7)
    - FC: 3136->256->output_dim

    Total params: ~850K
    """

    def __init__(self, output_dim=14):
        super(ImprovedCNN, self).__init__()

        # Block 1: 1 -> 32 channels
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)  # 28->14

        # Block 2: 32 -> 64 channels
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)  # 14->7

        # FC layers
        self.fc1 = nn.Linear(64 * 7 * 7, 256)  # 3136 -> 256
        self.bn_fc = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(0.4)
        self.fc2 = nn.Linear(256, output_dim)

    def forward(self, x):
        # Block 1
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)

        # Block 2
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # FC layers
        x = F.relu(self.bn_fc(self.fc1(x)))
        x = self.dropout(x)
        x = self.fc2(x)

        return x


def count_parameters(model):
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    # Test model
    print("ImprovedCNN Test")
    print("=" * 50)

    # Test with 17 classes (unified Sudoku/HexaSudoku)
    model = ImprovedCNN(output_dim=17)
    print(f"Parameters: {count_parameters(model):,}")

    # Test forward pass
    x = torch.randn(4, 1, 28, 28)
    out = model(x)
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {out.shape}")

    # Test predictions
    predictions = out.argmax(dim=1)
    print(f"Predictions: {predictions.tolist()}")

    # For Sudoku-only (digits 1-9), mask to first 10 classes
    digit_logits = out[:, :10]
    digit_predictions = digit_logits.argmax(dim=1)
    print(f"Digit-only predictions: {digit_predictions.tolist()}")
