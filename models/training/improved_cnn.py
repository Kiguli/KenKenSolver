# -*- coding: utf-8 -*-
"""
Improved CNN architecture for handwritten KenKen character recognition.

Key improvements over CNN_v2:
- 4 convolutional layers (vs 2) for more capacity
- BatchNorm after each conv for stable training
- 256 FC hidden units (vs 128)
- Dropout 0.4 (vs 0.25)
- ~800K parameters (vs ~400K)
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
    - FC: 3136->256->14

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


class ImprovedCNNWithAttention(nn.Module):
    """
    Enhanced CNN with channel attention (SE block).

    The squeeze-and-excitation block adaptively weights channels
    to focus on the most informative features.
    """

    def __init__(self, output_dim=14, reduction=4):
        super(ImprovedCNNWithAttention, self).__init__()

        # Block 1: 1 -> 32 channels
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)

        # SE block for block 1
        self.se1_fc1 = nn.Linear(32, 32 // reduction)
        self.se1_fc2 = nn.Linear(32 // reduction, 32)

        # Block 2: 32 -> 64 channels
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)

        # SE block for block 2
        self.se2_fc1 = nn.Linear(64, 64 // reduction)
        self.se2_fc2 = nn.Linear(64 // reduction, 64)

        # FC layers
        self.fc1 = nn.Linear(64 * 7 * 7, 256)
        self.bn_fc = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(0.4)
        self.fc2 = nn.Linear(256, output_dim)

    def _se_block(self, x, fc1, fc2):
        """Squeeze-and-Excitation block."""
        # Global average pooling
        b, c, h, w = x.size()
        squeeze = x.view(b, c, -1).mean(dim=2)

        # Excitation
        excite = F.relu(fc1(squeeze))
        excite = torch.sigmoid(fc2(excite))

        # Scale
        return x * excite.view(b, c, 1, 1)

    def forward(self, x):
        # Block 1
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self._se_block(x, self.se1_fc1, self.se1_fc2)
        x = self.pool1(x)

        # Block 2
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self._se_block(x, self.se2_fc1, self.se2_fc2)
        x = self.pool2(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # FC layers
        x = F.relu(self.bn_fc(self.fc1(x)))
        x = self.dropout(x)
        x = self.fc2(x)

        return x


# Keep Grid_CNN from original for size detection (unchanged)
class Grid_CNN(nn.Module):
    """Grid size detection model (unchanged from original)."""

    def __init__(self, output_dim=6):
        super(Grid_CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(262144, 128)
        self.fc2 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Also keep CNN_v2 for comparison
class CNN_v2(nn.Module):
    """Original CNN_v2 for comparison (baseline)."""

    def __init__(self, output_dim=14):
        super(CNN_v2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(3136, 128)  # 64 * 7 * 7 = 3136
        self.fc2 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def count_parameters(model):
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    # Test models
    print("Model Parameter Comparison")
    print("=" * 50)

    cnn_v2 = CNN_v2(output_dim=14)
    improved = ImprovedCNN(output_dim=14)
    improved_attn = ImprovedCNNWithAttention(output_dim=14)
    grid = Grid_CNN(output_dim=6)

    print(f"CNN_v2:                {count_parameters(cnn_v2):,} params")
    print(f"ImprovedCNN:           {count_parameters(improved):,} params")
    print(f"ImprovedCNN+Attention: {count_parameters(improved_attn):,} params")
    print(f"Grid_CNN:              {count_parameters(grid):,} params")

    # Test forward pass
    print("\nForward pass test:")
    x = torch.randn(4, 1, 28, 28)

    out_v2 = cnn_v2(x)
    print(f"CNN_v2 output shape: {out_v2.shape}")

    out_improved = improved(x)
    print(f"ImprovedCNN output shape: {out_improved.shape}")

    out_attn = improved_attn(x)
    print(f"ImprovedCNN+Attention output shape: {out_attn.shape}")
