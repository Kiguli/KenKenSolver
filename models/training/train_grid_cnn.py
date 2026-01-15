"""
Train Grid_CNN for KenKen board size detection.

Supports sizes 3-9 (output_dim=7).
Maps: output 0-6 -> sizes 3, 4, 5, 6, 7, 8, 9

Usage:
    python train_grid_cnn.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import random
from pathlib import Path

# Configuration
BOARD_SIZES = [3, 4, 5, 6, 7, 8, 9]  # Now includes size 8
SIZE_TO_LABEL = {3: 0, 4: 1, 5: 2, 6: 3, 7: 4, 8: 5, 9: 6}
LABEL_TO_SIZE = {v: k for k, v in SIZE_TO_LABEL.items()}
OUTPUT_DIM = len(BOARD_SIZES)

BATCH_SIZE = 16
EPOCHS = 30
LEARNING_RATE = 0.001
TRAIN_SPLIT = 0.8


class Grid_CNN(nn.Module):
    """CNN for grid size detection."""
    def __init__(self, output_dim):
        super(Grid_CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(262144, 128)
        self.fc2 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class BoardImageDataset(Dataset):
    """Dataset of KenKen board images with size labels."""

    def __init__(self, image_dir, sizes=BOARD_SIZES, transform=None, indices=None):
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.samples = []

        for size in sizes:
            # Support subfolder structure (e.g., 8x8/board8_0.png)
            size_dir = self.image_dir / f"{size}x{size}"
            if size_dir.exists():
                pattern = f"board{size}_"
                images = sorted([f for f in os.listdir(size_dir) if f.startswith(pattern)])
                for img_name in images:
                    idx = int(img_name.replace(pattern, "").replace(".png", ""))
                    if indices is None or idx in indices:
                        self.samples.append((size_dir / img_name, SIZE_TO_LABEL[size]))
            else:
                # Fallback to flat structure (e.g., board3_0.png)
                pattern = f"board{size}_"
                if self.image_dir.exists():
                    images = sorted([f for f in os.listdir(self.image_dir) if f.startswith(pattern)])
                    for img_name in images:
                        idx = int(img_name.replace(pattern, "").replace(".png", ""))
                        if indices is None or idx in indices:
                            self.samples.append((self.image_dir / img_name, SIZE_TO_LABEL[size]))

        print(f"Loaded {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGBA")

        if self.transform:
            image = self.transform(image)

        return image, label


def train_model(model, train_loader, val_loader, device, epochs=EPOCHS):
    """Train the Grid_CNN model."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_acc = 0
    best_model_state = None

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

        train_acc = 100 * train_correct / train_total

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_acc = 100 * val_correct / val_total

        print(f"Epoch {epoch+1}/{epochs}: "
              f"Train Loss: {train_loss/len(train_loader):.4f}, "
              f"Train Acc: {train_acc:.2f}%, "
              f"Val Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()

    # Load best model
    if best_model_state:
        model.load_state_dict(best_model_state)

    return model, best_val_acc


def main():
    print("=" * 60)
    print("Grid_CNN Training - KenKen Size Detection")
    print(f"Sizes: {BOARD_SIZES} (output_dim={OUTPUT_DIM})")
    print("=" * 60)
    print()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Change to script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    # Image transforms
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize(128),
        transforms.CenterCrop(128),
        transforms.ToTensor(),
    ])

    # Split indices for train/val
    all_indices = list(range(100))
    random.seed(42)
    random.shuffle(all_indices)
    split_idx = int(len(all_indices) * TRAIN_SPLIT)
    train_indices = set(all_indices[:split_idx])
    val_indices = set(all_indices[split_idx:])

    print(f"Train samples per size: {len(train_indices)}")
    print(f"Val samples per size: {len(val_indices)}")
    print()

    # Create datasets - use benchmarks directory (supports subfolder structure)
    base_dir = Path(script_dir).parent.parent  # models/training -> root
    image_dir = base_dir / "benchmarks" / "KenKen" / "Handwritten"
    print(f"Using image directory: {image_dir}")
    train_dataset = BoardImageDataset(image_dir, transform=transform, indices=train_indices)
    val_dataset = BoardImageDataset(image_dir, transform=transform, indices=val_indices)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Create model
    model = Grid_CNN(output_dim=OUTPUT_DIM).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()

    # Train
    print("Training...")
    model, best_acc = train_model(model, train_loader, val_loader, device)
    print()
    print(f"Best validation accuracy: {best_acc:.2f}%")

    # Save model to grid_detection directory
    grid_detection_dir = base_dir / "models" / "grid_detection"
    grid_detection_dir.mkdir(parents=True, exist_ok=True)
    model_path = grid_detection_dir / "kenken_grid_cnn.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    # Test final accuracy per size
    print()
    print("Final per-size accuracy:")
    model.eval()

    with torch.no_grad():
        for size in BOARD_SIZES:
            correct = 0
            total = 0
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                # Filter to this size only
                mask = labels == SIZE_TO_LABEL[size]
                if mask.sum() == 0:
                    continue
                outputs = model(images[mask])
                _, predicted = outputs.max(1)
                total += mask.sum().item()
                correct += predicted.eq(labels[mask]).sum().item()
            if total > 0:
                print(f"  {size}x{size}: {correct}/{total} ({100*correct/total:.1f}%)")

    print()
    print("Done!")


if __name__ == "__main__":
    main()
