# -*- coding: utf-8 -*-
"""
Train unified 14-class CNN for KenKen character recognition.
Classes 0-9: Handwritten MNIST digits (inverted to match board convention)
Classes 10-13: Computer-generated operators (add, div, mul, sub)
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from PIL import Image
import random

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# CNN Architecture (same as KenKen CNN_v2)
class CNN_v2(nn.Module):
    def __init__(self, output_dim):
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
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def load_operator_templates():
    """Load operator PNG templates and create augmented samples."""
    operators_dir = './symbols/operators'
    operators = ['add', 'div', 'mul', 'sub']
    operator_samples = []
    operator_labels = []

    for idx, op in enumerate(operators):
        filepath = os.path.join(operators_dir, f'{op}.png')
        if not os.path.exists(filepath):
            print(f"Warning: {filepath} not found")
            continue

        # Load image - handle transparency properly
        img = Image.open(filepath)

        # If RGBA (transparent), composite on white background
        if img.mode == 'RGBA':
            background = Image.new('RGBA', img.size, (255, 255, 255, 255))
            img = Image.alpha_composite(background, img)

        # Convert to grayscale
        img = img.convert('L')
        img = img.resize((28, 28), Image.Resampling.LANCZOS)
        img_array = np.array(img, dtype=np.float32)

        # Normalize to 0-1 range
        # After compositing: ink=dark (~0), background=white (~255)
        # After normalization: ink≈0, background≈1
        img_array = img_array / 255.0

        print(f"  {op}: min={img_array.min():.2f}, max={img_array.max():.2f}")

        label = 10 + idx  # Labels 10-13 for operators

        # Create augmented samples (5000 per operator to match digit count)
        for _ in range(5000):
            augmented = augment_operator(img_array)
            operator_samples.append(augmented)
            operator_labels.append(label)

    return np.array(operator_samples), np.array(operator_labels)


def augment_operator(img):
    """Apply data augmentation to operator image."""
    from scipy import ndimage

    # Random rotation (-10 to 10 degrees)
    angle = random.uniform(-10, 10)
    rotated = ndimage.rotate(img, angle, reshape=False, mode='constant', cval=1.0)

    # Random shift (-2 to 2 pixels)
    shift_x = random.randint(-2, 2)
    shift_y = random.randint(-2, 2)
    shifted = ndimage.shift(rotated, [shift_y, shift_x], mode='constant', cval=1.0)

    # Random noise
    noise = np.random.normal(0, 0.02, shifted.shape)
    noisy = np.clip(shifted + noise, 0, 1)

    return noisy.astype(np.float32)


def prepare_data():
    """Load MNIST digits and operators, prepare for training."""
    print("Loading MNIST data...")

    # Load MNIST split
    train_images = np.load('./handwritten_data/train_images.npy')
    train_labels = np.load('./handwritten_data/train_labels.npy')

    print(f"MNIST training samples: {len(train_images)}")

    # Invert MNIST digits (MNIST: ink=HIGH, board convention: ink=LOW)
    # After inversion: background=1.0 (white), ink=0.0 (black)
    train_images = 255 - train_images  # Invert pixel values
    train_images = train_images.astype(np.float32) / 255.0  # Normalize to 0-1

    print("Loading and augmenting operator templates...")
    operator_images, operator_labels = load_operator_templates()
    print(f"Operator samples: {len(operator_images)}")

    # Combine digits and operators
    all_images = np.concatenate([train_images, operator_images])
    all_labels = np.concatenate([train_labels, operator_labels])

    print(f"Total training samples: {len(all_images)}")

    # Shuffle
    indices = np.arange(len(all_images))
    np.random.shuffle(indices)
    all_images = all_images[indices]
    all_labels = all_labels[indices]

    # Split 85/15 for train/val
    split_idx = int(len(all_images) * 0.85)
    train_imgs = all_images[:split_idx]
    train_lbls = all_labels[:split_idx]
    val_imgs = all_images[split_idx:]
    val_lbls = all_labels[split_idx:]

    print(f"Training set: {len(train_imgs)}")
    print(f"Validation set: {len(val_imgs)}")

    # Add channel dimension
    train_imgs = train_imgs[:, np.newaxis, :, :]
    val_imgs = val_imgs[:, np.newaxis, :, :]

    # Convert to tensors
    train_dataset = TensorDataset(
        torch.FloatTensor(train_imgs),
        torch.LongTensor(train_lbls)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(val_imgs),
        torch.LongTensor(val_lbls)
    )

    return train_dataset, val_dataset


def train_model(train_dataset, val_dataset, epochs=30, batch_size=64):
    """Train the CNN model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on: {device}")

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model
    model = CNN_v2(output_dim=14)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    best_val_acc = 0.0

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
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

        train_acc = 100.0 * train_correct / train_total

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_acc = 100.0 * val_correct / val_total

        print(f"Epoch {epoch+1:2d}/{epochs}: "
              f"Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%, "
              f"Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.2f}%")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), './models/unified_kenken_cnn.pth')
            print(f"  -> Saved best model (Val Acc: {val_acc:.2f}%)")

    print(f"\nTraining complete. Best validation accuracy: {best_val_acc:.2f}%")
    return model


def evaluate_by_class(model, val_dataset):
    """Evaluate model accuracy per class."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    class_correct = {i: 0 for i in range(14)}
    class_total = {i: 0 for i in range(14)}

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)

            for label, pred in zip(labels, predicted):
                class_total[label.item()] += 1
                if label == pred:
                    class_correct[label.item()] += 1

    print("\nPer-class accuracy:")
    class_names = {i: str(i) for i in range(10)}
    class_names.update({10: 'add', 11: 'div', 12: 'mul', 13: 'sub'})

    for cls in range(14):
        if class_total[cls] > 0:
            acc = 100.0 * class_correct[cls] / class_total[cls]
            print(f"  {class_names[cls]:>3}: {acc:.2f}% ({class_correct[cls]}/{class_total[cls]})")


if __name__ == '__main__':
    os.makedirs('./models', exist_ok=True)

    print("Preparing data...")
    train_dataset, val_dataset = prepare_data()

    print("\nTraining model...")
    model = train_model(train_dataset, val_dataset, epochs=30, batch_size=64)

    # Reload best model for evaluation
    model = CNN_v2(output_dim=14)
    model.load_state_dict(torch.load('./models/unified_kenken_cnn.pth', weights_only=True))

    evaluate_by_class(model, val_dataset)
