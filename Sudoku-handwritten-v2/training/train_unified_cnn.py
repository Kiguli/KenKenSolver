# -*- coding: utf-8 -*-
"""
Training script for unified ImprovedCNN character recognition model.

This model handles ALL Sudoku/HexaSudoku puzzle types:
- Sudoku 4x4 and 9x9 (digits 1-9)
- HexaSudoku A-G (digits 0-9 + letters A-G)
- HexaSudoku numeric (digits 0-9 for two-digit numbers 10-16)

Key improvements:
- Extracts training data from actual board images
- Uses advanced augmentation (elastic deformation, rotation, erosion/dilation)
- Class balancing via oversampling
- Focal loss for hard examples
- Tracks per-class accuracy and confusion matrix

Output Classes (17 total):
- 0-9: Digits
- 10-16: Letters A-G (A=10, B=11, ..., G=16)

Usage:
    python train_unified_cnn.py
"""

import sys
# Force unbuffered output
sys.stdout = sys.stderr

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import os
from pathlib import Path
from collections import Counter
import json

# Add parent directory to path for imports
script_dir = Path(__file__).parent
parent_dir = script_dir.parent
sys.path.insert(0, str(parent_dir / 'models'))
sys.path.insert(0, str(parent_dir / 'data'))

from improved_cnn import ImprovedCNN
from augmentation import augment_handwritten, augment_for_confusion_pairs


# =============================================================================
# Focal Loss for Hard Examples
# =============================================================================

class FocalLoss(nn.Module):
    """
    Focal loss for addressing class imbalance and hard examples.

    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)

    Args:
        alpha: Class weighting factor
        gamma: Focusing parameter (higher = more focus on hard examples)
    """

    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


# =============================================================================
# Dataset
# =============================================================================

class UnifiedCharacterDataset(Dataset):
    """Dataset for unified character recognition with augmentation."""

    def __init__(self, images, labels, augment=True, augmentation_level='medium',
                 augment_multiplier=10, confusion_aware=True):
        """
        Args:
            images: List of 28x28 numpy arrays (0-1 range, white bg)
            labels: List of integer labels (0-16)
            augment: Whether to apply augmentation
            augmentation_level: 'none', 'light', 'medium', 'heavy'
            augment_multiplier: How many augmented versions per original
            confusion_aware: Use confusion-pair aware augmentation
        """
        self.base_images = images
        self.base_labels = labels
        self.augment = augment
        self.augmentation_level = augmentation_level
        self.augment_multiplier = augment_multiplier if augment else 1
        self.confusion_aware = confusion_aware

    def __len__(self):
        return len(self.base_images) * self.augment_multiplier

    def __getitem__(self, idx):
        base_idx = idx % len(self.base_images)
        img = self.base_images[base_idx].copy()
        label = self.base_labels[base_idx]

        # Apply augmentation (except for first copy of each image)
        if self.augment and (idx >= len(self.base_images)):
            if self.confusion_aware:
                img = augment_for_confusion_pairs(img, label, self.augmentation_level)
            else:
                img = augment_handwritten(img, self.augmentation_level)

        # Convert to tensor
        img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
        return img_tensor, label


# =============================================================================
# Data Loading
# =============================================================================

def load_extracted_data(data_dir):
    """Load pre-extracted training data from numpy files."""
    images_path = os.path.join(data_dir, 'extracted_images.npy')
    labels_path = os.path.join(data_dir, 'extracted_labels.npy')

    if not os.path.exists(images_path):
        raise FileNotFoundError(f"Training data not found at {images_path}. "
                                "Run extract_training_data.py first.")

    images = np.load(images_path)
    labels = np.load(labels_path)

    print(f"Loaded {len(images)} samples from {data_dir}")

    return list(images), list(labels)


def get_label_name(label):
    """Get human-readable name for a label."""
    if label <= 9:
        return str(label)
    else:
        return chr(ord('A') + label - 10)


# =============================================================================
# Training
# =============================================================================

def train_model(train_loader, val_loader, model, criterion, optimizer, scheduler,
                num_epochs=100, device='cpu', patience=15):
    """Train model with early stopping."""
    best_val_loss = float('inf')
    best_val_acc = 0.0
    best_model_state = None
    epochs_no_improve = 0

    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            train_correct += predicted.eq(labels).sum().item()
            train_total += labels.size(0)

        train_loss /= train_total
        train_acc = train_correct / train_total * 100

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                val_correct += predicted.eq(labels).sum().item()
                val_total += labels.size(0)

        val_loss /= val_total
        val_acc = val_correct / val_total * 100

        # Record history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # Learning rate scheduling
        scheduler.step(val_loss)

        print(f'Epoch [{epoch+1}/{num_epochs}] '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.1f}% | '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.1f}%')

        # Early stopping check (based on accuracy, not loss)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f'\nRestored best model: Val Acc = {best_val_acc:.1f}%')

    return model, history


def evaluate_per_class(model, data_loader, num_classes=17, device='cpu'):
    """Evaluate accuracy per class and build confusion matrix."""
    model.eval()

    class_correct = {}
    class_total = {}
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = outputs.max(1)

            for label, pred in zip(labels.cpu().numpy(), predicted.cpu().numpy()):
                if label not in class_total:
                    class_total[label] = 0
                    class_correct[label] = 0

                class_total[label] += 1
                confusion_matrix[label, pred] += 1

                if label == pred:
                    class_correct[label] += 1

    print('\nPer-class accuracy:')
    for label in sorted(class_total.keys()):
        acc = class_correct[label] / class_total[label] * 100
        name = get_label_name(label)
        print(f'  {label:2d} ({name}): {acc:.1f}% ({class_correct[label]}/{class_total[label]})')

        # Show top misclassifications
        misclass = []
        for pred in range(num_classes):
            if pred != label and confusion_matrix[label, pred] > 0:
                misclass.append((get_label_name(pred), confusion_matrix[label, pred]))
        misclass.sort(key=lambda x: -x[1])
        if misclass:
            top_mis = ', '.join(f'{n}:{c}' for n, c in misclass[:3])
            print(f'       Misclassified as: {top_mis}')

    return confusion_matrix


# =============================================================================
# Main
# =============================================================================

def main():
    print('=' * 70)
    print('Unified ImprovedCNN Training')
    print('For Sudoku & HexaSudoku (all types)')
    print('=' * 70)
    print()

    # Configuration
    NUM_CLASSES = 17  # 0-9 digits + A-G letters
    BATCH_SIZE = 64
    NUM_EPOCHS = 100
    LEARNING_RATE = 0.0003
    AUGMENT_MULTIPLIER = 10
    PATIENCE = 15
    USE_FOCAL_LOSS = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Set seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # Load training data
    data_dir = parent_dir / 'data'
    images, labels = load_extracted_data(str(data_dir))

    print(f"\nTotal samples: {len(images)}")
    label_counts = Counter(labels)
    print(f"Label distribution:")
    for label in sorted(label_counts.keys()):
        name = get_label_name(label)
        print(f"   {label:2d} ({name}): {label_counts[label]:5d}")

    # Split into train/val (80/20)
    indices = list(range(len(images)))
    random.shuffle(indices)

    split_idx = int(0.8 * len(indices))
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]

    train_images = [images[i] for i in train_indices]
    train_labels = [labels[i] for i in train_indices]
    val_images = [images[i] for i in val_indices]
    val_labels = [labels[i] for i in val_indices]

    print(f'\nTrain: {len(train_images)} base samples')
    print(f'Val: {len(val_images)} base samples')

    # Create datasets
    train_dataset = UnifiedCharacterDataset(
        train_images, train_labels,
        augment=True,
        augmentation_level='medium',
        augment_multiplier=AUGMENT_MULTIPLIER,
        confusion_aware=True
    )
    val_dataset = UnifiedCharacterDataset(
        val_images, val_labels,
        augment=False
    )

    print(f'Train dataset size (with augmentation): {len(train_dataset)}')
    print(f'Val dataset size: {len(val_dataset)}')

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Create model
    model = ImprovedCNN(output_dim=NUM_CLASSES).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f'\nModel: ImprovedCNN')
    print(f'Parameters: {num_params:,}')
    print(f'Output classes: {NUM_CLASSES}')

    # Loss and optimizer
    if USE_FOCAL_LOSS:
        criterion = FocalLoss(alpha=1.0, gamma=2.0)
        print('Using Focal Loss')
    else:
        criterion = nn.CrossEntropyLoss()
        print('Using CrossEntropy Loss')

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    # Train
    print('\nTraining...')
    model, history = train_model(
        train_loader, val_loader, model, criterion, optimizer, scheduler,
        num_epochs=NUM_EPOCHS, device=device, patience=PATIENCE
    )

    # Evaluate
    print('\n' + '=' * 70)
    print('Final Evaluation')
    print('=' * 70)
    confusion_matrix = evaluate_per_class(model, val_loader, NUM_CLASSES, device)

    # Save model
    models_dir = parent_dir / 'models'
    models_dir.mkdir(exist_ok=True)

    save_path = models_dir / 'unified_cnn_weights.pth'
    torch.save(model.state_dict(), save_path)
    print(f'\nModel saved to {save_path}')

    # Save confusion matrix
    np.save(models_dir / 'confusion_matrix.npy', confusion_matrix)
    print(f'Confusion matrix saved to {models_dir / "confusion_matrix.npy"}')

    # Save training history
    history_path = models_dir / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f'Training history saved to {history_path}')

    print('\nDone!')


if __name__ == '__main__':
    main()
