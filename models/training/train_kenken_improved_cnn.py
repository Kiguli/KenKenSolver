# -*- coding: utf-8 -*-
"""
Training script for ImprovedCNN character recognition model.

Key improvements:
- Extracts training data from actual board images
- Uses advanced augmentation (elastic deformation, rotation, erosion/dilation)
- Combines handwritten and computer-generated data
- Class balancing via oversampling
- Focal loss for hard examples
- Tracks per-class accuracy and confusion matrix

Updated to include 8x8 puzzles and use current benchmarks directory structure.

Usage:
    python train_kenken_improved_cnn.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import os
import sys
from pathlib import Path
from collections import Counter
import json

# Add parent directory to path for imports
script_dir = Path(__file__).parent
parent_dir = script_dir.parent
kenken_dir = parent_dir.parent  # KenKenSolver root
sys.path.insert(0, str(script_dir))

from improved_cnn import ImprovedCNN, ImprovedCNNWithAttention, CNN_v2
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

class HandwrittenCharacterDataset(Dataset):
    """Dataset for handwritten character recognition with augmentation."""

    def __init__(self, images, labels, augment=True, augmentation_level='medium',
                 augment_multiplier=10, confusion_aware=False):
        """
        Args:
            images: List of 28x28 numpy arrays (0-1 range, white bg)
            labels: List of integer labels (0-9 digits, 10-13 operators)
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

def extract_training_from_boards(board_dir, puzzles_path, sizes, num_per_size=100,
                                  cell_size=300, border_thickness=16):
    """
    Extract character images from board images using ground truth labels.

    Uses the same extraction pipeline as inference to ensure consistency.
    Supports both flat (board{size}_{idx}.png) and subfolder ({size}x{size}/board{size}_{idx}.png) structures.
    """
    import cv2 as cv
    from PIL import Image

    with open(puzzles_path, 'r') as f:
        puzzles_dict = json.load(f)

    images = []
    labels = []

    # Operator mapping
    op_to_label = {'add': 10, 'div': 11, 'mul': 12, 'sub': 13}

    for size in sizes:
        if str(size) not in puzzles_dict:
            continue

        puzzles = puzzles_dict[str(size)]
        num_puzzles = min(num_per_size, len(puzzles))

        for puzzle_idx in range(num_puzzles):
            # Support both flat and subfolder directory structures
            board_path = os.path.join(board_dir, f'{size}x{size}', f'board{size}_{puzzle_idx}.png')
            if not os.path.exists(board_path):
                # Fallback to flat structure
                board_path = os.path.join(board_dir, f'board{size}_{puzzle_idx}.png')
                if not os.path.exists(board_path):
                    continue

            # Load board image
            img = Image.open(board_path).convert('L')
            grid = np.array(img)

            # Get puzzle cages
            puzzle = puzzles[puzzle_idx]
            if isinstance(puzzle, dict) and 'cages' in puzzle:
                cages = puzzle['cages']
            else:
                cages = puzzle

            # Process each cage
            for cage in cages:
                target = cage['target']
                op = cage['op']
                start_cell = cage['cells'][0]

                # Find top-left cell
                for cell in cage['cells']:
                    if cell[0] < start_cell[0] or (cell[0] == start_cell[0] and cell[1] < start_cell[1]):
                        start_cell = cell

                # Extract digits from target
                digit_labels = []
                temp = int(target)
                if temp == 0:
                    digit_labels = [0]
                else:
                    while temp > 0:
                        digit_labels.append(temp % 10)
                        temp //= 10
                    digit_labels = digit_labels[::-1]

                # Add operator if present
                if op and op in op_to_label:
                    digit_labels.append(op_to_label[op])

                # Extract characters from cell
                row, col = start_cell

                # Calculate cell region (top portion where numbers are)
                height_factor = 0.5 if size <= 6 else (0.6 if size == 7 else 0.7)
                y_start = row * cell_size + border_thickness + 5
                y_end = int(row * cell_size + cell_size * height_factor)
                x_start = col * cell_size + border_thickness + 5
                x_end = (col + 1) * cell_size - border_thickness

                if y_end > grid.shape[0] or x_end > grid.shape[1]:
                    continue

                cell_img = grid[y_start:y_end, x_start:x_end]
                cell_img = (cell_img / 255.0).astype('float64')

                # Find character contours
                img_uint8 = (cell_img * 255).astype(np.uint8)
                _, thresh = cv.threshold(img_uint8, 127, 255, cv.THRESH_BINARY_INV)
                kernel = np.ones((2, 2), np.uint8)
                thresh = cv.dilate(thresh, kernel, iterations=1)
                contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

                if len(contours) == 0:
                    continue

                # Sort contours left to right
                boxes = [cv.boundingRect(c) for c in contours]
                boxes = [b for b in boxes if b[2] >= 5 and b[3] >= 5]  # Filter small noise
                boxes = sorted(boxes, key=lambda b: b[0])

                if len(boxes) != len(digit_labels):
                    continue  # Skip if mismatch

                for box, label in zip(boxes, digit_labels):
                    x, y, w, h = box
                    char_img = cell_img[y:y+h, x:x+w]

                    if char_img.size == 0:
                        continue

                    # Resize to 28x28 with aspect ratio preservation
                    char_pil = Image.fromarray((char_img * 255).astype(np.uint8))
                    aspect = w / h
                    if aspect > 1:
                        new_w = 28
                        new_h = max(1, int(28 / aspect))
                    else:
                        new_h = 28
                        new_w = max(1, int(28 * aspect))

                    resized = char_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)

                    # Center on 28x28 canvas
                    canvas = Image.new('L', (28, 28), color=255)
                    paste_x = (28 - new_w) // 2
                    paste_y = (28 - new_h) // 2
                    canvas.paste(resized, (paste_x, paste_y))

                    char_array = np.array(canvas).astype(np.float32) / 255.0
                    images.append(char_array)
                    labels.append(label)

    return images, labels


def balance_classes(images, labels, target_per_class=None):
    """Balance dataset by oversampling minority classes."""
    label_counts = Counter(labels)

    if target_per_class is None:
        target_per_class = int(np.median(list(label_counts.values())))

    # Group by label
    by_label = {label: [] for label in range(14)}
    for img, lbl in zip(images, labels):
        by_label[lbl].append(img)

    # Oversample/undersample
    balanced_images = []
    balanced_labels = []

    for label in range(14):
        class_images = by_label[label]
        count = len(class_images)

        if count == 0:
            continue

        if count >= target_per_class:
            # Undersample
            selected = random.sample(class_images, target_per_class)
        else:
            # Oversample
            selected = class_images.copy()
            while len(selected) < target_per_class:
                selected.append(random.choice(class_images))

        balanced_images.extend(selected)
        balanced_labels.extend([label] * len(selected))

    # Shuffle
    combined = list(zip(balanced_images, balanced_labels))
    random.shuffle(combined)
    balanced_images, balanced_labels = zip(*combined)

    return list(balanced_images), list(balanced_labels)


def load_training_data(handwritten_ratio=0.7, num_per_size=100, balance=True):
    """Load training data from both handwritten and computer-generated boards."""
    sizes = [3, 4, 5, 6, 7, 8, 9]  # Include size 8

    print("Loading training data...")
    print("=" * 50)

    all_images = []
    all_labels = []

    # Try handwritten boards first (from benchmarks directory)
    print("\n1. Handwritten boards:")
    handwritten_dir = kenken_dir / 'benchmarks' / 'KenKen' / 'Handwritten'
    puzzles_path = kenken_dir / 'puzzles' / 'kenken_puzzles.json'

    if handwritten_dir.exists() and puzzles_path.exists():
        hw_images, hw_labels = extract_training_from_boards(
            str(handwritten_dir), str(puzzles_path), sizes, num_per_size
        )
        print(f"   Extracted {len(hw_images)} handwritten samples")
        all_images.extend(hw_images)
        all_labels.extend(hw_labels)
    else:
        print(f"   Warning: Handwritten directory not found at {handwritten_dir}")

    # Try computer-generated boards (from benchmarks directory)
    print("\n2. Computer-generated boards:")
    computer_dir = kenken_dir / 'benchmarks' / 'KenKen' / 'Computer'

    if computer_dir.exists() and puzzles_path.exists():
        cg_images, cg_labels = extract_training_from_boards(
            str(computer_dir), str(puzzles_path), sizes, num_per_size
        )
        print(f"   Extracted {len(cg_images)} computer-generated samples")

        # Subsample to achieve desired ratio
        if all_images and cg_images:
            target_cg = int(len(all_images) * (1 - handwritten_ratio) / handwritten_ratio)
            if len(cg_images) > target_cg:
                indices = random.sample(range(len(cg_images)), target_cg)
                cg_images = [cg_images[i] for i in indices]
                cg_labels = [cg_labels[i] for i in indices]

        all_images.extend(cg_images)
        all_labels.extend(cg_labels)
    else:
        print(f"   Warning: Computer-generated directory not found at {computer_dir}")

    if not all_images:
        raise ValueError("No training data found!")

    print(f"\n3. Total samples: {len(all_images)}")
    print(f"   Label distribution: {dict(sorted(Counter(all_labels).items()))}")

    # Balance classes
    if balance:
        print("\n4. Balancing classes...")
        all_images, all_labels = balance_classes(all_images, all_labels)
        print(f"   Balanced to: {len(all_images)} samples")
        print(f"   New distribution: {dict(sorted(Counter(all_labels).items()))}")

    return all_images, all_labels


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


def evaluate_per_class(model, data_loader, device='cpu'):
    """Evaluate accuracy per class and build confusion matrix."""
    model.eval()

    class_correct = {}
    class_total = {}
    confusion_matrix = np.zeros((14, 14), dtype=int)

    label_names = {
        0: '0', 1: '1', 2: '2', 3: '3', 4: '4',
        5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
        10: '+', 11: '/', 12: '*', 13: '-'
    }

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
        name = label_names.get(label, str(label))
        print(f'  {name}: {acc:.1f}% ({class_correct[label]}/{class_total[label]})')

        # Show top misclassifications
        misclass = []
        for pred in range(14):
            if pred != label and confusion_matrix[label, pred] > 0:
                misclass.append((label_names.get(pred, str(pred)), confusion_matrix[label, pred]))
        misclass.sort(key=lambda x: -x[1])
        if misclass:
            top_mis = ', '.join(f'{n}:{c}' for n, c in misclass[:3])
            print(f'      Misclassified as: {top_mis}')

    return confusion_matrix


# =============================================================================
# Main
# =============================================================================

def main():
    print('=' * 70)
    print('ImprovedCNN Character Recognition Training')
    print('For KenKen-handwritten-v2 (with 8x8 support)')
    print('=' * 70)
    print()

    # Configuration
    BATCH_SIZE = 64
    NUM_EPOCHS = 100
    LEARNING_RATE = 0.0003
    AUGMENT_MULTIPLIER = 10
    PATIENCE = 15
    USE_FOCAL_LOSS = True
    USE_ATTENTION = False  # Set to True for attention model

    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Set seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # Load training data
    images, labels = load_training_data(
        handwritten_ratio=0.7,
        num_per_size=100,
        balance=True
    )

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
    train_dataset = HandwrittenCharacterDataset(
        train_images, train_labels,
        augment=True,
        augmentation_level='medium',
        augment_multiplier=AUGMENT_MULTIPLIER,
        confusion_aware=True
    )
    val_dataset = HandwrittenCharacterDataset(
        val_images, val_labels,
        augment=False
    )

    print(f'Train dataset size (with augmentation): {len(train_dataset)}')
    print(f'Val dataset size: {len(val_dataset)}')

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Create model
    if USE_ATTENTION:
        model = ImprovedCNNWithAttention(output_dim=14).to(device)
        model_name = 'ImprovedCNN+Attention'
    else:
        model = ImprovedCNN(output_dim=14).to(device)
        model_name = 'ImprovedCNN'

    num_params = sum(p.numel() for p in model.parameters())
    print(f'\nModel: {model_name}')
    print(f'Parameters: {num_params:,}')

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
    confusion_matrix = evaluate_per_class(model, val_loader, device)

    # Save model to handwritten_v2 directory (used by solve_handwritten_v2.py)
    handwritten_v2_dir = parent_dir / 'handwritten_v2'
    handwritten_v2_dir.mkdir(exist_ok=True)

    save_path = handwritten_v2_dir / 'kenken_improved_cnn.pth'
    torch.save(model.state_dict(), save_path)
    print(f'\nModel saved to {save_path}')

    # Save confusion matrix
    np.save(handwritten_v2_dir / 'confusion_matrix.npy', confusion_matrix)
    print(f'Confusion matrix saved to {handwritten_v2_dir / "confusion_matrix.npy"}')

    # Save training history
    history_path = handwritten_v2_dir / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f'Training history saved to {history_path}')

    print('\nDone!')


if __name__ == '__main__':
    main()
