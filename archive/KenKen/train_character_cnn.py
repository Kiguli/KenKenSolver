"""
Training script for CNN_v2 character recognition model.

Includes data augmentation to improve recognition of smaller characters
(e.g., from 9x9 KenKen boards).

Usage:
    python train_character_cnn.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from PIL import Image, ImageFilter
import random
import os
from pathlib import Path


# =============================================================================
# Model Architecture
# =============================================================================

class CNN_v2(nn.Module):
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


# =============================================================================
# Data Augmentation
# =============================================================================

def scale_and_center(img_array, target_size=28, scale_factor=1.0):
    """Scale image and center in target_size canvas."""
    img = Image.fromarray((img_array * 255).astype(np.uint8), mode='L')

    # Scale the image
    new_size = int(28 * scale_factor)
    if new_size < 10:
        new_size = 10
    if new_size > 28:
        new_size = 28

    scaled = img.resize((new_size, new_size), Image.Resampling.LANCZOS)

    # Center on target canvas
    canvas = Image.new('L', (target_size, target_size), color=255)
    offset = (target_size - new_size) // 2
    canvas.paste(scaled, (offset, offset))

    return np.array(canvas).astype(np.float32) / 255.0


def add_gaussian_noise(img_array, sigma=0.05):
    """Add Gaussian noise to image."""
    noise = np.random.normal(0, sigma, img_array.shape)
    noisy = img_array + noise
    return np.clip(noisy, 0, 1).astype(np.float32)


def add_salt_pepper_noise(img_array, amount=0.02):
    """Add salt and pepper noise."""
    noisy = img_array.copy()
    # Salt
    num_salt = int(amount * img_array.size * 0.5)
    salt_coords = [np.random.randint(0, i, num_salt) for i in img_array.shape]
    noisy[salt_coords[0], salt_coords[1]] = 1.0
    # Pepper
    num_pepper = int(amount * img_array.size * 0.5)
    pepper_coords = [np.random.randint(0, i, num_pepper) for i in img_array.shape]
    noisy[pepper_coords[0], pepper_coords[1]] = 0.0
    return noisy.astype(np.float32)


def apply_blur(img_array, radius=0.5):
    """Apply Gaussian blur."""
    img = Image.fromarray((img_array * 255).astype(np.uint8), mode='L')
    blurred = img.filter(ImageFilter.GaussianBlur(radius=radius))
    return np.array(blurred).astype(np.float32) / 255.0


def translate_image(img_array, dx=0, dy=0):
    """Translate image by (dx, dy) pixels. Assumes white background (high values)."""
    result = np.ones_like(img_array)  # White background (1.0)
    h, w = img_array.shape

    # Calculate source and destination regions
    src_x1 = max(0, -dx)
    src_y1 = max(0, -dy)
    src_x2 = min(w, w - dx)
    src_y2 = min(h, h - dy)

    dst_x1 = max(0, dx)
    dst_y1 = max(0, dy)
    dst_x2 = min(w, w + dx)
    dst_y2 = min(h, h + dy)

    result[dst_y1:dst_y2, dst_x1:dst_x2] = img_array[src_y1:src_y2, src_x1:src_x2]
    return result


def augment_image(img_array, augmentation_level='medium'):
    """Apply random augmentations to simulate 9x9 extraction artifacts."""
    img = img_array.copy()

    if augmentation_level == 'none':
        return img

    # Scale variations (simulate smaller characters from 9x9 boards)
    if random.random() < 0.7:
        if augmentation_level == 'heavy':
            scale = random.uniform(0.6, 1.0)
        else:
            scale = random.uniform(0.75, 1.0)
        img = scale_and_center(img, scale_factor=scale)

    # Position jitter
    if random.random() < 0.5:
        dx = random.randint(-2, 2)
        dy = random.randint(-2, 2)
        img = translate_image(img, dx, dy)

    # Blur (simulate downsampling artifacts)
    if random.random() < 0.4:
        radius = random.uniform(0.3, 0.8)
        img = apply_blur(img, radius)

    # Noise
    if random.random() < 0.3:
        if random.random() < 0.5:
            img = add_gaussian_noise(img, sigma=random.uniform(0.02, 0.08))
        else:
            img = add_salt_pepper_noise(img, amount=random.uniform(0.01, 0.03))

    return img


# =============================================================================
# Dataset
# =============================================================================

class CharacterDataset(Dataset):
    def __init__(self, images, labels, augment=True, augmentation_level='medium',
                 augment_multiplier=10):
        """
        Args:
            images: List of 28x28 numpy arrays
            labels: List of integer labels
            augment: Whether to apply augmentation
            augmentation_level: 'none', 'light', 'medium', 'heavy'
            augment_multiplier: How many augmented versions per original image
        """
        self.base_images = images
        self.base_labels = labels
        self.augment = augment
        self.augmentation_level = augmentation_level
        self.augment_multiplier = augment_multiplier if augment else 1

    def __len__(self):
        return len(self.base_images) * self.augment_multiplier

    def __getitem__(self, idx):
        base_idx = idx % len(self.base_images)
        img = self.base_images[base_idx].copy()
        label = self.base_labels[base_idx]

        # Apply augmentation (except for first copy of each image)
        if self.augment and (idx >= len(self.base_images)):
            img = augment_image(img, self.augmentation_level)

        # Invert: TMNIST has white on black, but we work with black on white
        # After inversion, we want: 0 = white background, 1 = black text
        # The CSV has: 0 = background, 255 = character
        # We normalize to 0-1, so: 0 = background, 1 = character
        # For the model, we need: high values = character pixels

        img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
        return img_tensor, label


def load_tmnist_data(csv_path):
    """Load TMNIST data from CSV."""
    df = pd.read_csv(csv_path)

    images = []
    labels = []

    for _, row in df.iterrows():
        label = int(row['labels'])
        # Get pixel values (columns 1 onwards, after 'names' and 'labels')
        pixels = row.iloc[2:].values.astype(np.float32)

        # Reshape to 28x28
        img = pixels.reshape(28, 28)

        # Normalize to 0-1 (0 = background, 1 = character)
        img = img / 255.0

        images.append(img)
        labels.append(label)

    return images, labels


def extract_training_from_boards(board_dir='./board_images', puzzles_path='./puzzles/puzzles_dict.json',
                                   sizes=[3, 4, 5, 6, 7], num_puzzles_per_size=50):
    """Extract character images from actual board images using ground truth labels."""
    import json
    import cv2 as cv

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
        num_puzzles = min(num_puzzles_per_size, len(puzzles))

        for puzzle_idx in range(num_puzzles):
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
                cell_size = 900 // size
                border_thickness = 16
                row, col = start_cell

                # Calculate cell region (top portion where numbers are)
                height_factor = 0.5 if size <= 6 else (0.6 if size == 7 else 0.7)
                y_start = row * cell_size + border_thickness + 5
                y_end = int(row * cell_size + cell_size * height_factor)
                x_start = col * cell_size + border_thickness + 5
                x_end = (col + 1) * cell_size - border_thickness

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
                boxes = sorted(boxes, key=lambda b: b[0])

                # Merge overlapping boxes
                merged_boxes = []
                for box in boxes:
                    x, y, w, h = box
                    if merged_boxes and x < merged_boxes[-1][0] + merged_boxes[-1][2]:
                        # Merge with previous box
                        px, py, pw, ph = merged_boxes[-1]
                        new_x = min(px, x)
                        new_y = min(py, y)
                        new_w = max(px + pw, x + w) - new_x
                        new_h = max(py + ph, y + h) - new_y
                        merged_boxes[-1] = (new_x, new_y, new_w, new_h)
                    else:
                        merged_boxes.append(box)

                # Extract and resize each character
                if len(merged_boxes) != len(digit_labels):
                    continue  # Skip if mismatch

                for box, label in zip(merged_boxes, digit_labels):
                    x, y, w, h = box
                    char_img = cell_img[y:y+h, x:x+w]

                    if char_img.size == 0:
                        continue

                    # Resize to 28x28 with aspect ratio preservation
                    char_pil = Image.fromarray((char_img * 255).astype(np.uint8))
                    aspect = w / h
                    if aspect > 1:
                        new_w = 28
                        new_h = int(28 / aspect)
                    else:
                        new_h = 28
                        new_w = int(28 * aspect)

                    if new_w < 1:
                        new_w = 1
                    if new_h < 1:
                        new_h = 1

                    resized = char_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)

                    # Center on 28x28 canvas
                    canvas = Image.new('L', (28, 28), color=255)
                    paste_x = (28 - new_w) // 2
                    paste_y = (28 - new_h) // 2
                    canvas.paste(resized, (paste_x, paste_y))

                    # Convert to array - keep same convention as inference
                    # White background (high values), dark characters (low values)
                    # This matches what get_character() produces without inversion
                    char_array = np.array(canvas).astype(np.float32) / 255.0
                    # DO NOT invert - keep consistent with inference pipeline

                    images.append(char_array)
                    labels.append(label)

    print(f"Extracted {len(images)} character samples from board images")
    return images, labels


# =============================================================================
# Training
# =============================================================================

def train_model(train_loader, val_loader, model, criterion, optimizer, scheduler,
                num_epochs=50, device='cpu', patience=10):
    """Train the model with early stopping."""
    best_val_loss = float('inf')
    best_model_state = None
    epochs_no_improve = 0

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

        # Learning rate scheduling
        scheduler.step(val_loss)

        print(f'Epoch [{epoch+1}/{num_epochs}] '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.1f}% | '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.1f}%')

        # Early stopping check
        if val_loss < best_val_loss:
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

    return model


def evaluate_per_class(model, data_loader, device='cpu'):
    """Evaluate accuracy per class."""
    model.eval()

    class_correct = {}
    class_total = {}
    confusion = {}  # Track misclassifications

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
                    confusion[label] = {}

                class_total[label] += 1
                if label == pred:
                    class_correct[label] += 1
                else:
                    if pred not in confusion[label]:
                        confusion[label][pred] = 0
                    confusion[label][pred] += 1

    print('\nPer-class accuracy:')
    for label in sorted(class_total.keys()):
        acc = class_correct[label] / class_total[label] * 100
        name = label_names.get(label, str(label))
        print(f'  {name}: {acc:.1f}% ({class_correct[label]}/{class_total[label]})')
        if confusion[label]:
            misclass = ', '.join(f'{label_names.get(k, str(k))}:{v}'
                                for k, v in sorted(confusion[label].items(),
                                                  key=lambda x: -x[1])[:3])
            print(f'      Misclassified as: {misclass}')


def main():
    print('=' * 70)
    print('CNN_v2 Character Recognition Training')
    print('With augmentation for 9x9 KenKen character sizes')
    print('=' * 70)
    print()

    # Configuration
    BATCH_SIZE = 32
    NUM_EPOCHS = 50
    LEARNING_RATE = 0.001
    AUGMENT_MULTIPLIER = 10  # Generate 10x augmented samples
    PATIENCE = 10

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Change to script directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)

    # Extract training data from actual board images
    print('\nExtracting training data from board images...')
    images, labels = extract_training_from_boards(
        board_dir='./board_images',
        puzzles_path='./puzzles/puzzles_dict.json',
        sizes=[3, 4, 5, 6, 7, 9],  # Include all available sizes
        num_puzzles_per_size=100
    )
    print(f'Extracted {len(images)} base samples')

    # Count per class
    from collections import Counter
    label_counts = Counter(labels)
    print(f'Classes: {dict(sorted(label_counts.items()))}')

    # Split into train/val (80/20)
    indices = list(range(len(images)))
    random.seed(42)
    random.shuffle(indices)

    split_idx = int(0.8 * len(indices))
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]

    train_images = [images[i] for i in train_indices]
    train_labels = [labels[i] for i in train_indices]
    val_images = [images[i] for i in val_indices]
    val_labels = [labels[i] for i in val_indices]

    print(f'Train: {len(train_images)} base samples')
    print(f'Val: {len(val_images)} base samples')

    # Create datasets with augmentation
    train_dataset = CharacterDataset(
        train_images, train_labels,
        augment=True,
        augmentation_level='medium',
        augment_multiplier=AUGMENT_MULTIPLIER
    )
    val_dataset = CharacterDataset(
        val_images, val_labels,
        augment=False
    )

    print(f'Train dataset size (with augmentation): {len(train_dataset)}')
    print(f'Val dataset size: {len(val_dataset)}')

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Create model
    model = CNN_v2(output_dim=14).to(device)
    print(f'\nModel parameters: {sum(p.numel() for p in model.parameters()):,}')

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    # Train
    print('\nTraining...')
    model = train_model(
        train_loader, val_loader, model, criterion, optimizer, scheduler,
        num_epochs=NUM_EPOCHS, device=device, patience=PATIENCE
    )

    # Evaluate per class
    print('\n' + '=' * 70)
    print('Final Evaluation')
    print('=' * 70)
    evaluate_per_class(model, val_loader, device)

    # Save model
    save_path = './models/character_recognition_v2_model_weights.pth'
    torch.save(model.state_dict(), save_path)
    print(f'\nModel saved to {save_path}')

    print('\nDone!')


if __name__ == '__main__':
    main()
