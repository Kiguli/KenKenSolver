"""
Train CNN for Digit Recognition (MNIST-Only, Digits 0-9 + Empty).

This version trains a CNN with 11 classes:
- Classes 0-9: MNIST digits
- Class 10: Empty cell

Uses same CNN_v2 architecture and augmentation as other splits.
"""

import os
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from PIL import Image
from scipy.ndimage import rotate, shift, zoom

# Configuration
RANDOM_SEED = 9010
IMG_SIZE = 28
NUM_CLASSES = 11  # Digits 0-9 + empty
EPOCHS = 30
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EMPTY_SAMPLES = 2000
AUGMENTATION_FACTOR = 2

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


# =============================================================================
# CNN Model (same architecture as other splits)
# =============================================================================

class CNN_v2(nn.Module):
    """CNN for digit recognition."""

    def __init__(self, output_dim):
        super(CNN_v2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(3136, 128)
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

def augment_image(img, max_rotation=5, max_shift=2, max_scale=0.1):
    """Apply random augmentation to an image."""
    # Random rotation
    angle = np.random.uniform(-max_rotation, max_rotation)
    augmented = rotate(img, angle, reshape=False, mode='constant', cval=0)

    # Random shift
    shift_y = np.random.uniform(-max_shift, max_shift)
    shift_x = np.random.uniform(-max_shift, max_shift)
    augmented = shift(augmented, [shift_y, shift_x], mode='constant', cval=0)

    # Random scale
    scale = np.random.uniform(1 - max_scale, 1 + max_scale)
    h, w = augmented.shape
    scaled = zoom(augmented, scale, mode='constant', cval=0)

    # Crop/pad back to original size
    new_h, new_w = scaled.shape
    if new_h > h:
        start = (new_h - h) // 2
        scaled = scaled[start:start + h, :]
    elif new_h < h:
        pad = (h - new_h) // 2
        scaled = np.pad(scaled, ((pad, h - new_h - pad), (0, 0)), mode='constant')

    if new_w > w:
        start = (new_w - w) // 2
        scaled = scaled[:, start:start + w]
    elif new_w < w:
        pad = (w - new_w) // 2
        scaled = np.pad(scaled, ((0, 0), (pad, w - new_w - pad)), mode='constant')

    return np.clip(scaled, 0, 1)


def augment_dataset(images, labels, factor=2):
    """Augment dataset by a given factor."""
    print(f"Augmenting dataset with factor {factor}...")

    all_images = [images]
    all_labels = [labels]

    for i in range(factor):
        print(f"  Creating augmented set {i+1}/{factor}...")
        augmented = np.array([augment_image(img) for img in images])
        all_images.append(augmented)
        all_labels.append(labels)

    combined_images = np.concatenate(all_images, axis=0)
    combined_labels = np.concatenate(all_labels, axis=0)

    # Shuffle
    indices = np.random.permutation(len(combined_images))
    combined_images = combined_images[indices]
    combined_labels = combined_labels[indices]

    print(f"  Dataset size: {len(images)} -> {len(combined_images)}")

    return combined_images, combined_labels


# =============================================================================
# Empty Cell Generation
# =============================================================================

def generate_empty_samples(n_samples, img_size=28):
    """Generate synthetic empty cell samples."""
    print(f"Generating {n_samples} empty cell samples...")

    samples = []
    for _ in range(n_samples):
        # Mostly white with slight noise
        noise_level = np.random.uniform(0.0, 0.05)
        img = np.ones((img_size, img_size)) * (1 - noise_level)
        img += np.random.normal(0, 0.02, (img_size, img_size))
        img = np.clip(img, 0, 1)
        samples.append(img)

    return np.array(samples)


# =============================================================================
# Training
# =============================================================================

def train_model(train_images, train_labels, val_images, val_labels):
    """Train the CNN model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # Prepare data
    train_tensor = torch.tensor(train_images, dtype=torch.float32).unsqueeze(1)
    train_labels_tensor = torch.tensor(train_labels, dtype=torch.long)
    val_tensor = torch.tensor(val_images, dtype=torch.float32).unsqueeze(1)
    val_labels_tensor = torch.tensor(val_labels, dtype=torch.long)

    train_dataset = TensorDataset(train_tensor, train_labels_tensor)
    val_dataset = TensorDataset(val_tensor, val_labels_tensor)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Model
    model = CNN_v2(output_dim=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    print(f"\nTraining Configuration:")
    print(f"  Training samples: {len(train_images)}")
    print(f"  Validation samples: {len(val_images)}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    best_val_acc = 0
    best_model_state = None

    print("\n" + "-" * 60)
    for epoch in range(EPOCHS):
        # Training
        model.train()
        train_loss = 0
        correct = 0
        total = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        train_acc = correct / total

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()

        val_acc = val_correct / val_total
        scheduler.step(val_acc)

        marker = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            marker = " *"

        print(f"Epoch {epoch+1:2d}/{EPOCHS}: "
              f"Train Loss={train_loss/len(train_loader):.4f}, "
              f"Train Acc={train_acc:.4f}, "
              f"Val Acc={val_acc:.4f}{marker}")

    print("-" * 60)
    print(f"\nBest validation accuracy: {best_val_acc:.4f}")

    # Load best model
    model.load_state_dict(best_model_state)

    return model, best_val_acc


def evaluate_per_class(model, val_images, val_labels):
    """Evaluate per-class accuracy."""
    device = next(model.parameters()).device

    print("\nPer-class accuracy:")
    for cls in range(NUM_CLASSES):
        mask = val_labels == cls
        if mask.sum() == 0:
            continue

        cls_images = val_images[mask]
        cls_tensor = torch.tensor(cls_images, dtype=torch.float32).unsqueeze(1).to(device)

        with torch.no_grad():
            outputs = model(cls_tensor)
            _, predicted = outputs.max(1)

        correct = (predicted.cpu().numpy() == cls).sum()
        total = len(cls_images)
        acc = correct / total

        if cls == 10:
            name = "Empty"
        else:
            name = str(cls)

        print(f"  Class {cls:2d} ({name:5s}): {acc:.4f} ({correct}/{total})")


def main():
    print("=" * 60)
    print("Digit-Only CNN Training (MNIST 0-9 + Empty)")
    print("90-10 Split")
    print("=" * 60)

    # Load data
    print("\n1. Loading handwritten training data...")
    data_dir = "./data"

    if not os.path.exists(f"{data_dir}/train_images.npy"):
        print("Error: Training data not found. Run download_datasets.py first.")
        return

    train_images = np.load(f"{data_dir}/train_images.npy")
    train_labels = np.load(f"{data_dir}/train_labels.npy")

    print(f"Loaded {len(train_images)} handwritten samples")
    print(f"  Classes present: {sorted(np.unique(train_labels).tolist())}")

    # Generate empty samples (class 10)
    print("\n2. Generating empty cell samples...")
    empty_images = generate_empty_samples(EMPTY_SAMPLES)
    empty_labels = np.full(EMPTY_SAMPLES, 10)  # Class 10 = empty

    # Combine
    all_images = np.concatenate([train_images, empty_images], axis=0)
    all_labels = np.concatenate([train_labels, empty_labels], axis=0)
    print(f"   Total samples (with empty): {len(all_images)}")

    # Augment
    print("\n3. Applying data augmentation...")
    aug_images, aug_labels = augment_dataset(all_images, all_labels, AUGMENTATION_FACTOR)

    # Split for training/validation
    n_samples = len(aug_images)
    indices = np.random.permutation(n_samples)
    split = int(0.85 * n_samples)

    train_idx = indices[:split]
    val_idx = indices[split:]

    train_imgs = aug_images[train_idx]
    train_lbls = aug_labels[train_idx]
    val_imgs = aug_images[val_idx]
    val_lbls = aug_labels[val_idx]

    # Save sample for verification
    print("\n4. Saving training data sample...")
    os.makedirs("./symbols", exist_ok=True)
    sample_size = min(10000, len(train_imgs))
    sample_idx = np.random.choice(len(train_imgs), sample_size, replace=False)

    with open("./symbols/handwritten_digit_data.csv", "w", newline="") as f:
        writer = csv.writer(f)
        header = ["label"] + [f"pixel{i}" for i in range(784)]
        writer.writerow(header)
        for idx in sample_idx:
            row = [int(train_lbls[idx])] + (train_imgs[idx].flatten() * 255).astype(int).tolist()
            writer.writerow(row)
    print(f"Saved {sample_size} samples to ./symbols/handwritten_digit_data.csv")

    # Train
    print("\n5. Training CNN model...")
    model, best_acc = train_model(train_imgs, train_lbls, val_imgs, val_lbls)

    # Evaluate per-class
    print("\n6. Evaluating per-class accuracy...")
    evaluate_per_class(model, val_imgs, val_lbls)

    # Save model
    print("\n7. Saving model...")
    os.makedirs("./models", exist_ok=True)
    model_path = "./models/handwritten_digit_cnn.pth"
    torch.save(model.state_dict(), model_path)
    print(f"   Model saved to {model_path}")

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Run generate_images.py to create board images")
    print("  2. Run detect_errors.py to evaluate constraint-based correction")
    print("  3. Run predict_digits.py to evaluate top-K prediction correction")


if __name__ == '__main__':
    main()
