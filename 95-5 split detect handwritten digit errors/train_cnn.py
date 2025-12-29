"""
Train CNN for handwritten HexaSudoku character recognition.

95-5 Split Configuration:
- Uses 5700 training samples per class (vs 5400 in 90-10 split)
- Uses 300 test samples per class (vs 600 in 90-10 split)
- Random seed 9505 (vs 9010 in 90-10 split) for different test set
- IDENTICAL augmentation to 90-10 split (only the data split differs)

Classes (17 total, matching HexaSudoku):
- 0: Empty cell
- 1-9: Digits 1-9
- 10-16: Letters A-G (representing values 10-16)

Datasets used:
- MNIST: http://yann.lecun.com/exdb/mnist/
- EMNIST-Letters: https://www.nist.gov/itl/products-and-services/emnist-dataset
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import csv
from scipy import ndimage

# Set random seed for reproducibility
# Using 9010 (same as 90-10 split) to compare with same test set composition
RANDOM_SEED = 9010
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# Constants
IMG_SIZE = 28
NUM_CLASSES = 17  # 0-9 digits + A-G letters + empty
EPOCHS = 30
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EMPTY_SAMPLES = 2000  # Number of empty cell samples to generate


class CNN_v2(nn.Module):
    """CNN for character recognition (same architecture as KenKen/Sudoku/HexaSudoku)."""

    def __init__(self, output_dim):
        super(CNN_v2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(3136, 128)  # 64 * 7 * 7 = 3136
        self.fc2 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 28x28 -> 14x14
        x = self.pool(F.relu(self.conv2(x)))  # 14x14 -> 7x7
        x = x.view(x.size(0), -1)             # Flatten to 3136
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def load_handwritten_data():
    """
    Load prepared handwritten training data from numpy files.

    Returns:
        images: numpy array of shape (N, 28, 28) with values in [0, 1]
        labels: numpy array of shape (N,) with class labels
    """
    data_dir = "./data"

    if not os.path.exists(f"{data_dir}/train_images.npy"):
        raise FileNotFoundError(
            f"Training data not found in {data_dir}/. "
            "Run download_datasets.py first."
        )

    images = np.load(f"{data_dir}/train_images.npy")
    labels = np.load(f"{data_dir}/train_labels.npy")

    print(f"Loaded {len(images)} handwritten samples")
    print(f"  Classes present: {sorted(np.unique(labels))}")

    return images, labels


def generate_empty_samples(num_samples=EMPTY_SAMPLES):
    """
    Generate empty cell samples (class 0).

    Creates nearly white images with slight noise to represent empty cells.
    The CNN learns to distinguish these from cells containing characters.

    Returns:
        images: numpy array of shape (num_samples, 28, 28)
        labels: numpy array of zeros
    """
    print(f"Generating {num_samples} empty cell samples...")

    # Empty cells are mostly white (high values) with slight noise
    images = np.random.uniform(0.95, 1.0, (num_samples, IMG_SIZE, IMG_SIZE))
    labels = np.zeros(num_samples, dtype=np.int64)

    return images, labels


def augment_image(img, max_rotation=5, max_shift=2, max_scale=0.1):
    """
    Apply random augmentation to an image.

    IDENTICAL to 90-10 split augmentation parameters.

    Args:
        img: 28x28 numpy array
        max_rotation: Maximum rotation in degrees (5)
        max_shift: Maximum translation in pixels (2)
        max_scale: Maximum scale variation (0.1 = ±10%)

    Returns:
        Augmented 28x28 numpy array
    """
    # Random rotation
    angle = np.random.uniform(-max_rotation, max_rotation)
    img = ndimage.rotate(img, angle, reshape=False, mode='constant', cval=0)

    # Random shift
    shift_x = np.random.uniform(-max_shift, max_shift)
    shift_y = np.random.uniform(-max_shift, max_shift)
    img = ndimage.shift(img, (shift_y, shift_x), mode='constant', cval=0)

    # Random scale (zoom)
    scale = np.random.uniform(1 - max_scale, 1 + max_scale)
    if scale != 1.0:
        img = ndimage.zoom(img, scale, mode='constant', cval=0)
        # Crop or pad to 28x28
        if img.shape[0] > IMG_SIZE:
            start = (img.shape[0] - IMG_SIZE) // 2
            img = img[start:start+IMG_SIZE, start:start+IMG_SIZE]
        elif img.shape[0] < IMG_SIZE:
            pad = (IMG_SIZE - img.shape[0]) // 2
            img = np.pad(img, ((pad, IMG_SIZE-img.shape[0]-pad),
                               (pad, IMG_SIZE-img.shape[1]-pad)),
                         mode='constant', constant_values=0)

    # Ensure valid range and size
    img = np.clip(img, 0, 1)
    if img.shape != (IMG_SIZE, IMG_SIZE):
        # Safety resize if needed
        from PIL import Image as PILImage
        img_pil = PILImage.fromarray((img * 255).astype(np.uint8))
        img_pil = img_pil.resize((IMG_SIZE, IMG_SIZE))
        img = np.array(img_pil) / 255.0

    return img


def augment_dataset(images, labels, augmentation_factor=2):
    """
    Augment the dataset by creating additional transformed samples.

    Args:
        images: Original images
        labels: Original labels
        augmentation_factor: How many augmented copies to create

    Returns:
        Augmented images and labels (includes originals)
    """
    print(f"Augmenting dataset with factor {augmentation_factor}...")

    augmented_images = [images]
    augmented_labels = [labels]

    for i in range(augmentation_factor):
        print(f"  Creating augmented set {i+1}/{augmentation_factor}...")
        aug_imgs = np.array([augment_image(img) for img in images])
        augmented_images.append(aug_imgs)
        augmented_labels.append(labels.copy())

    final_images = np.concatenate(augmented_images, axis=0)
    final_labels = np.concatenate(augmented_labels, axis=0)

    print(f"  Dataset size: {len(images)} -> {len(final_images)}")

    return final_images, final_labels


def save_training_data_csv(images, labels, output_path):
    """
    Save training data to CSV in TMNIST format for reference/debugging.

    Format: names, labels, pixel_1, pixel_2, ..., pixel_784
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Only save a subset to avoid huge files
    max_samples = min(10000, len(images))
    indices = np.random.choice(len(images), max_samples, replace=False)

    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)

        # Header
        header = ['names', 'labels'] + [str(i) for i in range(1, 785)]
        writer.writerow(header)

        # Data rows
        for idx in indices:
            img, label = images[idx], labels[idx]
            if label == 0:
                name = "Empty"
            elif label <= 9:
                name = str(label)
            else:
                name = chr(ord('A') + label - 10)

            pixels = (img.flatten() * 255).astype(int).tolist()
            row = [f'Handwritten-{name}', label] + pixels
            writer.writerow(row)

    print(f"Saved {max_samples} samples to {output_path}")


def train_model(images, labels):
    """
    Train CNN on the prepared data.

    Args:
        images: numpy array of shape (N, 28, 28)
        labels: numpy array of shape (N,)

    Returns:
        Trained model
    """
    # Shuffle and split into train/validation
    n_samples = len(images)
    indices = np.random.permutation(n_samples)
    train_size = int(0.85 * n_samples)

    train_idx = indices[:train_size]
    val_idx = indices[train_size:]

    X_train = images[train_idx]
    y_train = labels[train_idx]
    X_val = images[val_idx]
    y_val = labels[val_idx]

    # Convert to tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).unsqueeze(1)
    y_val_t = torch.tensor(y_val, dtype=torch.long)

    # Create data loaders
    train_dataset = TensorDataset(X_train_t, y_train_t)
    val_dataset = TensorDataset(X_val_t, y_val_t)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    # Initialize model
    model = CNN_v2(output_dim=NUM_CLASSES)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )

    print(f"\nTraining Configuration:")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Validation samples: {len(X_val)}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    print("\n" + "-" * 60)

    # Training loop
    best_val_acc = 0
    best_model_state = None

    for epoch in range(EPOCHS):
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0

        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_correct += (outputs.argmax(1) == y_batch).sum().item()

        train_acc = train_correct / len(X_train)

        # Validation phase
        model.eval()
        val_correct = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                outputs = model(X_batch)
                val_correct += (outputs.argmax(1) == y_batch).sum().item()

        val_acc = val_correct / len(X_val)

        # Update scheduler
        scheduler.step(val_acc)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()

        print(f"Epoch {epoch+1:2d}/{EPOCHS}: "
              f"Train Loss={train_loss/len(train_loader):.4f}, "
              f"Train Acc={train_acc:.4f}, "
              f"Val Acc={val_acc:.4f}"
              f"{' *' if val_acc == best_val_acc else ''}")

    print("-" * 60)
    print(f"\nBest validation accuracy: {best_val_acc:.4f}")

    # Load best model weights
    model.load_state_dict(best_model_state)

    return model


def evaluate_per_class(model, images, labels):
    """Evaluate model accuracy per class."""
    model.eval()

    # Convert to tensors
    X = torch.tensor(images, dtype=torch.float32).unsqueeze(1)

    # Get predictions
    with torch.no_grad():
        outputs = model(X)
        predictions = outputs.argmax(1).numpy()

    print("\nPer-class accuracy:")
    for cls in sorted(np.unique(labels)):
        mask = labels == cls
        cls_correct = np.sum(predictions[mask] == labels[mask])
        cls_total = np.sum(mask)
        cls_acc = cls_correct / cls_total if cls_total > 0 else 0

        if cls == 0:
            name = "Empty"
        elif cls <= 9:
            name = str(cls)
        else:
            name = chr(ord('A') + cls - 10)

        print(f"  Class {cls:2d} ({name:5s}): {cls_acc:.4f} ({cls_correct}/{cls_total})")


def main():
    """Main training pipeline."""
    print("=" * 60)
    print("Handwritten HexaSudoku CNN Training")
    print("95-5 Split (Identical augmentation to 90-10)")
    print("=" * 60)

    # Load handwritten data
    print("\n1. Loading handwritten training data...")
    images, labels = load_handwritten_data()

    # Generate empty cell samples
    print("\n2. Generating empty cell samples...")
    empty_images, empty_labels = generate_empty_samples()

    # Combine with handwritten data
    images = np.concatenate([images, empty_images], axis=0)
    labels = np.concatenate([labels, empty_labels], axis=0)
    print(f"   Total samples (with empty): {len(images)}")

    # Apply data augmentation
    print("\n3. Applying data augmentation...")
    images, labels = augment_dataset(images, labels, augmentation_factor=2)

    # Shuffle the combined dataset
    shuffle_idx = np.random.permutation(len(images))
    images = images[shuffle_idx]
    labels = labels[shuffle_idx]

    # Save training data sample to CSV (for debugging/visualization)
    print("\n4. Saving training data sample...")
    csv_path = "./symbols/handwritten_hex_data.csv"
    os.makedirs("./symbols", exist_ok=True)
    save_training_data_csv(images, labels, csv_path)

    # Train model
    print("\n5. Training CNN model...")
    model = train_model(images, labels)

    # Evaluate per-class accuracy
    print("\n6. Evaluating per-class accuracy...")
    # Use a subset for evaluation
    eval_idx = np.random.choice(len(images), min(5000, len(images)), replace=False)
    evaluate_per_class(model, images[eval_idx], labels[eval_idx])

    # Save model
    print("\n7. Saving model...")
    model_dir = "./models"
    os.makedirs(model_dir, exist_ok=True)
    model_path = f"{model_dir}/handwritten_hex_cnn.pth"
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
